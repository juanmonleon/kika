"""
Read access to the built benchmarks SQLite database.

The database is produced by :func:`kika.benchmarks.build_benchmarks_db` from a
user's own DICE data. This module only reads it. Connection handling mirrors
``kika.exfor.database.X4ProDatabase`` (lazy connection, ``sqlite3.Row`` factory,
parameterized queries).
"""

import json
import os
import sqlite3
from typing import List, Optional

from kika.benchmarks import config
from kika.benchmarks._blob import unpack_f32
from kika.benchmarks._constants import SCHEMA_VERSION
from kika.sensitivities.profile import SensitivityProfile, SensitivityReaction
from kika.benchmarks.exceptions import (
    BenchmarkNotFoundError,
    BenchmarksError,
    DatabaseNotConfiguredError,
)

# Whitelist mapping energy region -> indexed integral-sensitivity column. Used to
# build screening SQL safely (never interpolate user input into column names).
_REGION_COLUMN = {
    "thermal": "s_thermal",
    "epithermal": "s_epithermal",
    "fast": "s_fast",
    "total": "s_total",
}


class BenchmarksDatabase:
    """Interface to a built ``kika_benchmarks.db`` SQLite database.

    It is built by hand and works as a context manager that closes the
    connection on exit. The connection opens at the first query, not in the
    constructor. That first query raises
    :class:`~kika.benchmarks.DatabaseNotConfiguredError` when the file does not
    exist, and :class:`~kika.benchmarks.BenchmarksError` when it has no ``meta``
    table or its schema version or sensitivity-uncertainty convention is not the
    one this version of kika writes.

    Parameters
    ----------
    db_path : str, optional
        The database file. None resolves it as
        :func:`~kika.benchmarks.get_db_path` does: the configured path, then
        ``KIKA_BENCHMARKS_DB_PATH``, then ``~/.kika/benchmarks/kika_benchmarks.db``.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = config.get_db_path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # -- connection lifecycle ------------------------------------------------
    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.db_path or not os.path.isfile(self.db_path):
                raise DatabaseNotConfiguredError(
                    f"Benchmarks database not found at {self.db_path!r}. Build it with "
                    "kika.benchmarks.build_benchmarks_db(...) or set the path via "
                    "benchmarks.configure(db_path=...) / KIKA_BENCHMARKS_DB_PATH."
                )
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                meta = {
                    row["key"]: row["value"]
                    for row in conn.execute("SELECT key, value FROM meta")
                }
            except sqlite3.Error as exc:
                conn.close()
                raise BenchmarksError(
                    "Invalid benchmarks database; rebuild it with "
                    "kika.benchmarks.build_benchmarks_db(...)."
                ) from exc
            if (
                meta.get("schema_version") != str(SCHEMA_VERSION)
                or meta.get("sensitivity_uncertainty_convention") != "absolute"
            ):
                conn.close()
                raise BenchmarksError(
                    f"Benchmarks database schema/convention is incompatible "
                    f"(expected schema_version={SCHEMA_VERSION} and absolute "
                    "sensitivity uncertainties); rebuild the database."
                )
            self._conn = conn
        return self._conn

    def close(self) -> None:
        """Close the connection; a later query opens it again."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "BenchmarksDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- metadata ------------------------------------------------------------
    def get_statistics(self) -> dict:
        """Return counts and ingest metadata for the database."""
        conn = self._get_connection()
        stats = {
            "benchmarks": conn.execute("SELECT COUNT(*) FROM benchmarks").fetchone()[0],
            "profiles": conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0],
            "reactions": conn.execute(
                "SELECT COUNT(*) FROM profile_sensitivities"
            ).fetchone()[0],
        }
        meta = {
            row["key"]: row["value"]
            for row in conn.execute("SELECT key, value FROM meta")
        }
        stats["meta"] = meta
        return stats

    # -- catalog -------------------------------------------------------------
    def list_benchmarks(
        self,
        category: Optional[str] = None,
        spectrum: Optional[str] = None,
        library: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[dict]:
        """List benchmarks with optional filters (one row per benchmark).

        Parameters
        ----------
        category, spectrum
            Exact match on the benchmark's category (``"HEU"``) or spectrum
            (``"THERM"``), the first and third fields of its id. None or an empty
            string applies no filter.
        library
            Keep the benchmarks that have at least one profile of this nuclear
            data library (exact match). None or an empty string applies no filter.
        search
            A SQL ``LIKE`` pattern matched against the id anywhere in it, so
            ``%`` and ``_`` in it are wildcards. None or an empty string applies
            no filter.
        limit, offset
            Page of the result, taken after the rows are ordered by
            ``benchmark_id``.

        Returns
        -------
        list of dict
            One dict per benchmark, ordered by ``benchmark_id``, with the keys
            ``benchmark_id``, ``category``, ``btype``, ``spectrum``, ``series``,
            ``case_number`` (``None`` for an id that does not follow
            ``CATEGORY-TYPE-SPECTRUM-SERIES-CASE``) and ``n_variants`` (the
            number of profiles stored for it).
        """
        conn = self._get_connection()
        conditions, params = [], []
        if category:
            conditions.append("b.category = ?")
            params.append(category)
        if spectrum:
            conditions.append("b.spectrum = ?")
            params.append(spectrum)
        if search:
            conditions.append("b.benchmark_id LIKE ?")
            params.append(f"%{search}%")
        if library:
            conditions.append(
                "EXISTS (SELECT 1 FROM profiles p WHERE p.benchmark_id = b.benchmark_id "
                "AND p.library = ?)"
            )
            params.append(library)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = (
            "SELECT b.benchmark_id, b.category, b.btype, b.spectrum, b.series, "
            "b.case_number, "
            "(SELECT COUNT(*) FROM profiles p WHERE p.benchmark_id = b.benchmark_id) "
            "AS n_variants "
            f"FROM benchmarks b {where} ORDER BY b.benchmark_id LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        return [dict(r) for r in conn.execute(query, params)]

    def get_benchmark(self, benchmark_id: str) -> dict:
        """Return a benchmark's metadata plus the list of its profile variants.

        Parameters
        ----------
        benchmark_id
            The id as stored, e.g. ``"HEU-COMP-THERM-021-079"``.

        Returns
        -------
        dict
            The stored row of the benchmark (``benchmark_id``, ``category``,
            ``btype``, ``spectrum``, ``series``, ``case_number``, ``keff_exp``,
            ``keff_exp_unc``; the last two are ``None`` without an experimental
            value, see :meth:`get_experimental_keff`) and ``profiles``: a list
            of dicts, the preferred profile first and then by ``profile_id``,
            each with ``profile_id``, ``code``, ``library``,
            ``group_structure``, ``ngroups``, ``keff`` (the response of the
            sensitivity file, ``None`` for an ABBN table), ``keff_unc`` (its
            absolute one-sigma uncertainty: 0.0 when the file's k-eff line has no
            ``+/-``, ``None`` for an ABBN table),
            ``is_preferred`` (1 or 0) and ``source_filename``.

        Raises
        ------
        ~kika.benchmarks.BenchmarkNotFoundError
            If the id is not in the database.
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM benchmarks WHERE benchmark_id = ?", (benchmark_id,)
        ).fetchone()
        if row is None:
            raise BenchmarkNotFoundError(f"Unknown benchmark id: {benchmark_id!r}")
        result = dict(row)
        result["profiles"] = [
            dict(p)
            for p in conn.execute(
                "SELECT profile_id, code, library, group_structure, ngroups, keff, "
                "keff_unc, is_preferred, source_filename "
                "FROM profiles WHERE benchmark_id = ? "
                "ORDER BY is_preferred DESC, profile_id",
                (benchmark_id,),
            )
        ]
        return result

    def get_preferred_profile(self, benchmark_id: str) -> dict:
        """Return the preferred profile row for a benchmark.

        The preferred profile is the one flagged when the database was built:
        the lowest priority index of transport code (KENO before MCNP), nuclear
        data library (newest ENDF/B first, ABBN-93 last) and group structure
        (238-group, 299-group, continuous), then of the file's variant number.

        Parameters
        ----------
        benchmark_id
            The id as stored.

        Returns
        -------
        dict
            The stored ``profiles`` row: ``profile_id``, ``benchmark_id``,
            ``code``, ``library``, ``group_structure``, ``ngroups``,
            ``variant_index``, ``keff``, ``keff_unc`` (absolute one-sigma; 0.0
            when the file's k-eff line has no ``+/-``, ``None`` for an ABBN
            table),
            ``pert_energies`` (the compressed energy boundaries as stored, not
            decoded), ``is_preferred`` and ``source_filename``.

        Raises
        ------
        ~kika.benchmarks.BenchmarkNotFoundError
            If the benchmark has no profile (or is not in the database).
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM profiles WHERE benchmark_id = ? "
            "ORDER BY is_preferred DESC, profile_id LIMIT 1",
            (benchmark_id,),
        ).fetchone()
        if row is None:
            raise BenchmarkNotFoundError(
                f"No profiles for benchmark id: {benchmark_id!r}"
            )
        return dict(row)

    # -- vectors (for plotting / UQ) ----------------------------------------
    def list_similarity_profiles(
        self,
        benchmark_ids: Optional[List[str]] = None,
        preferred_only: bool = True,
    ) -> List[dict]:
        """Return deterministic profile candidates for c-k ranking.

        Parameters
        ----------
        benchmark_ids
            Restrict to these benchmarks. None keeps all of them; an empty list
            returns an empty list.
        preferred_only
            True keeps only the preferred profile of each benchmark; False keeps
            every profile.

        Returns
        -------
        list of dict
            Dicts with ``profile_id``, ``benchmark_id``, ``code``, ``library``,
            ``group_structure`` and ``is_preferred``, ordered by
            ``benchmark_id``, then preferred first, then ``profile_id``.
        """
        conditions = []
        params: list = []
        if preferred_only:
            conditions.append("is_preferred = 1")
        if benchmark_ids is not None:
            if not benchmark_ids:
                return []
            placeholders = ", ".join("?" for _ in benchmark_ids)
            conditions.append(f"benchmark_id IN ({placeholders})")
            params.extend(benchmark_ids)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._get_connection().execute(
            "SELECT profile_id, benchmark_id, code, library, group_structure, "
            f"is_preferred FROM profiles {where} "
            "ORDER BY benchmark_id, is_preferred DESC, profile_id",
            params,
        )
        return [dict(row) for row in rows]

    def get_profile_vector(
        self, profile_id: int, zaid: Optional[int] = None, mt: Optional[int] = None
    ) -> dict:
        """
        Return a profile's full per-group sensitivity vectors.

        Each reaction error vector contains absolute one-sigma standard
        deviations, or None when errors were not stored.
        The returned dict matches the app's ``SDFSensitivityData`` shape so the
        existing SDF plotting components can render it directly::

            {"pert_energies": [...], "reactions": [
                {"zaid", "mt", "nuclide", "reaction_name", "sensitivity": [...],
                 "error": [...] | None}, ...]}

        Parameters
        ----------
        profile_id
            The ``profile_id`` of a profile of the database.
        zaid, mt
            Keep only the reactions with this ZAID (1000 Z + A) or this ENDF MT.
            None applies no filter.

        Returns
        -------
        dict
            ``pert_energies``: the ``ngroups + 1`` group boundaries in MeV,
            increasing. ``reactions``: one dict per reaction of the system-total
            profile (the per-mixture sub-profiles are not returned), ordered by
            ZAID and MT, with ``zaid``, ``mt``, ``nuclide``, ``reaction_name``
            (as the sensitivity file names them), ``sensitivity`` and
            ``error``. ``sensitivity`` holds ``ngroups`` values in the order of
            increasing energy: the dimensionless relative sensitivity
            coefficient of the profile's response (k-eff) in each group, per
            group and not per unit lethargy. ``error`` holds
            the absolute one-sigma uncertainty of each sensitivity coefficient,
            in the same unit as the coefficient (not a fraction of it). It is
            ``None`` when the database was built without errors and for the
            30-group ABBN tables, which have none. A profile of the compact
            TSUNAMI-1D dialect has no uncertainties either, but the build stores
            an all-zero vector for it, so ``error`` is zeros, not ``None``.
            Values are stored as float32.

        Raises
        ------
        ~kika.benchmarks.BenchmarkNotFoundError
            If ``profile_id`` is not in the database.
        """
        conn = self._get_connection()
        prof = conn.execute(
            "SELECT ngroups, pert_energies FROM profiles WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
        if prof is None:
            raise BenchmarkNotFoundError(f"Unknown profile_id: {profile_id}")
        ngroups = prof["ngroups"]
        pert_energies = unpack_f32(prof["pert_energies"], ngroups + 1).tolist()

        # Only the system-total profiles (unit, region) == (0, 0); per-mixture
        # sub-profiles (when ingested) are excluded from the plotting vector.
        conditions = ["profile_id = ?", "unit = 0", "region = 0"]
        params = [profile_id]
        if zaid is not None:
            conditions.append("zaid = ?")
            params.append(zaid)
        if mt is not None:
            conditions.append("mt = ?")
            params.append(mt)
        query = (
            "SELECT zaid, mt, nuclide, reaction, sensitivity, error "
            f"FROM profile_sensitivities WHERE {' AND '.join(conditions)} "
            "ORDER BY zaid, mt"
        )
        reactions = []
        for r in conn.execute(query, params):
            reactions.append(
                {
                    "zaid": r["zaid"],
                    "mt": r["mt"],
                    "nuclide": r["nuclide"],
                    "reaction_name": r["reaction"],
                    "sensitivity": unpack_f32(r["sensitivity"], ngroups).tolist(),
                    "error": (
                        unpack_f32(r["error"], ngroups).tolist()
                        if r["error"] is not None
                        else None
                    ),
                }
            )
        return {"pert_energies": pert_energies, "reactions": reactions}

    def get_sensitivity_profile(self, profile_id: int) -> SensitivityProfile:
        """Return a profile as a validated, format-neutral UQ object.

        Parameters
        ----------
        profile_id
            The ``profile_id`` of a profile of the database.

        Returns
        -------
        ~kika.sensitivities.SensitivityProfile
            Built from the system-total reactions of
            :meth:`get_profile_vector`: ``energy_grid`` the group boundaries in
            MeV (``energy_unit="MeV"``); each reaction's ``sensitivity`` a
            dimensionless relative coefficient per group and ``uncertainty`` its
            absolute one-sigma uncertainty (``None`` when not stored and for ABBN
            tables; all zeros for the compact TSUNAMI-1D dialect, which has none);
            ``response`` the profile's ``keff`` and ``response_uncertainty`` its
            ``keff_unc`` (absolute one-sigma; 0.0 when the file's k-eff line has
            no ``+/-``), both ``None`` for an ABBN table; ``label`` the benchmark id; ``metadata`` with ``source``
            (``"benchmarks-db"``), ``profile_id``, ``benchmark_id``, ``code``,
            ``library`` and ``group_structure``.

        Raises
        ------
        ~kika.benchmarks.BenchmarkNotFoundError
            If ``profile_id`` is not in the database.
        ValueError
            If the stored vectors do not satisfy the profile's checks (a
            non-finite value, a negative uncertainty, boundaries that do not
            increase).
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT benchmark_id, code, library, group_structure, ngroups, "
            "keff, keff_unc, pert_energies FROM profiles WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
        if row is None:
            raise BenchmarkNotFoundError(f"Unknown profile_id: {profile_id}")

        vector = self.get_profile_vector(profile_id)
        reactions = tuple(
            SensitivityReaction(
                zaid=reaction["zaid"],
                mt=reaction["mt"],
                sensitivity=reaction["sensitivity"],
                uncertainty=reaction["error"],
                label=reaction["reaction_name"],
            )
            for reaction in vector["reactions"]
        )
        return SensitivityProfile(
            energy_grid=vector["pert_energies"],
            energy_unit="MeV",
            reactions=reactions,
            response=row["keff"],
            response_uncertainty=row["keff_unc"],
            label=row["benchmark_id"],
            metadata={
                "source": "benchmarks-db",
                "profile_id": profile_id,
                "benchmark_id": row["benchmark_id"],
                "code": row["code"],
                "library": row["library"],
                "group_structure": row["group_structure"],
            },
        )

    # -- screening -----------------------------------------------------------
    def screen(
        self,
        zaid: int,
        mt: Optional[int] = None,
        region: str = "total",
        sensitivity_threshold: float = 0.0,
        fraction_threshold: Optional[float] = None,
        preferred_only: bool = True,
        limit: int = 100,
    ) -> List[dict]:
        """
        Return benchmark reactions sensitive to ``(zaid, mt)`` in an energy region,
        ranked by absolute integral sensitivity (descending).

        The integral sensitivities are not computed here: the build stored, for
        each reaction, the sum of the per-group sensitivity coefficients over
        the groups of each region (and over all groups, and the sum of their
        absolute values). A group belongs to a region by its upper energy edge:
        thermal up to 0.625 eV, epithermal above that up to 100 keV, fast above
        100 keV. Only the system-total profiles are searched, never the
        per-mixture sub-profiles. With the default threshold of 0, a reaction whose
        integral in the region is exactly zero does not match.

        Parameters
        ----------
        zaid
            The nuclide as ``1000 Z + A`` (26056 for Fe-56).
        mt
            ENDF MT of the reaction. None matches every reaction of the nuclide
            except MT 0 (the pseudo-total).
        region : {"thermal", "epithermal", "fast", "total"}
            The energy region whose integral is ranked; ``"total"`` is the sum over
            all groups.
        sensitivity_threshold
            Keep a reaction only when ``abs(s_region)`` is strictly above this,
            in the unit of the sensitivity (a dimensionless relative coefficient
            summed over the groups).
        fraction_threshold
            When not None, also keep a reaction only when ``abs(s_region)`` is
            strictly above this fraction of ``s_abs_total``, the sum of the
            absolute group sensitivities of the same reaction over all groups.
            A fraction, not a percentage. A reaction whose ``s_abs_total`` is
            not positive is then dropped.
        preferred_only
            True searches only the preferred profile of each benchmark; False
            searches every profile, so a benchmark can appear once per profile.
        limit
            Maximum number of rows, taken after sorting.

        Returns
        -------
        list of dict
            One dict per matching (profile, reaction), sorted by
            ``abs(s_region)`` from largest to smallest (the order of equal
            values is not defined). The keys are ``benchmark_id``, ``category``,
            ``spectrum``, ``btype``, ``profile_id``, ``code``, ``library``,
            ``group_structure``, ``keff`` (the profile's response), ``zaid``,
            ``mt``, ``nuclide``, ``reaction``, ``s_region`` (the integral
            sensitivity in the chosen region, signed), ``s_total`` (the signed
            sum over all groups), ``s_abs_total`` (the sum of the absolute group
            values) and ``fraction``, ``abs(s_region) / s_abs_total`` as a
            fraction (0.0 when ``s_abs_total`` is zero).

        Raises
        ------
        ValueError
            If ``region`` is not one of the four above.
        """
        if region not in _REGION_COLUMN:
            raise ValueError(
                f"Invalid region {region!r}; expected one of {tuple(_REGION_COLUMN)}"
            )
        col = _REGION_COLUMN[region]
        conn = self._get_connection()

        # Screen only against system-total profiles; per-mixture sub-profiles
        # (when ingested) never participate in screening.
        conditions = ["ps.zaid = ?", "ps.unit = 0", "ps.region = 0"]
        params: list = [zaid]
        if mt is not None:
            conditions.append("ps.mt = ?")
            params.append(mt)
        else:
            conditions.append("ps.mt != 0")
        if preferred_only:
            conditions.append("p.is_preferred = 1")
        conditions.append(f"ABS(ps.{col}) > ?")
        params.append(sensitivity_threshold)
        if fraction_threshold is not None:
            conditions.append(
                f"ps.s_abs_total > 0 AND ABS(ps.{col}) / ps.s_abs_total > ?"
            )
            params.append(fraction_threshold)

        query = (
            "SELECT b.benchmark_id, b.category, b.spectrum, b.btype, "
            "p.profile_id, p.code, p.library, p.group_structure, p.keff, "
            "ps.zaid, ps.mt, ps.nuclide, ps.reaction, "
            f"ps.{col} AS s_region, ps.s_total, ps.s_abs_total "
            "FROM profile_sensitivities ps "
            "JOIN profiles p ON p.profile_id = ps.profile_id "
            "JOIN benchmarks b ON b.benchmark_id = p.benchmark_id "
            f"WHERE {' AND '.join(conditions)} "
            f"ORDER BY ABS(ps.{col}) DESC LIMIT ?"
        )
        params.append(limit)

        results = []
        for r in conn.execute(query, params):
            d = dict(r)
            d["fraction"] = (
                abs(d["s_region"]) / d["s_abs_total"] if d["s_abs_total"] else 0.0
            )
            results.append(d)
        return results

    # -- auxiliary per-benchmark data ----------------------------------------
    def get_experimental_keff(self, benchmark_id: str) -> dict:
        """Return ``{'keff', 'keff_unc'}`` (the ICSBEP benchmark value), or empty dict.

        The values are read from the experimental k-eff workbook that the build
        matched to the benchmark id; kika does not convert them.

        Parameters
        ----------
        benchmark_id
            The id as stored.

        Returns
        -------
        dict
            ``{"keff": ..., "keff_unc": ...}``, the experimental k-eff and its
            uncertainty as the workbook gives them (``keff_unc`` is ``None``
            when the workbook has none). An empty dict when no experimental
            value was matched to this benchmark.

        Raises
        ------
        ~kika.benchmarks.BenchmarkNotFoundError
            If the id is not in the database.
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT keff_exp, keff_exp_unc FROM benchmarks WHERE benchmark_id = ?",
            (benchmark_id,),
        ).fetchone()
        if row is None:
            raise BenchmarkNotFoundError(f"Unknown benchmark id: {benchmark_id!r}")
        if row["keff_exp"] is None:
            return {}
        return {"keff": row["keff_exp"], "keff_unc": row["keff_exp_unc"]}

    def get_spectrum(self, benchmark_id: str) -> Optional[dict]:
        """Return the 299-group flux spectrum ``{'energies', 'flux'}`` or None.

        Parameters
        ----------
        benchmark_id
            The id as stored.

        Returns
        -------
        dict or None
            ``None`` when no spectrum was stored for the benchmark (an unknown
            id also gives ``None``). Otherwise two lists of equal length, one
            entry per group and in the order of the DICE file (not sorted):
            ``energies``, the upper energy edge of each group in MeV (the build
            divides the file's value by 10**6, taking it as eV), and ``flux``,
            the group flux as the file gives it (kika does not normalise it or
            state its unit).
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT ngroups, energies, flux FROM benchmark_spectra WHERE benchmark_id = ?",
            (benchmark_id,),
        ).fetchone()
        if row is None:
            return None
        n = row["ngroups"]
        return {
            "energies": unpack_f32(row["energies"], n).tolist(),
            "flux": unpack_f32(row["flux"], n).tolist(),
        }

    def get_balance(self, benchmark_id: str) -> Optional[dict]:
        """Return the reaction-rate balance ``{'summary', 'text'}`` or None.

        ``summary`` holds the extracted scalars (keff, leakage, n_zones);
        ``text`` is the full balance file rendered verbatim.

        Parameters
        ----------
        benchmark_id
            The id as stored.

        Returns
        -------
        dict or None
            ``None`` when no balance was stored (an unknown id also gives
            ``None``). Otherwise ``summary``, a dict with those of ``keff``,
            ``leakage`` (floats as written in the balance file) and ``n_zones``
            (int) that the file text yields, so a key can be missing; and
            ``text``, the decompressed file, decoded as UTF-8 with undecodable
            bytes replaced.
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT summary, raw FROM benchmark_balance WHERE benchmark_id = ?",
            (benchmark_id,),
        ).fetchone()
        if row is None:
            return None
        import gzip

        text = gzip.decompress(row["raw"]).decode("utf-8", errors="replace")
        return {"summary": json.loads(row["summary"]), "text": text}

    def get_input_deck(
        self, benchmark_id: str, code: Optional[str] = None
    ) -> List[dict]:
        """Return the raw input deck(s) as ``[{'code', 'text'}]`` (verbatim).

        Pass ``code`` (``'KENO'`` / ``'MCNP'``) to fetch a single deck; otherwise all
        decks stored for the benchmark are returned.

        Parameters
        ----------
        benchmark_id
            The id as stored.
        code
            Exact match on the deck's code. None returns every deck, ordered by
            code.

        Returns
        -------
        list of dict
            One dict per deck, ``{"code": ..., "text": ...}``: the code the
            build read from the file name (``"KENO"``, ``"MCNP"``, or ``None``
            when the name has neither) and the decompressed deck decoded as
            UTF-8 with undecodable bytes replaced. Empty when nothing matches,
            an unknown id included.
        """
        import gzip

        conn = self._get_connection()
        if code is not None:
            rows = conn.execute(
                "SELECT code, deck FROM benchmark_inputs "
                "WHERE benchmark_id = ? AND code = ?",
                (benchmark_id, code),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT code, deck FROM benchmark_inputs WHERE benchmark_id = ? "
                "ORDER BY code",
                (benchmark_id,),
            ).fetchall()
        return [
            {
                "code": r["code"],
                "text": gzip.decompress(r["deck"]).decode("utf-8", errors="replace"),
            }
            for r in rows
        ]

    def get_benchmark_dashboard(self, benchmark_id: str) -> dict:
        """Return the whole per-benchmark record in one bundle (for the app dashboard).

        Combines metadata, experimental keff, the profile-variant list, the preferred
        profile's sensitivity vectors, the flux spectrum, the balance, and the list of
        available input-deck codes.

        Parameters
        ----------
        benchmark_id
            The id as stored.

        Returns
        -------
        dict
            The dict of :meth:`get_benchmark` with these keys added:
            ``experimental_keff`` (:meth:`get_experimental_keff`, empty when
            none), ``preferred_profile_id`` (the flagged profile, else the first
            listed; ``None`` without profiles), ``profile_vector``
            (:meth:`get_profile_vector` of that profile, or ``None``),
            ``spectrum`` (:meth:`get_spectrum`), ``balance``
            (:meth:`get_balance`) and ``input_codes`` (the codes of the stored
            decks, sorted; the decks themselves are not included).

        Raises
        ------
        ~kika.benchmarks.BenchmarkNotFoundError
            If the id is not in the database.
        """
        result = self.get_benchmark(benchmark_id)  # metadata + keff_exp + profiles
        preferred = next(
            (p for p in result["profiles"] if p.get("is_preferred")),
            result["profiles"][0] if result["profiles"] else None,
        )
        result["experimental_keff"] = self.get_experimental_keff(benchmark_id)
        result["preferred_profile_id"] = preferred["profile_id"] if preferred else None
        result["profile_vector"] = (
            self.get_profile_vector(preferred["profile_id"]) if preferred else None
        )
        result["spectrum"] = self.get_spectrum(benchmark_id)
        result["balance"] = self.get_balance(benchmark_id)
        result["input_codes"] = [
            r["code"]
            for r in self._get_connection().execute(
                "SELECT code FROM benchmark_inputs WHERE benchmark_id = ? ORDER BY code",
                (benchmark_id,),
            )
        ]
        return result
