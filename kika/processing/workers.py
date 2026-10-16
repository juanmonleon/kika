"""How many workers a processing stage runs on.

Every stage of the processing (reconstruction and its verification, Doppler
broadening, URR tables, heating, thermal scattering) takes a ``workers``
count, and none of their results depends on it. The count is chosen once:

1. ``workers=N`` given by the caller: N, on every stage that can use it;
2. otherwise the ``KIKA_WORKERS`` environment variable, for a server where
   the caller's code should not change;
3. otherwise each stage's own conservative default, since the machine is
   usually the user's (:func:`verification_default`, :func:`pool_default`).

A stage that runs processes stays in this one where processes cannot be
started (:func:`can_spawn`); threads are always allowed.
"""
from __future__ import annotations

import os

#: The environment variable that sets the count when the caller gives none.
ENV = "KIKA_WORKERS"


def env_workers() -> int | None:
    """``KIKA_WORKERS`` as a positive integer, ``None`` when unset or empty.

    Raises ``ValueError`` for anything else: a mistyped value must not fall
    back to a default the user did not choose."""
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return None
    try:
        count = int(raw)
    except ValueError:
        count = 0
    if count < 1:
        raise ValueError(f"{ENV}={raw!r}: expected a positive integer (the number of workers)")
    return count


def resolve(workers: int | None, default: int) -> int:
    """The count a stage uses: ``workers``, else ``KIKA_WORKERS``, else ``default``."""
    if workers is not None:
        return max(1, int(workers))
    env = env_workers()
    return env if env is not None else max(1, int(default))


def verification_default() -> int:
    """The reconstruction's threads: half the cores, at most four, at least one."""
    return max(1, min(4, (os.cpu_count() or 1) // 2))


def pool_default() -> int:
    """The broadening, URR, heating and thermal workers: the cores, at most eight."""
    return max(1, min(8, os.cpu_count() or 1))


def can_spawn() -> bool:
    """Whether this process may start worker processes.

    A daemonic process may not have children, and a spawned worker that is
    still importing the caller's main script must not start another pool."""
    import multiprocessing
    current = multiprocessing.current_process()
    return not (current.daemon or getattr(current, "_inheriting", False))
