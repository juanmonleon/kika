"""What an application needs to know about the engine, format-free.

The rejection categories it can raise, and whether the compiled kernel is
loaded. The ENDF-6 entry built on these is ``kika.endf.reconstruct_endf``.
"""

#: Every ``category`` a rejection can carry. Stable names: a change here is an
#: API change for applications that translate them into user text.
REJECTION_CATEGORIES = (
    # The source evaluation or its conversion to the model.
    "already-reconstructed",      # LRP=2: MF3 already holds resonance contributions
    "conversion-not-clean",       # the conversion to the model lost cross-section data
    "context-missing",            # no target mass or spin declared for the resonances
    "missing-resonance-regions",  # backgrounds or LRP=1 without usable resonance regions
    "incomplete-reactions",
    "domain-not-covered",         # modeled cross sections do not span the evaluated domain
    "internal-gap",               # a source function has a hole inside the domain
    "total-without-graph",        # MT1 present but not an explicit sum of partials
    "sum-without-graph",
    "incomplete-total-graph",
    "missing-radius-policy",
    # Physics the engine does not implement.
    "unsupported-physics",
    # Numerical limits and checks; never relaxed.
    "budget-exhausted",           # point or iteration budget
    "memory-budget-exhausted",
    "verification-failed",        # a table, sum or reload falls outside the budget
    "publication-failed",         # the processed file could not be written cleanly
)


def native_available() -> bool:
    """Whether the compiled resonance kernel is loaded (else NumPy, slower)."""
    from . import _rm_acceleration
    return _rm_acceleration._native is not None
