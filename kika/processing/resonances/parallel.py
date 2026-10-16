"""Exact verification physics shared among a few low-priority threads.

Publication verification evaluates the resonance physics at every grid node
and four probes per panel; for a heavy actinide that is millions of energies
times thousands of levels. The energies are independent, and the compiled
kernel releases the GIL while it sums over levels, so threads of this process
run the blocks in parallel (U-238: x3.4 with four) and give bit-identical
numbers.

Threads, not processes, on purpose: a spawned process re-imports the caller's
``__main__``, which re-runs any script without an ``if __name__ ==
"__main__"`` guard (every generated script), and a frozen application needs
its own start-up support. Threads also end with the process when the user
cancels.

The machine is the user's, so the defaults are conservative:

* at most ``min(4, cores // 2)`` threads (``default_workers``), unless the
  caller or ``KIKA_WORKERS`` asks for another count (:mod:`kika.processing.workers`);
* each thread at below-normal priority where the system allows it;
* only when the work is large enough to be worth it.
"""
from __future__ import annotations

import os
import sys

#: Energy x level products below which the serial path is used.
#: U-238 (ENDF/B-VIII.1) is ~2.7e10; a full Fe-56 ~1e8.
PARALLEL_WORK_THRESHOLD = 1e9


def default_workers() -> int:
    """``KIKA_WORKERS`` when set, else half the cores, at most four, at least one."""
    from kika.processing.workers import resolve, verification_default
    return resolve(None, verification_default())


def _lower_thread_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.SetThreadPriority(kernel.GetCurrentThread(), -1)  # THREAD_PRIORITY_BELOW_NORMAL
        elif sys.platform.startswith("linux"):
            # On Linux a thread id is a valid PRIO_PROCESS target.
            os.setpriority(os.PRIO_PROCESS, __import__("threading").get_native_id(), 10)
    except Exception:
        pass


def compute_blocks(items, compute, workers, store, progress=None):
    """Run ``compute(item)`` for every item on ``workers`` threads.

    ``store(item, result)`` and ``progress(fraction)`` are called on the
    calling thread as results arrive, so callers need no locking. An error in
    any block propagates unchanged after the remaining work is cancelled.
    ``items`` are sequences whose last element has a length (the energies).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    total = sum(len(item[-1]) for item in items) or 1
    done = 0
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kika-verify",
                              initializer=_lower_thread_priority)
    try:
        futures = {pool.submit(compute, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            store(item, future.result())
            done += len(item[-1])
            if progress is not None:
                progress(float(min(1., done / total)))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
