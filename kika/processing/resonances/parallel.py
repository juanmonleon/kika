"""Exact verification physics shared among low-priority workers.

Publication verification evaluates the resonance physics at every grid node
and four probes per panel; for a heavy actinide that is millions of energies
times thousands of levels. The energies are independent, so blocks of them
run in parallel and give bit-identical numbers.

Processes where this process may start them (:func:`compute_exact_blocks`),
threads otherwise (:func:`compute_blocks`). The compiled RM sums release the
GIL and threads scale on them (U-238: x3.4 with four); the NumPy reference
solver of RML does not: with large blocks the threads wait on memory, with
small ones on the GIL, and Pu-239 stops near two threads (398 s on one,
~218 s on six to sixteen). Processes with small, cache-sized blocks have
neither limit (Pu-239, eight processes: 125 s, the same maxima bit for bit).

A spawned process re-imports the caller's ``__main__``, which re-runs any
script without an ``if __name__ == "__main__"`` guard. ``reconstruct_suite``
refuses to start in a worker that is still importing it, so such a pool
breaks at once and the verification falls back to threads with a warning.
Daemonic processes (the app's workers) may not start children and use
threads. Threads also end with the process when the user cancels.

The machine is the user's, so the defaults are conservative:

* at most ``min(4, cores // 2)`` workers (``default_workers``), unless the
  caller or ``KIKA_WORKERS`` asks for another count (:mod:`kika.processing.workers`);
* each worker at below-normal priority where the system allows it;
* only when the work is large enough to be worth it.
"""
from __future__ import annotations

import os
import sys

#: Energy x level products below which the serial path is used.
#: U-238 (ENDF/B-VIII.1) is ~2.7e10; a full Fe-56 ~1e8.
PARALLEL_WORK_THRESHOLD = 1e9

#: Work bytes of each exact-physics sub-block on a worker process: small
#: enough to stay in cache (Pu-239 on eight processes: 125 s at 4 MiB,
#: 244 s at the reconstruction's 64 MiB). Sub-blocking changes the cost,
#: not the numbers.
PROCESS_WORK_BYTES = 4 * 1024 ** 2


class ProcessesUnavailable(RuntimeError):
    """The worker processes could not run (a broken pool): use threads."""


def default_workers() -> int:
    """``KIKA_WORKERS`` when set, else half the cores, at most four, at least one."""
    from kika.processing.workers import resolve, verification_default
    return resolve(None, verification_default())


def _lower_process_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except Exception:
        pass


#: A worker process's prepared regions, context and sub-block bytes.
_EXACT = {}


def _start_exact_worker(regions, context, work_bytes):
    _lower_process_priority()
    _EXACT.update(regions=regions, context=context, work_bytes=work_bytes)


def _exact_block(job):
    from .suite import _exact_physical_block
    index, energies = job
    return _exact_physical_block(_EXACT["regions"][index], _EXACT["context"], energies,
                                 _EXACT["work_bytes"])


def compute_exact_blocks(items, regions, context, workers, store, progress=None):
    """The exact physical partials of every block on ``workers`` processes.

    ``items`` are ``(segment, ..., energies)``; ``regions[segment]`` is each
    segment's prepared region, sent once to every worker. ``store`` and
    ``progress`` as :func:`compute_blocks`. Returns False, having stored
    nothing, where this process may not start processes. Raises
    :class:`ProcessesUnavailable` if the pool breaks (the caller removes what
    was stored and uses threads); any other error propagates unchanged.
    """
    from kika.processing.workers import can_spawn
    if not can_spawn():
        return False
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    total = sum(len(item[-1]) for item in items) or 1
    done = 0
    try:
        pool = ProcessPoolExecutor(max_workers=min(workers, len(items)), initializer=_start_exact_worker,
                                   initargs=(regions, context, PROCESS_WORK_BYTES))
    except (OSError, NotImplementedError) as error:
        raise ProcessesUnavailable(str(error)) from error
    try:
        futures = {pool.submit(_exact_block, (item[0], item[-1])): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            store(item, future.result())
            done += len(item[-1])
            if progress is not None:
                progress(float(min(1., done / total)))
    except BrokenProcessPool as error:
        raise ProcessesUnavailable(str(error)) from error
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return True


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
