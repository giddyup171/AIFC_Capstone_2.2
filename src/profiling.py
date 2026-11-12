"""Lightweight step-level profiling: wall time, CPU time and peak Python memory per step.

Usage::

    prof = Profiler()
    with prof.step("lasso_stability"):
        ...
    prof.table()          # one row per step
    prof.save(path)       # CSV

Memory is measured with ``tracemalloc`` (Python allocations, including NumPy buffers), which is
portable across Windows/Linux and needs no extra dependency. Tracing adds overhead, so wall times
measured with ``track_memory=True`` are slightly pessimistic; that is acceptable for comparing
steps against each other. Optional ``cprofile`` gives a function-level breakdown of one call.
"""

from __future__ import annotations

import cProfile
import io
import logging
import pstats
import time
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class Profiler:
    def __init__(self, enabled: bool = True, track_memory: bool = True) -> None:
        self.enabled = enabled
        self.track_memory = track_memory
        self.records: list[dict[str, Any]] = []

    @contextmanager
    def step(self, name: str, **info: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started_tracing = False
        if self.track_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            started_tracing = True
        if self.track_memory:
            tracemalloc.reset_peak()
        wall0, cpu0 = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
            peak = tracemalloc.get_traced_memory()[1] / 2**20 if self.track_memory else float("nan")
            if started_tracing:
                tracemalloc.stop()
            self.records.append({"step": name, "wall_seconds": wall, "cpu_seconds": cpu,
                                 "peak_memory_mb": peak, **info})
            logger.info("Profile | %-34s %7.2fs wall | %7.2fs cpu | peak %7.1f MB", name, wall, cpu, peak)

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records)
        if df.empty:
            return df
        total = df["wall_seconds"].sum()
        df["share_of_wall"] = df["wall_seconds"] / total if total else 0.0
        return df

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self.table().to_csv(path, index=False)
        return path


def cprofile_call(fn: Callable[..., Any], *args: Any, top: int = 20, **kwargs: Any) -> tuple[Any, str]:
    """Run ``fn`` under cProfile; return its result and the top-N functions by cumulative time."""
    prof = cProfile.Profile()
    result = prof.runcall(fn, *args, **kwargs)
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(top)
    return result, buf.getvalue()
