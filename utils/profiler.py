"""Lightweight wall-clock profiler for periodic performance logging.

Each actor creates one Profiler instance. Operations are timed with the
`time()` context manager; `step()` is called once per logical unit of work
(training step, episode, reanalyze batch). Stats are logged and reset every
`log_interval` steps.

JAX note: JAX dispatches GPU kernels asynchronously. To measure actual GPU
compute time (not just dispatch time), call `jax.block_until_ready(result)`
inside the `time()` block before it exits.

Example::

    profiler = Profiler("LearnerActor", log_interval=100)

    with profiler.time("sample_wait"):
        batch = ray.get(prefetch_future)

    with profiler.time("train_step"):
        params, metrics = train_step(...)
        jax.block_until_ready(params)   # blocks until GPU kernel finishes

    profiler.step()                     # logs every log_interval calls
"""

import time
from collections import defaultdict
from contextlib import contextmanager

from utils.logging_utils import logger


class Profiler:
    def __init__(self, name: str, log_interval: int = 100):
        self.name = name
        self.log_interval = log_interval
        self._step_count = 0
        self._totals: dict = defaultdict(float)
        self._counts: dict = defaultdict(int)

    @contextmanager
    def time(self, key: str):
        t0 = time.monotonic()
        yield
        self._totals[key] += time.monotonic() - t0
        self._counts[key] += 1

    def step(self):
        """Increment step counter; log and reset at log_interval."""
        self._step_count += 1
        if self._step_count % self.log_interval == 0:
            self._log()
            self._reset()

    def _log(self):
        if not self._totals:
            return
        parts = []
        for key in sorted(self._totals):
            n = self._counts[key]
            mean_ms = self._totals[key] / n * 1000
            parts.append(f"{key}={mean_ms:.1f}ms")
        logger.info(f"[profile:{self.name} @ step {self._step_count}]  " + "  |  ".join(parts))

    def _reset(self):
        self._totals.clear()
        self._counts.clear()
