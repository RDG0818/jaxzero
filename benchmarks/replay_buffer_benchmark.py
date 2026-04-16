"""
Replay Buffer Benchmark
=======================
Directly compares the C++ backend (_replay_buffer_cpp) against the pure-Python
fallback (cpprb) on the operations that matter for training throughput:

  - add()             — data collection throughput
  - sample()          — training step latency
  - sample_for_reanalysis() — reanalysis latency
  - update_priorities() — post-training step latency
  - RSS memory delta  — heap footprint after filling the buffer
  - H2D transfer      — jax.device_put() time on pinned vs regular arrays

Run:
    python benchmarks/replay_buffer_benchmark.py

Optional flags (as positional env overrides):
    CAPACITY=50000 BATCH=512 python benchmarks/replay_buffer_benchmark.py

Requirements:
    pip install psutil
    python setup.py build_ext --inplace  # for C++ backend
"""

import os
import sys
import time
import gc

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

CAPACITY  = int(os.environ.get("CAPACITY",  10_000))
OBS_SIZE  = int(os.environ.get("OBS_SIZE",  64))
N         = int(os.environ.get("N",         3))      # agents
U         = int(os.environ.get("U",         5))      # unroll steps
A         = int(os.environ.get("A",         5))      # action space size
BATCH     = int(os.environ.get("BATCH",     256))
REANALYZE = int(os.environ.get("REANALYZE", 64))
REPS      = int(os.environ.get("REPS",      200))    # sample/update reps for stable timing
FILL_N    = CAPACITY                                  # items to add before sampling

ALPHA      = 0.6
BETA_START = 0.4
BETA_FRAMES = 100_000

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_item(rng):
    return dict(
        observation   = rng.random((N, OBS_SIZE),  dtype=np.float32),
        actions       = rng.integers(0, A, (U, N), dtype=np.int32),
        policy_target = rng.random((U+1, N, A),    dtype=np.float32),
        value_target  = rng.random((U+1, N),       dtype=np.float32),
        reward_target = rng.random((U, N),         dtype=np.float32),
        agent_order   = np.arange(N,               dtype=np.int32),
    )


def rss_mb() -> float:
    """Current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return float("nan")


def fmt_speed(items_per_sec: float) -> str:
    if items_per_sec >= 1e6:
        return f"{items_per_sec/1e6:.2f}M items/s"
    if items_per_sec >= 1e3:
        return f"{items_per_sec/1e3:.1f}K items/s"
    return f"{items_per_sec:.1f} items/s"


def separator(char="─", width=60):
    print(char * width)


# ── Backend benchmark ────────────────────────────────────────────────────────

def benchmark_backend(buf, label: str):
    rng = np.random.default_rng(42)
    print(f"\n── {label} {'─' * max(0, 52 - len(label))}")

    # ── add() throughput ────────────────────────────────────────────────────
    items = [make_item(rng) for _ in range(FILL_N)]
    gc.collect()
    mem_before = rss_mb()
    t0 = time.perf_counter()
    for item in items:
        buf.add(item, priority=1.0)
    t_add = time.perf_counter() - t0
    mem_after = rss_mb()

    add_per_sec = FILL_N / t_add
    add_us = t_add / FILL_N * 1e6
    print(f"  add()             : {fmt_speed(add_per_sec):>20s}  ({add_us:.1f} µs/item)")
    print(f"  RSS delta         : {mem_after - mem_before:>18.1f} MB")

    # ── sample() latency ────────────────────────────────────────────────────
    # Warm up JIT / cpprb internal structures.
    for _ in range(5):
        buf.sample(BATCH)

    t0 = time.perf_counter()
    for _ in range(REPS):
        batch, weights, indices = buf.sample(BATCH)
    t_sample = (time.perf_counter() - t0) / REPS * 1e3  # ms

    pinned = getattr(getattr(buf, "_buf", None), "is_pinned", lambda: False)
    pin_str = f"  [pinned={pinned()}]" if callable(pinned) else ""
    print(f"  sample({BATCH})      : {t_sample:>16.3f} ms/call{pin_str}")

    # ── sample_for_reanalysis() latency ─────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(REPS):
        buf.sample_for_reanalysis(REANALYZE)
    t_reanalyze = (time.perf_counter() - t0) / REPS * 1e3
    print(f"  sample_reanalysis  : {t_reanalyze:>16.3f} ms/call")

    # ── update_priorities() latency ─────────────────────────────────────────
    _, _, indices = buf.sample(BATCH)
    priorities = np.random.rand(BATCH).astype(np.float32) + 0.01

    t0 = time.perf_counter()
    for _ in range(REPS):
        buf.update_priorities(indices, priorities)
    t_upd = (time.perf_counter() - t0) / REPS * 1e3
    print(f"  update_priorities  : {t_upd:>16.3f} ms/call")

    return {
        "add_us": add_us,
        "sample_ms": t_sample,
        "reanalyze_ms": t_reanalyze,
        "update_ms": t_upd,
        "rss_delta_mb": mem_after - mem_before,
        "batch": batch,
        "weights": weights,
    }


# ── H2D transfer benchmark ───────────────────────────────────────────────────

def benchmark_h2d(cpp_result, py_result):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("\n── H2D Transfer  (JAX not available — skipping) ──────────────────")
        return

    print(f"\n── H2D Transfer (jax.device_put, batch={BATCH} obs) ─────────────────")

    def time_device_put(arr: np.ndarray, label: str, warmup: int = 3, reps: int = 50):
        # Warm up.
        for _ in range(warmup):
            jax.block_until_ready(jax.device_put(arr))
        t0 = time.perf_counter()
        for _ in range(reps):
            jax.block_until_ready(jax.device_put(arr))
        ms = (time.perf_counter() - t0) / reps * 1e3
        print(f"  {label:<22s}: {ms:>8.3f} ms/call")
        return ms

    # Compare C++ (possibly pinned) vs Python (regular numpy).
    cpp_obs = cpp_result["batch"].observation if cpp_result["batch"] is not None else None
    py_obs  = py_result["batch"].observation  if py_result["batch"]  is not None else None

    t_cpp = t_py = None
    if cpp_obs is not None:
        t_cpp = time_device_put(cpp_obs, "C++ (pinned?)")
    if py_obs is not None:
        t_py  = time_device_put(py_obs,  "Python (regular)")

    if t_cpp is not None and t_py is not None and t_cpp > 0:
        speedup = t_py / t_cpp
        print(f"  {'speedup':<22s}: {speedup:>8.2f}x")

    # Also time a fresh regular numpy array of the same shape for a fair baseline.
    baseline = np.random.rand(BATCH, N, OBS_SIZE).astype(np.float32)
    t_baseline = time_device_put(baseline, "fresh numpy (baseline)")

    if t_cpp is not None and t_baseline > 0:
        print(f"  {'pinned vs baseline':<22s}: {t_baseline / t_cpp:>8.2f}x")


# ── Build Python-fallback buffer ─────────────────────────────────────────────

def make_py_buffer():
    """Forces the Python fallback regardless of whether the C++ .so is built."""
    import utils.replay_buffer as rb_module

    orig_cpp  = rb_module._CppReplayBuffer
    orig_cfg  = rb_module._CppConfig
    rb_module._CppReplayBuffer = None
    rb_module._CppConfig = None

    from utils.replay_buffer import ReplayBuffer, ReplayItem
    import numpy as _np

    # Monkey-patch the add() call to accept the raw dicts we produce.
    class _BufWrapper(ReplayBuffer):
        def add(self, item_dict, priority=1.0):
            item = ReplayItem(**item_dict)
            super().add(item, priority)
        def sample_for_reanalysis(self, n):
            result = super().sample_for_reanalysis(n)
            return result

    buf = _BufWrapper(
        capacity=CAPACITY,
        observation_shape=(OBS_SIZE,),
        action_space_size=A,
        num_agents=N,
        unroll_steps=U,
        alpha=ALPHA,
        beta_start=BETA_START,
        beta_frames=BETA_FRAMES,
    )

    # Restore so other imports aren't affected.
    rb_module._CppReplayBuffer = orig_cpp
    rb_module._CppConfig = orig_cfg

    return buf


def make_cpp_buffer():
    """Returns a ReplayBuffer using the C++ backend, or None if not available."""
    from utils.replay_buffer import ReplayBuffer, ReplayItem

    class _BufWrapper(ReplayBuffer):
        def add(self, item_dict, priority=1.0):
            item = ReplayItem(**item_dict)
            super().add(item, priority)
        def sample_for_reanalysis(self, n):
            return super().sample_for_reanalysis(n)

    buf = _BufWrapper(
        capacity=CAPACITY,
        observation_shape=(OBS_SIZE,),
        action_space_size=A,
        num_agents=N,
        unroll_steps=U,
        alpha=ALPHA,
        beta_start=BETA_START,
        beta_frames=BETA_FRAMES,
    )
    if not buf._use_cpp:
        return None, buf  # C++ not available; returned buf is actually Python
    return buf, None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    separator("=")
    print("  Replay Buffer Benchmark")
    separator("=")
    print(f"  capacity={CAPACITY:,} | obs_size={OBS_SIZE} | agents={N} | unroll={U} | A={A}")
    print(f"  batch={BATCH} | reanalyze_batch={REANALYZE} | timing_reps={REPS}")

    cpp_buf, fallback = make_cpp_buffer()

    cpp_result = None
    py_result  = None

    # ── C++ backend ─────────────────────────────────────────────────────────
    if cpp_buf is not None:
        cpp_result = benchmark_backend(cpp_buf, "C++ backend (_replay_buffer_cpp)")
        del cpp_buf; gc.collect()
    else:
        print("\n── C++ backend  (not built — run: python setup.py build_ext --inplace) ──")

    # ── Python backend ───────────────────────────────────────────────────────
    try:
        py_buf = make_py_buffer()
        py_result = benchmark_backend(py_buf, "Python backend (cpprb)")
        del py_buf; gc.collect()
    except Exception as e:
        print(f"\n── Python backend  (unavailable: {e}) ──")

    # ── Speedup summary ──────────────────────────────────────────────────────
    if cpp_result is not None and py_result is not None:
        print(f"\n── Speedup  C++ / Python {'─' * 35}")
        for key, label in [
            ("add_us",        "add()"),
            ("sample_ms",     "sample()"),
            ("reanalyze_ms",  "sample_reanalysis()"),
            ("update_ms",     "update_priorities()"),
        ]:
            py_val  = py_result[key]
            cpp_val = cpp_result[key]
            if cpp_val > 0 and py_val > 0:
                speedup = py_val / cpp_val
                print(f"  {label:<26s}: {speedup:>6.1f}x")

    # ── H2D transfer ─────────────────────────────────────────────────────────
    if cpp_result is not None or py_result is not None:
        benchmark_h2d(
            cpp_result if cpp_result else {"batch": None},
            py_result  if py_result  else {"batch": None},
        )

    separator()
    print()


if __name__ == "__main__":
    # Ensure repo root is on sys.path so utils/ is importable.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    main()
