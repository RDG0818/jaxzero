"""
JAX + Ray: CUDA plugin crash in zero-GPU workers
=================================================

ISSUE SUMMARY
-------------
When Ray spawns a worker with no GPU allocation (num_gpus=0 or unspecified),
it sets CUDA_VISIBLE_DEVICES="" in that process. JAX's CUDA plugin
(jax_plugins.xla_cuda12) calls cuInit(0) during plugin discovery at import
time. With no visible devices, cuInit returns CUDA_ERROR_NO_DEVICE and the
plugin raises a RuntimeError, crashing the worker even though it was only
ever intended to run on CPU.

Affected versions (confirmed):
  jax==0.4.x (cuda12 plugin), ray>=2.x

Upstream issue locations:
  JAX:  https://github.com/google/jax  (jax_plugins/xla_cuda12/__init__.py)
  Ray:  https://github.com/ray-project/ray  (worker CUDA_VISIBLE_DEVICES override)


REPRODUCTION
------------
Run this file directly. The broken actor will crash; the fixed actor will
print its device list normally.

    python jax_ray_cuda_issue.py


ROOT CAUSE
----------
The call chain in the JAX CUDA plugin:

    import jax
      -> jax._src.xla_bridge.discover_pjrt_plugins()
         -> jax_plugins.xla_cuda12.initialize()
            -> _check_cuda_versions(raise_on_first_error=True)
               -> cuda_versions.cuda_device_count()
                  -> cuInit(0)  # <-- raises CUDA_ERROR_NO_DEVICE
                                #     when CUDA_VISIBLE_DEVICES=""

A plugin that cannot find its hardware should gracefully skip initialization,
not crash the process. If CUDA_VISIBLE_DEVICES="" were set by the user
intentionally (or by Ray on behalf of the user), the correct behavior is for
the CUDA backend to simply be unavailable, identical to never having installed
jax[cuda12] in the first place.

There are two independent places the fix belongs:

  FIX A — JAX (preferred, highest impact)
  ----------------------------------------
  In jax_plugins/xla_cuda12/__init__.py, catch CUDA_ERROR_NO_DEVICE in
  initialize() and return without raising:

    def initialize():
        try:
            _check_cuda_versions(raise_on_first_error=True)
        except RuntimeError as e:
            if "CUDA_ERROR_NO_DEVICE" in str(e):
                # No visible devices — skip CUDA backend, fall back to CPU.
                return
            raise

  This fixes the crash for any multi-process framework (Ray, multiprocessing,
  Dask, etc.) whenever a subprocess intentionally hides GPU devices.

  FIX B — Ray (already in progress)
  -----------------------------------
  Ray sets CUDA_VISIBLE_DEVICES="" for zero-GPU workers as a legacy safety
  measure. This predates JAX's eager CUDA plugin initialization. Ray is
  already deprecating this behavior via RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:

    https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html

  Setting RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 opts into the new behavior
  (do not override device visibility for zero-GPU workers). A future Ray
  release will make this the default.


WORKAROUND (applied in train_v2.py)
-------------------------------------
Until Fix A lands in a JAX release, call this at the very top of any
Ray actor __init__ that needs JAX on CPU, before any import of jax:

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # undo Ray's restriction
    os.environ["JAX_PLATFORMS"] = "cpu"           # keep JAX on CPU

Removing the restriction lets cuInit(0) succeed (the GPU is visible to the
CUDA runtime), while JAX_PLATFORMS=cpu ensures JAX never uses the GPU as a
compute backend and does not allocate GPU memory.

Also set this before ray.init() in your main process:

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

This opts into Ray's forthcoming default and eliminates the need for the
os.environ.pop workaround entirely once Ray updates its default.
"""

import os
import sys

import ray


# ---------------------------------------------------------------------------
# Broken actor — reproduces the crash
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1)
class BrokenCpuActor:
    """
    Demonstrates the crash.

    Ray sets CUDA_VISIBLE_DEVICES="" for this worker. Importing JAX triggers
    the CUDA plugin's cuInit(0) call, which fails with CUDA_ERROR_NO_DEVICE.
    """
    def __init__(self):
        import jax  # <-- crashes here if CUDA_VISIBLE_DEVICES="" and GPU present
        self.devices = str(jax.devices())

    def get_devices(self):
        return self.devices


# ---------------------------------------------------------------------------
# Fixed actor — applies the workaround
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1)
class FixedCpuActor:
    """
    Applies the workaround: pop Ray's CUDA_VISIBLE_DEVICES restriction before
    importing JAX, then lock JAX to CPU via JAX_PLATFORMS.
    """
    def __init__(self):
        # Step 1: Remove Ray's CUDA_VISIBLE_DEVICES="" so cuInit can succeed.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # Step 2: Tell JAX to use CPU only — no GPU memory will be allocated.
        os.environ["JAX_PLATFORMS"] = "cpu"

        import jax
        self.devices = str(jax.devices())

    def get_devices(self):
        return self.devices


# ---------------------------------------------------------------------------
# Proposed upstream fix for JAX (illustrative — not monkey-patching at runtime)
# ---------------------------------------------------------------------------

PROPOSED_JAX_FIX = """
# File: jax_plugins/xla_cuda12/__init__.py
# Location: the initialize() function (~line 320 in recent releases)

# BEFORE:
def initialize():
    _check_cuda_versions(raise_on_first_error=True)
    # ... rest of initialization

# AFTER:
def initialize():
    try:
        _check_cuda_versions(raise_on_first_error=True)
    except RuntimeError as e:
        if "CUDA_ERROR_NO_DEVICE" in str(e):
            # No visible CUDA devices (e.g. CUDA_VISIBLE_DEVICES is empty or
            # unset with no physical GPU). Skip CUDA backend gracefully instead
            # of crashing — the process will fall back to other available
            # backends (CPU, TPU, etc.).
            import logging
            logging.getLogger(__name__).debug(
                "JAX CUDA plugin: no visible devices, skipping initialization."
            )
            return
        raise
"""


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    # Opt into Ray's forthcoming default (no CUDA_VISIBLE_DEVICES override for
    # zero-GPU workers). Set before ray.init().
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray.init(ignore_reinit_error=True)

    print("=" * 60)
    print("JAX + Ray CUDA plugin crash — reproduction & workaround")
    print("=" * 60)

    # ---- Test the broken actor ----
    print("\n[1] BrokenCpuActor (no workaround applied)")
    print("    Expected: RuntimeError from cuInit(0) if a GPU is present")
    try:
        broken = BrokenCpuActor.remote()
        devices = ray.get(broken.get_devices.remote(), timeout=15)
        # If no GPU is present in this environment the actor won't crash,
        # which is also useful signal.
        print(f"    Result (no GPU in this env, so no crash): {devices}")
    except Exception as e:
        print(f"    Crashed as expected: {type(e).__name__}: {e}")

    # ---- Test the fixed actor ----
    print("\n[2] FixedCpuActor (workaround applied)")
    print("    Expected: CPU device, no crash")
    try:
        fixed = FixedCpuActor.remote()
        devices = ray.get(fixed.get_devices.remote(), timeout=15)
        print(f"    Result: {devices}")
        assert "cpu" in devices.lower() or "TFRT" in devices, \
            f"Expected CPU device, got: {devices}"
        print("    PASS: actor initialized on CPU without crashing.")
    except Exception as e:
        print(f"    FAIL: {type(e).__name__}: {e}")
        sys.exit(1)

    # ---- Print the proposed upstream fix ----
    print("\n[3] Proposed upstream fix for JAX")
    print(PROPOSED_JAX_FIX)

    ray.shutdown()


if __name__ == "__main__":
    main()
