"""
Build the C++ replay buffer extension in-place.

Usage:
    pip install pybind11
    python setup.py build_ext --inplace

The compiled _replay_buffer_cpp.*.so lands at the repo root and is
automatically found by utils/replay_buffer.py at import time.

CUDA is auto-detected by CMake; no manual flag needed.
"""
import subprocess
import sys
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    def build_extension(self, ext):
        build_dir = Path(self.build_temp) / ext.name
        build_dir.mkdir(parents=True, exist_ok=True)
        src_dir = Path(__file__).parent.resolve()

        cmake_args = [
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        build_args = ["--build", ".", "--", f"-j{self._cpu_count()}"]

        subprocess.run(
            ["cmake", str(src_dir)] + cmake_args,
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            ["cmake"] + build_args,
            cwd=build_dir,
            check=True,
        )

        # CMake places the .so directly in the repo root (LIBRARY_OUTPUT_DIRECTORY).
        # If setuptools expects it elsewhere for --inplace, copy it there too.
        import glob, shutil
        pattern = str(src_dir / f"_replay_buffer_cpp*.so")
        built = glob.glob(pattern)
        if built:
            dest = Path(self.get_ext_fullpath(ext.name))
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.resolve() != Path(built[0]).resolve():
                shutil.copy2(built[0], dest)

    @staticmethod
    def _cpu_count():
        try:
            import os
            return os.cpu_count() or 4
        except Exception:
            return 4


setup(
    name="replay_buffer_cpp",
    version="0.1.0",
    description="Lock-free C++ prioritized replay buffer with pybind11 bindings",
    ext_modules=[Extension("_replay_buffer_cpp", sources=[])],
    cmdclass={"build_ext": CMakeBuildExt},
    python_requires=">=3.10",
)
