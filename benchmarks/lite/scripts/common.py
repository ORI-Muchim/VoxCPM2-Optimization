"""Shared locations for the benchmark scripts.

Environment variables:
    VOXCPM2_DIR       original VoxCPM2 checkpoint directory
    VOXCPM2_LITE_DIR  directory written by ``python -m voxcpm.lite.convert``
    VOXCPM_BENCH_OUT  where generated audio, traces and statistics are written
                      (default: benchmarks/lite/out, which is git-ignored)
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Reference clip used for the voice-cloning items and as the SQUIM anchor.
REF_WAV = str(REPO_ROOT / "examples" / "reference_speaker.wav")


def _env_dir(var: str, what: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise SystemExit(f"Set {var} to {what}.")
    return value


def model_dir() -> str:
    return _env_dir("VOXCPM2_DIR", "the original VoxCPM2 checkpoint directory")


def lite_dir() -> str:
    return _env_dir("VOXCPM2_LITE_DIR", "a directory written by `python -m voxcpm.lite.convert`")


def out_path(*parts: str) -> str:
    """Path under the output directory; parent directories are created."""
    root = Path(os.environ.get("VOXCPM_BENCH_OUT", REPO_ROOT / "benchmarks" / "lite" / "out"))
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def out_dir(*parts: str) -> str:
    """Directory under the output directory; created if missing."""
    root = Path(os.environ.get("VOXCPM_BENCH_OUT", REPO_ROOT / "benchmarks" / "lite" / "out"))
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
