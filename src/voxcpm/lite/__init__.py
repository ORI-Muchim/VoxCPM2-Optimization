"""Lite (CPU / low-memory) runtime for VoxCPM2.

Float weights, small footprint (original checkpoint)::

    from voxcpm.lite import load_voxcpm2_lite
    model = load_voxcpm2_lite("path/to/VoxCPM2")              # bf16, mmapped
    wav = model.generate(target_text="Hello!", inference_timesteps=10, cfg_value=2.0)

Int8 weights (convert once, then load the converted directory)::

    python -m voxcpm.lite.convert path/to/VoxCPM2 path/to/VoxCPM2-lite
    model = load_voxcpm2_lite("path/to/VoxCPM2-lite")
"""

from .loader import GROUPS, load_voxcpm2_lite
from .quant import Int8Embedding, Int8Linear, Int8LinearSlice

__all__ = ["GROUPS", "Int8Embedding", "Int8Linear", "Int8LinearSlice", "load_voxcpm2_lite"]
