"""Convert a VoxCPM2 checkpoint into the compact int8 ``lite`` layout.

    python -m voxcpm.lite.convert /path/to/VoxCPM2 /path/to/VoxCPM2-lite

What it does:

1. loads the original model in float and runs a few short generations to
   record the per-channel range of every transformer-block Linear input
   (calibration takes about a minute on a 4-core CPU);
2. picks the handful of outlier input channels per layer that stay in float
   (see ``quant.py``) and quantizes the rest of each weight to int8;
3. writes::

       config.json                 original config + a "lite" section
       model.lite.safetensors      int8 block weights (+ scales, float columns),
                                   int8 embedding, everything else unchanged
       audiovae.lite.safetensors   AudioVAE with weight-norm folded (fp16)
       tokenizer files             copied verbatim

The files are named ``*.lite.*`` on purpose so that the stock loader fails
loudly instead of silently mis-reading int8 weights.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .loader import (
    FCOLS_SUFFIX,
    FIDX_SUFFIX,
    GROUPS,
    LITE_FORMAT_VERSION,
    LITE_MODEL_FILE,
    LITE_VAE_FILE,
    SCALE_SUFFIX,
    _open,
    block_linear_groups,
    find_checkpoint_files,
    fold_weight_norm,
    group_of,
    load_voxcpm2_lite,
)
from .quant import quantize_weight_per_channel, select_outlier_channels

_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenization_voxcpm2.py")

CALIBRATION_TEXTS = (
    "Scientists have discovered a new species of deep sea fish living near volcanic vents.",
    "이 근처에 맛있는 한식당이 있으면 추천해 주시겠어요?",
    "今天天气真不错，我们一起去公园散步吧。",
)


def _cpu_has_bf16() -> bool:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    return bool({"avx512_bf16", "amx_bf16"} & flags)
    except OSError:
        pass
    return False


@torch.inference_mode()
def calibrate(
    model: nn.Module,
    texts: Sequence[str] = CALIBRATION_TEXTS,
    steps_per_text: int = 12,
) -> Dict[str, torch.Tensor]:
    """Per-input-channel abs-max of every transformer-block Linear, observed
    while generating ``steps_per_text`` patches for each calibration text."""
    stats: Dict[str, torch.Tensor] = {}
    hooks = []
    for block, leaves in block_linear_groups(model):
        # all projections of a group see the same tensor: observe the first only
        lin = model.get_submodule(f"{block}.{leaves[0]}")
        key = f"{block}.{leaves[0]}"

        def pre(mod, inp, _k=key):
            x = inp[0].detach().to(torch.float32)
            cur = x.reshape(-1, x.shape[-1]).abs().amax(0)
            stats[_k] = cur if _k not in stats else torch.maximum(stats[_k], cur)

        hooks.append(lin.register_forward_pre_hook(pre))
    try:
        for i, text in enumerate(texts):
            model.generate(target_text=text, max_len=steps_per_text, inference_timesteps=10, cfg_value=2.0, seed=i)
    finally:
        for h in hooks:
            h.remove()
    return stats


def convert(
    src: str,
    dst: str,
    keep_float: Sequence[str] = (),
    outlier_ratio: float = 6.0,
    max_float_channels: int = 64,
    vae_dtype: str = "float16",
    with_vae_encoder: bool = True,
    calibration_precision: Optional[str] = None,
    calibration_texts: Sequence[str] = CALIBRATION_TEXTS,
    calibration_steps: int = 12,
    verbose: bool = True,
) -> Dict[str, float]:
    """Write an int8 lite checkpoint for ``src`` into ``dst``.

    Args:
        keep_float: stacks (``GROUPS``) left untouched, e.g. ``["feat_decoder"]``.
        outlier_ratio / max_float_channels: an input channel is computed in
            float when its calibrated abs-max exceeds ``outlier_ratio`` x the
            median channel (at most ``max_float_channels`` per layer).
        vae_dtype: storage dtype of the AudioVAE (it always runs in float32).
        with_vae_encoder: ``False`` drops the encoder (no voice cloning).
        calibration_precision: ``"bfloat16"`` / ``"float32"`` for the
            calibration run (default: bf16 when the CPU accelerates it).
    """
    from safetensors.torch import save_file

    keep_float = sorted(set(keep_float))
    unknown = set(keep_float) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unknown groups: {sorted(unknown)} (choose from {GROUPS})")
    model_file, vae_file, is_lite = find_checkpoint_files(src)
    if is_lite:
        raise ValueError(f"{src} is already a lite checkpoint")
    os.makedirs(dst, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(f"[voxcpm.lite.convert] {msg}", file=sys.stderr, flush=True)

    # ---- 1. calibration ------------------------------------------------------
    if calibration_precision is None:
        calibration_precision = "bfloat16" if _cpu_has_bf16() else "float32"
    log(f"calibrating in {calibration_precision} on {len(calibration_texts)} texts ...")
    model = load_voxcpm2_lite(src, precision=calibration_precision, load_vae_encoder=False, verbose=False)
    stats = calibrate(model, calibration_texts, calibration_steps)
    quant_names: Dict[str, str] = {}  # weight tensor name -> stats key
    for block, leaves in block_linear_groups(model):
        if group_of(block) in keep_float:
            continue
        for leaf in leaves:
            quant_names[f"{block}.{leaf}.weight"] = f"{block}.{leaves[0]}"
    del model

    # ---- 2. transformer stacks ------------------------------------------------
    reader = _open(model_file)
    out: Dict[str, torch.Tensor] = {}
    n_int8 = n_float = n_split = 0
    for name in sorted(reader.keys()):
        t = reader.get(name)
        if name in quant_names:
            fidx = select_outlier_channels(stats[quant_names[name]], outlier_ratio, max_float_channels)
            q, scale = quantize_weight_per_channel(t, zero_cols=fidx)
            out[name] = q
            out[name + SCALE_SUFFIX] = scale
            if fidx.numel() > 0:
                out[name + FIDX_SUFFIX] = fidx
                out[name + FCOLS_SUFFIX] = t.index_select(1, fidx.to(torch.long)).contiguous()
                n_split += fidx.numel()
            n_int8 += q.numel()
        elif name == "base_lm.embed_tokens.weight":
            q, scale = quantize_weight_per_channel(t)
            out[name] = q
            out[name + SCALE_SUFFIX] = scale
            n_int8 += q.numel()
        else:
            out[name] = t.contiguous()
            n_float += t.numel()
    meta = {"format": "pt", "voxcpm_lite": str(LITE_FORMAT_VERSION)}
    save_file(out, os.path.join(dst, LITE_MODEL_FILE), metadata=meta)
    out.clear()

    # ---- 3. AudioVAE ------------------------------------------------------------
    vae_reader = _open(vae_file)
    vae_state = fold_weight_norm({k: vae_reader.get(k) for k in vae_reader.keys()})
    if not with_vae_encoder:
        vae_state = {k: v for k, v in vae_state.items() if not k.startswith("encoder.")}
    vdt = {"float16": torch.float16, "float32": torch.float32}[vae_dtype]
    vae_out = {k: (v.to(vdt) if v.is_floating_point() else v).contiguous() for k, v in vae_state.items()}
    save_file(vae_out, os.path.join(dst, LITE_VAE_FILE), metadata=meta)

    # ---- 4. config / tokenizer ------------------------------------------------------
    with open(os.path.join(src, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["lite"] = {
        "format": LITE_FORMAT_VERSION,
        "precision": "int8",
        "keep_float": keep_float,
        "outlier_ratio": outlier_ratio,
        "max_float_channels": max_float_channels,
        "vae_dtype": vae_dtype,
        "vae_encoder": bool(with_vae_encoder),
    }
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    for fn in _TOKENIZER_FILES:
        p = os.path.join(src, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, fn))

    stats_out = {
        "src_mb": round(sum(os.path.getsize(p) for p in (model_file, vae_file)) / 1e6, 1),
        "dst_mb": round(sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst)) / 1e6, 1),
        "int8_params_m": round(n_int8 / 1e6, 1),
        "float_params_m": round(n_float / 1e6, 1),
        "float_channels_total": int(n_split),
    }
    log(json.dumps(stats_out))
    return stats_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="original VoxCPM2 checkpoint directory")
    ap.add_argument("dst", help="output directory")
    ap.add_argument("--keep-float", default="", help=f"comma separated stacks left in float ({', '.join(GROUPS)})")
    ap.add_argument("--outlier-ratio", type=float, default=6.0)
    ap.add_argument("--max-float-channels", type=int, default=64)
    ap.add_argument("--vae-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--no-vae-encoder", action="store_true", help="drop the AudioVAE encoder (disables voice cloning)")
    ap.add_argument("--calibration-precision", default=None, choices=["bfloat16", "float32"])
    args = ap.parse_args(argv)
    convert(
        args.src,
        args.dst,
        keep_float=[g for g in args.keep_float.split(",") if g],
        outlier_ratio=args.outlier_ratio,
        max_float_channels=args.max_float_channels,
        vae_dtype=args.vae_dtype,
        with_vae_encoder=not args.no_vae_encoder,
        calibration_precision=args.calibration_precision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
