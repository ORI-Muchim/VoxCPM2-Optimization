"""Per-component profile of VoxCPM2 inference on CPU.

Usage: python profile_baseline.py <model_dir> [--dtype float32|bfloat16] [--steps 10] [--cfg 2.0]
Reports: parameter counts per sub-module, wall time per sub-module, RTF, peak RSS.
"""

import argparse
import json
import os
import resource
import sys
import time
from collections import defaultdict

import torch


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog, and then it runs away into the forest.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-length", type=int, default=None, help="override KV cache max_length")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from voxcpm.model import voxcpm2 as v2

    # Force CPU dtype without editing the checkpoint dir.
    orig_validate = v2.VoxCPMConfig.model_validate_json

    def patched(data):
        cfg = orig_validate(data)
        cfg.dtype = args.dtype
        cfg.device = "cpu"
        if args.max_length:
            cfg.max_length = args.max_length
        return cfg

    v2.VoxCPMConfig.model_validate_json = staticmethod(patched)

    t0 = time.time()
    model = v2.VoxCPM2Model.from_local(args.model_dir, optimize=False, device="cpu")
    load_s = time.time() - t0
    print(f"load: {load_s:.1f}s  rss_after_load: {rss_mb():.0f} MB", flush=True)

    groups = {
        "base_lm": model.base_lm,
        "residual_lm": model.residual_lm,
        "feat_encoder": model.feat_encoder,
        "feat_decoder(DiT)": model.feat_decoder,
        "audio_vae": model.audio_vae,
    }
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
    total = sum(p.numel() for p in model.parameters())
    counts["projections/other"] = total - sum(counts.values())
    counts["base_lm.embed_tokens"] = model.base_lm.embed_tokens.weight.numel()
    if hasattr(model.audio_vae, "encoder"):
        counts["audio_vae.encoder"] = sum(p.numel() for p in model.audio_vae.encoder.parameters())
        counts["audio_vae.decoder"] = sum(p.numel() for p in model.audio_vae.decoder.parameters())
    print("params (M):", json.dumps({k: round(v / 1e6, 1) for k, v in counts.items()}, indent=1))
    print(f"total params: {total/1e6:.1f} M", flush=True)

    # Timing wrappers
    times = defaultdict(float)
    calls = defaultdict(int)

    def wrap(obj, name, label):
        fn = getattr(obj, name)

        def timed(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                times[label] += time.perf_counter() - t
                calls[label] += 1

        setattr(obj, name, timed)

    wrap(model.base_lm, "forward", "base_lm.prefill")
    wrap(model.base_lm, "forward_step", "base_lm.step")
    wrap(model.residual_lm, "forward", "residual_lm.prefill")
    wrap(model.residual_lm, "forward_step", "residual_lm.step")
    wrap(model.feat_encoder, "forward", "feat_encoder")
    wrap(model.feat_decoder, "forward", "feat_decoder(DiT)")
    wrap(model.audio_vae, "decode", "audio_vae.decode")
    wrap(model.audio_vae, "encode", "audio_vae.encode")
    wrap(model.base_lm.kv_cache, "fill_caches", "kv_cache.fill(base)")
    wrap(model.residual_lm.kv_cache, "fill_caches", "kv_cache.fill(res)")

    t0 = time.perf_counter()
    wav = model.generate(
        target_text=args.text,
        inference_timesteps=args.steps,
        cfg_value=args.cfg,
        seed=args.seed,
    )
    gen_s = time.perf_counter() - t0
    dur = wav.shape[-1] / model.sample_rate
    print(f"\naudio: {dur:.2f}s @ {model.sample_rate}Hz   gen: {gen_s:.1f}s   RTF: {gen_s/dur:.2f}")
    print(f"peak rss: {rss_mb():.0f} MB")
    print("time breakdown:")
    acc = 0.0
    for k, v in sorted(times.items(), key=lambda kv: -kv[1]):
        acc += v
        print(f"  {k:24s} {v:8.2f}s  {100*v/gen_s:5.1f}%  calls={calls[k]}  per_call={1000*v/max(calls[k],1):.1f}ms")
    print(f"  {'(untracked)':24s} {gen_s-acc:8.2f}s  {100*(gen_s-acc)/gen_s:5.1f}%")

    if args.out:
        import soundfile as sf

        sf.write(args.out, wav.squeeze(0).numpy(), model.sample_rate)
        print("saved", args.out)


if __name__ == "__main__":
    main()
