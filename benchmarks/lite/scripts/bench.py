"""Benchmark one VoxCPM2 configuration on CPU: load time, memory, RTF, per-component time.

    python bench.py $VOXCPM2_DIR --loader upstream --dtype bfloat16       # as upstream VoxCPM
    python bench.py $VOXCPM2_DIR --loader lite --precision bfloat16       # lossless lite loader
    python bench.py $VOXCPM2_LITE_DIR --loader lite                       # int8
    python bench.py $VOXCPM2_LITE_DIR --loader lite --solver ab2 --steps 8
"""

import argparse
import json
import os
import resource
import sys
import time
from collections import defaultdict

import torch


def mem():
    out = {"peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    with open("/proc/self/status") as f:
        for line in f:
            for key in ("VmRSS", "RssAnon", "RssFile"):
                if line.startswith(key + ":"):
                    out[key.lower() + "_mb"] = int(line.split()[1]) / 1024.0
    return {k: round(v) for k, v in out.items()}


def load_stock(path, dtype, max_length=None, upstream_kv=False):
    from voxcpm.model import voxcpm2 as v2

    orig = v2.VoxCPMConfig.model_validate_json

    def patched(data):
        cfg = orig(data)
        cfg.dtype = dtype
        cfg.device = "cpu"
        if max_length:
            cfg.max_length = max_length
        return cfg

    v2.VoxCPMConfig.model_validate_json = staticmethod(patched)
    model = v2.VoxCPM2Model.from_local(path, optimize=False, device="cpu")
    if upstream_kv:
        # Reproduce upstream VoxCPM exactly: mask the whole static KV cache on
        # every step (this repo slices it whenever torch.compile is off).
        from voxcpm.modules.minicpm4.model import MiniCPMAttention

        for m in model.modules():
            if isinstance(m, MiniCPMAttention):
                m.slice_kv_cache = False
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument(
        "--loader",
        choices=["upstream", "stock", "lite"],
        default="lite",
        help="upstream: stock loader behaving like upstream VoxCPM; stock: stock loader of this repo (KV slicing on); lite: voxcpm.lite",
    )
    ap.add_argument("--dtype", default="bfloat16", help="stock loader dtype")
    ap.add_argument("--precision", default=None, help="lite loader on an original checkpoint: bfloat16|float32")
    ap.add_argument("--two-pass", default=None, help="int8: down_proj|all|none")
    ap.add_argument("--solver", default=None, help="euler|ab2")
    ap.add_argument("--reduce-range", type=int, default=None)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog, and then it runs away into the forest.")
    ap.add_argument("--ref", default=None, help="reference wav for cloning")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.time()
    if args.loader in ("upstream", "stock"):
        model = load_stock(args.model_dir, args.dtype, upstream_kv=args.loader == "upstream")
        label = f"{args.loader}/{args.dtype}"
    else:
        from voxcpm.lite import load_voxcpm2_lite

        model = load_voxcpm2_lite(
            args.model_dir,
            precision=args.precision,
            two_pass=args.two_pass,
            reduce_range=None if args.reduce_range is None else bool(args.reduce_range),
        )
        info = model.lite_info
        label = f"lite/{info['precision']}" + (f"({'7' if info['reduce_range'] else '8'}bit act)" if info["precision"] == "int8" else "")
    if args.solver:
        model.feat_decoder.solver = args.solver
        label += f"+{args.solver}"
    load_s = time.time() - t0
    m_load = mem()
    print(f"[{label}] load: {load_s:.1f}s  mem after load: {m_load}", flush=True)

    times = defaultdict(float)
    calls = defaultdict(int)

    def wrap(obj, name, lab):
        fn = getattr(obj, name)

        def timed(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                times[lab] += time.perf_counter() - t
                calls[lab] += 1

        setattr(obj, name, timed)

    wrap(model.base_lm, "forward", "base_lm.prefill")
    wrap(model.base_lm, "forward_step", "base_lm.step")
    wrap(model.residual_lm, "forward", "residual_lm.prefill")
    wrap(model.residual_lm, "forward_step", "residual_lm.step")
    wrap(model.feat_encoder, "forward", "feat_encoder")
    wrap(model.feat_decoder, "forward", "feat_decoder(DiT)")
    wrap(model.audio_vae, "decode", "audio_vae.decode")
    wrap(model.audio_vae, "encode", "audio_vae.encode")

    kw = dict(target_text=args.text, inference_timesteps=args.steps, cfg_value=args.cfg, seed=args.seed)
    if args.ref:
        kw["reference_wav_path"] = args.ref

    best = None
    for r in range(args.repeat):
        times.clear()
        calls.clear()
        t0 = time.perf_counter()
        if args.streaming:
            chunks = []
            first = None
            for c in model.generate_streaming(**kw):
                if first is None:
                    first = time.perf_counter() - t0
                chunks.append(c)
            wav = torch.cat(chunks, dim=-1)
        else:
            first = None
            wav = model.generate(**kw)
        gen_s = time.perf_counter() - t0
        dur = wav.shape[-1] / model.sample_rate
        res = {
            "label": label,
            "steps": args.steps,
            "cfg": args.cfg,
            "streaming": args.streaming,
            "load_s": round(load_s, 2),
            "audio_s": round(dur, 2),
            "gen_s": round(gen_s, 2),
            "rtf": round(gen_s / dur, 3),
            "first_chunk_s": None if first is None else round(first, 2),
            "mem_after_load": m_load,
            "mem_end": mem(),
            "breakdown_ms_per_call": {k: round(1000 * v / max(calls[k], 1), 1) for k, v in times.items()},
            "breakdown_s": {k: round(v, 2) for k, v in times.items()},
        }
        if best is None or res["rtf"] < best["rtf"]:
            best = res
        print(
            f"[{label}] run{r}: audio {dur:.2f}s gen {gen_s:.2f}s RTF {gen_s/dur:.2f}"
            + (f" first_chunk {first:.2f}s" if first else ""),
            flush=True,
        )
    print("mem end:", best["mem_end"])
    for k, v in sorted(times.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {v:7.2f}s {100*v/gen_s:5.1f}%  calls={calls[k]:3d}  {1000*v/max(calls[k],1):7.1f} ms/call")
    if args.out:
        import soundfile as sf

        sf.write(args.out, wav.squeeze(0).numpy(), model.sample_rate)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(best, f, indent=1)


if __name__ == "__main__":
    main()
