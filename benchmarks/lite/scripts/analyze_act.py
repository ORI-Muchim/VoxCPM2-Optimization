"""Activation statistics of every Linear input during real fp32 generation, and the
output SNR of several activation-quantization schemes.

Pass 1 collects per-input-channel abs-max. Pass 2 simulates schemes:
  tensor7 : torch dynamic quant as-is (one min/max for the whole input, 7 bit)
  row7    : each row (token) scaled to its own abs-max, 7 bit symmetric
  row8    : same, 8 bit
  split7  : row7 after removing the K largest-range channels (handled in float)
"""

import argparse
import json
import sys
from collections import defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, "/home/user/openbmb/voxcpm/src")
from voxcpm.lite import load_voxcpm2_lite  # noqa: E402
from voxcpm.lite.loader import group_of  # noqa: E402

TEXTS = [
    "Scientists have discovered a new species of deep sea fish living near volcanic vents.",
    "이 근처에 맛있는 한식당이 있으면 추천해 주시겠어요?",
]


def kind_of(name):
    leaf = name.rsplit(".", 1)[-1]
    return f"{group_of(name)}:{leaf}"


def quant_sym(x, bits, scale):
    qmax = 2 ** (bits - 1) - 1
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


def quant_tensor_asym(x, bits):
    lo, hi = min(x.min().item(), 0.0), max(x.max().item(), 0.0)
    n = 2**bits - 1
    scale = max((hi - lo) / n, 1e-12)
    zp = round(-lo / scale)
    return (torch.round(x / scale + zp).clamp(0, n) - zp) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=14)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", default="/home/user/work/out/act_stats.pt")
    args = ap.parse_args()

    model = load_voxcpm2_lite("/home/user/models/VoxCPM2", precision="float32")
    linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and group_of(n) != "audio_vae"}

    chan_max = {n: torch.zeros(m.in_features) for n, m in linears.items()}
    crest = defaultdict(list)

    def run():
        for i, t in enumerate(TEXTS):
            model.generate(target_text=t, max_len=args.max_len, inference_timesteps=10, cfg_value=2.0, seed=1000 + i)

    import os

    skip_pass1 = os.path.exists(args.out)
    if skip_pass1:
        chan_max = torch.load(args.out)["chan_max"]
    # ---- pass 1
    hooks = []
    for n, m in linears.items():

        def pre(mod, inp, _n=n):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            chan_max[_n] = torch.maximum(chan_max[_n], x.abs().amax(0))
            crest[_n].append((x.abs().amax(1) / x.pow(2).mean(1).sqrt().clamp_min(1e-9)).max().item())

        if not skip_pass1:
            hooks.append(m.register_forward_pre_hook(pre))
    if not skip_pass1:
        run()
    for h in hooks:
        h.remove()
    torch.save({"chan_max": chan_max}, args.out)
    top_idx32 = {n: torch.topk(c, min(32, c.numel())).indices for n, c in chan_max.items()}

    top_idx = {n: torch.topk(c, min(args.k, c.numel())).indices for n, c in chan_max.items()}

    # ---- pass 2
    num = defaultdict(lambda: defaultdict(float))
    den = defaultdict(float)
    hooks = []
    for n, m in linears.items():

        def pre(mod, inp, _n=n):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            W = mod.weight.float()
            y = x @ W.T
            den[_n] += y.pow(2).sum().item()

            def err(xq):
                return ((xq - x) @ W.T).pow(2).sum().item()

            idx = top_idx[_n]
            xm = x.clone()
            xm[:, idx] = 0
            rs2 = xm.abs().amax(1, keepdim=True).clamp_min(1e-9)
            xn = xm / rs2
            # split + row scale, asymmetric per-tensor grid as the kernel does
            xq = quant_tensor_asym(xn, 8) * rs2
            xq[:, idx] = x[:, idx]
            num[_n]["split8"] += err(xq)
            # split without row scaling
            xq = quant_tensor_asym(xm, 8)
            xq[:, idx] = x[:, idx]
            num[_n]["split8_norow"] += err(xq)
            # 32 float channels
            idx32 = top_idx32[_n]
            xm32 = x.clone()
            xm32[:, idx32] = 0
            r32 = xm32.abs().amax(1, keepdim=True).clamp_min(1e-9)
            xq = quant_tensor_asym(xm32 / r32, 8) * r32
            xq[:, idx32] = x[:, idx32]
            num[_n]["split8_k32"] += err(xq)
            # two-pass residual, no split
            rs = x.abs().amax(1, keepdim=True).clamp_min(1e-9)
            x0 = x / rs
            q1 = quant_tensor_asym(x0, 8)
            xq = (q1 + quant_tensor_asym(x0 - q1, 8)) * rs
            num[_n]["2pass8"] += err(xq)
            q1 = quant_tensor_asym(x0, 7)
            xq = (q1 + quant_tensor_asym(x0 - q1, 7)) * rs
            num[_n]["2pass7"] += err(xq)
            # split + two-pass at 7 bit
            q1 = quant_tensor_asym(xn, 7)
            xq = (q1 + quant_tensor_asym(xn - q1, 7)) * rs2
            xq[:, idx] = x[:, idx]
            num[_n]["split7_2pass"] += err(xq)

        hooks.append(m.register_forward_pre_hook(pre))
    run()
    for h in hooks:
        h.remove()

    import math

    def snr(n, s):
        return 10 * math.log10(den[n] / max(num[n][s], 1e-30))

    schemes = ["split8", "split8_norow", "split8_k32", "2pass8", "2pass7", "split7_2pass"]
    by_kind = defaultdict(list)
    for n in linears:
        by_kind[kind_of(n)].append(n)
    print(f"{'kind':34s} {'n':>3s} {'crest(max)':>10s}  " + "  ".join(f"{s+' min/mean':>16s}" for s in schemes))
    rows = {}
    for kind, names in sorted(by_kind.items()):
        cr = max((max(crest[n]) for n in names if crest[n]), default=0.0)
        cells = []
        for s in schemes:
            v = [snr(n, s) for n in names]
            cells.append(f"{min(v):7.1f}/{sum(v)/len(v):5.1f} dB")
        rows[kind] = cells
        print(f"{kind:34s} {len(names):3d} {cr:10.1f}  " + "  ".join(f"{c:>16s}" for c in cells))
    json.dump({n: {s: snr(n, s) for s in schemes} for n in linears}, open(args.out.replace(".pt", "_snr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
