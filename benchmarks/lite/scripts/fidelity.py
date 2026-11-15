"""Teacher-forced fidelity measurements.

record : run the fp32 model, log every sub-module call (inputs + outputs).
replay : feed the logged inputs to another configuration and report, per
         sub-module, the SNR of its outputs against the fp32 outputs.
solver : on the logged LocDiT inputs, compare samplers against a converged
         (many-step) solution.

    python fidelity.py record
    python fidelity.py replay stock_bf16 | lite:<two_pass>:<reduce_range>
    python fidelity.py solver
"""

import math
import sys
import warnings

import torch

warnings.simplefilter("ignore")
sys.path.insert(0, "/home/user/work/lite")
TRACE = "/home/user/work/out/trace_fp32.pt"
SRC = "/home/user/models/VoxCPM2"
LITE = "/home/user/models/VoxCPM2-lite"
TEXTS = [
    "After a long day at work, he likes to cook dinner and listen to quiet music.",
    "주말에는 가족들과 함께 바닷가로 여행을 떠날 계획이에요.",
]
MAX_LEN = 16
NOISE_SEED = 777000


def snr_db(ref, x):
    ref, x = ref.float(), x.float()
    return 10 * math.log10(ref.pow(2).sum().item() / max((ref - x).pow(2).sum().item(), 1e-30))


def record():
    from voxcpm.lite import load_voxcpm2_lite

    model = load_voxcpm2_lite(SRC, precision="float32", verbose=False)
    trace = []
    cur = None
    n_dit = [0]

    def wrap(obj, name, kind):
        fn = getattr(obj, name)

        def logged(*a, **k):
            if kind == "dit":
                torch.manual_seed(NOISE_SEED + n_dit[0])
                n_dit[0] += 1
            out = fn(*a, **k)
            if kind in ("base_prefill", "res_prefill"):
                cur.append((kind, k["inputs_embeds"].clone(), out[0].clone()))
            elif kind in ("base_step", "res_step"):
                cur.append((kind, a[0].clone(), a[1].clone(), out.clone()))
            elif kind == "enc":
                cur.append((kind, a[0].clone(), out.clone()))
            elif kind == "dit":
                cur.append((kind, k["mu"].clone(), k["cond"].clone(), out.clone()))
            return out

        setattr(obj, name, logged)

    wrap(model.base_lm, "forward", "base_prefill")
    wrap(model.base_lm, "forward_step", "base_step")
    wrap(model.residual_lm, "forward", "res_prefill")
    wrap(model.residual_lm, "forward_step", "res_step")
    wrap(model.feat_encoder, "forward", "enc")
    wrap(model.feat_decoder, "forward", "dit")
    for i, t in enumerate(TEXTS):
        cur = []
        n_dit[0] = 1000 * i
        model.generate(target_text=t, max_len=MAX_LEN, inference_timesteps=10, cfg_value=2.0, seed=100 + i)
        trace.append(cur)
    torch.save(trace, TRACE)
    print("recorded", [len(c) for c in trace])


def load_candidate(spec):
    if spec == "stock_bf16":
        from bench import load_stock

        return load_stock(SRC, "bfloat16")
    from voxcpm.lite import load_voxcpm2_lite

    if spec == "lite_bf16":
        return load_voxcpm2_lite(SRC, precision="bfloat16", verbose=False)
    if spec == "lite_f32":
        return load_voxcpm2_lite(SRC, precision="float32", verbose=False)
    _, tp, rr = spec.split(":")
    return load_voxcpm2_lite(LITE, two_pass=tp, reduce_range=bool(int(rr)), verbose=False)


@torch.inference_mode()
def replay(spec):
    model = load_candidate(spec)
    dt = next(p for n, p in model.named_parameters() if "norm" in n).dtype
    trace = torch.load(TRACE)
    acc = {}

    def add(kind, ref, out):
        num, den = acc.setdefault(kind, [0.0, 0.0])
        acc[kind] = [num + (ref.float() - out.float()).pow(2).sum().item(), den + ref.float().pow(2).sum().item()]

    for ui, utt in enumerate(trace):
        n_dit = 1000 * ui
        for ev in utt:
            kind = ev[0]
            if kind in ("base_prefill", "res_prefill"):
                lm = model.base_lm if kind == "base_prefill" else model.residual_lm
                out, kv = lm(inputs_embeds=ev[1].to(dt), is_causal=True)
                lm.kv_cache.fill_caches(kv)
                add(kind, ev[2], out)
            elif kind in ("base_step", "res_step"):
                lm = model.base_lm if kind == "base_step" else model.residual_lm
                pos = torch.tensor([lm.kv_cache.step()])
                assert int(pos) == int(ev[2])
                add(kind, ev[3], lm.forward_step(ev[1].to(dt), pos))
            elif kind == "enc":
                add(kind, ev[2], model.feat_encoder(ev[1].to(dt)))
            elif kind == "dit":
                torch.manual_seed(NOISE_SEED + n_dit)
                n_dit += 1
                out = model.feat_decoder(mu=ev[1].to(dt), patch_size=model.patch_size, cond=ev[2].to(dt), n_timesteps=10, cfg_value=2.0)
                add(kind, ev[3], out)
    row = "  ".join(f"{k}={10*math.log10(d/max(n,1e-30)):5.1f}dB" for k, (n, d) in acc.items())
    print(f"{spec:28s} {row}", flush=True)


# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #
def cfg_velocity(cfm, x, mu, t, cond, cfg_value, use_cfg=True):
    """One guided velocity evaluation, exactly as UnifiedCFM.solve_euler does it."""
    b = x.size(0)
    if not use_cfg:
        dt_in = torch.zeros([b], dtype=x.dtype)
        return cfm.estimator(x, mu, t.expand(b), cond, dt_in)
    x_in = torch.cat([x, x], 0)
    mu_in = torch.cat([mu, torch.zeros_like(mu)], 0)
    t_in = t.expand(2 * b)
    cond_in = torch.cat([cond, cond], 0)
    dt_in = torch.zeros([2 * b], dtype=x.dtype)
    v = cfm.estimator(x_in, mu_in, t_in, cond_in, dt_in)
    v_c, v_u = v[:b], v[b:]
    st = cfm.optimized_scale(v_c.reshape(b, -1), v_u.reshape(b, -1)).view(b, 1, 1)
    return v_u * st + cfg_value * (v_c - v_u * st)


def t_grid(n, sway=1.0):
    t = torch.linspace(1, 0, n + 1)
    return t + sway * (torch.cos(torch.pi / 2 * t) - 1 + t)


@torch.inference_mode()
def sample(cfm, z, mu, cond, n, cfg_value=2.0, method="euler", cfg_until=0.0):
    """method: euler | ab2 (2nd-order Adams-Bashforth, same number of evaluations).
    cfg_until: below this t the unconditional branch is skipped (plain conditional velocity)."""
    ts = t_grid(n)
    x = z
    zero_init = max(1, int(len(ts) * 0.04))
    v_prev, h_prev = None, None
    evals = 0
    for step in range(1, len(ts)):
        t, h = ts[step - 1], ts[step - 1] - ts[step]
        if step <= zero_init:
            continue  # CFG-Zero*: zero velocity on the first step(s)
        v = cfg_velocity(cfm, x, mu, t, cond, cfg_value, use_cfg=bool(t > cfg_until))
        evals += 1
        if method == "ab2" and v_prev is not None:
            r = h / h_prev
            x = x - h * ((1 + r / 2) * v - (r / 2) * v_prev)
        else:
            x = x - h * v
        v_prev, h_prev = v, h
    return x, evals


def cfg_velocity_multi(cfm, xs, mu, ts, cond, cfg_value):
    """Guided velocities at several (x, t) points in ONE batched estimator call."""
    k = len(xs)
    x = torch.cat(xs, 0)
    mu_k, cond_k = mu.expand(k, -1), cond.expand(k, -1, -1)
    t = torch.stack([tt.reshape(()) for tt in ts])
    x_in = torch.cat([x, x], 0)
    mu_in = torch.cat([mu_k, torch.zeros_like(mu_k)], 0)
    v = cfm.estimator(x_in, mu_in, torch.cat([t, t], 0), torch.cat([cond_k, cond_k], 0), torch.zeros([2 * k], dtype=x.dtype))
    v_c, v_u = v[:k], v[k:]
    st = cfm.optimized_scale(v_c.reshape(k, -1), v_u.reshape(k, -1)).view(k, 1, 1)
    out = v_u * st + cfg_value * (v_c - v_u * st)
    return [out[i : i + 1] for i in range(k)]


@torch.inference_mode()
def sample_lag2(cfm, z, mu, cond, n_evals, cfg_value=2.0, extrapolate=False):
    """Two Euler sub-steps per estimator call. The state at the second
    sub-step is predicted with the latest known velocity, so both velocities
    come out of one batched evaluation (rows are cheap when the estimator is
    memory-bandwidth bound)."""
    ts = t_grid(2 * n_evals)  # fine grid; first interval is the CFG-Zero* zero step
    x = z
    i = 1
    # first evaluation: single point (no velocity history yet)
    v = cfg_velocity(cfm, x, mu, ts[i], cond, cfg_value)
    x = x - (ts[i] - ts[i + 1]) * v
    v_hist = [(ts[i], v)]
    i += 1
    evals = 1
    while i < len(ts) - 1:
        h1 = ts[i] - ts[i + 1]
        last = i + 1 >= len(ts) - 1
        v_hat = v_hist[-1][1]
        if extrapolate and len(v_hist) >= 2:
            (t0, v0), (t1, v1) = v_hist[-2], v_hist[-1]
            v_hat = v1 + (v1 - v0) * ((t1 - ts[i]) / (t0 - t1))
        if last:
            v_a = cfg_velocity(cfm, x, mu, ts[i], cond, cfg_value)
            x = x - h1 * v_a
            i += 1
        else:
            h2 = ts[i + 1] - ts[i + 2]
            x_m = x - h1 * v_hat
            v_a, v_m = cfg_velocity_multi(cfm, [x, x_m], mu, [ts[i], ts[i + 1]], cond, cfg_value)
            x = x - h1 * v_a - h2 * v_m
            v_hist += [(ts[i], v_a), (ts[i + 1], v_m)]
            i += 2
        evals += 1
    return x, evals


@torch.inference_mode()
def solver(specs):
    from voxcpm.lite import load_voxcpm2_lite

    model = load_voxcpm2_lite(SRC, precision="float32", verbose=False)
    cfm = model.feat_decoder
    trace = torch.load(TRACE)
    dits = []
    for ui, utt in enumerate(trace):
        k = 0
        for ev in utt:
            if ev[0] == "dit":
                if k % 2 == 0:  # every other patch is plenty
                    dits.append((NOISE_SEED + 1000 * ui + k, ev[1], ev[2], ev[3]))
                k += 1
    zs = []
    for seed, mu, cond, _ in dits:
        torch.manual_seed(seed)
        zs.append(torch.randn((1, cfm.in_channels, model.patch_size)))
    # sanity: our Euler-10 must reproduce the recorded stock output
    x10 = [sample(cfm, z, mu, cond.contiguous(), 10)[0] for z, (_, mu, cond, _) in zip(zs, dits)]
    chk = sum(snr_db(ref, x) for x, (_, _, _, ref) in zip(x10, dits)) / len(dits)
    print(f"reimplemented euler-10 vs recorded stock output: {chk:.1f} dB ({len(dits)} patches)")
    ref = [sample(cfm, z, mu, cond.contiguous(), 96)[0] for z, (_, mu, cond, _) in zip(zs, dits)]
    print(f"{'sampler':28s} {'evals':>5s} {'SNR vs converged (dB)':>22s}")
    for spec in specs:
        parts = spec.split(":")
        method, n = parts[0], int(parts[1])
        cfg_until = float(parts[2]) if len(parts) > 2 else 0.0
        num = den = 0.0
        ev = 0
        for z, (_, mu, cond, _), r in zip(zs, dits, ref):
            if method.startswith("lag2"):
                x, ev = sample_lag2(cfm, z, mu, cond.contiguous(), n, extrapolate=method == "lag2x")
            else:
                x, ev = sample(cfm, z, mu, cond.contiguous(), n, method=method, cfg_until=cfg_until)
            num += (r - x).pow(2).sum().item()
            den += r.pow(2).sum().item()
        print(f"{spec:28s} {ev:5d} {10*math.log10(den/num):22.1f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "solver":
        solver(sys.argv[2:])
        sys.exit(0)
    if cmd == "record":
        record()
    elif cmd == "replay":
        for spec in sys.argv[2:]:
            replay(spec)
