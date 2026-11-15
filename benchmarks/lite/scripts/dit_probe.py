"""Record fp32 LocDiT estimator calls (inputs, per-layer hidden states, output) and replay them."""
import math, sys, warnings, torch
warnings.simplefilter("ignore")
from voxcpm.lite import load_voxcpm2_lite
P = "/home/user/work/out/dit_probe.pt"
def snr(ref, x):
    return 10 * math.log10(ref.float().pow(2).sum().item() / max((ref.float() - x.float()).pow(2).sum().item(), 1e-30))
def hook_layers(est, store):
    hs = []
    for i, layer in enumerate(est.decoder.layers):
        def hook(m, inp, out, _i=i):
            store.setdefault(_i, out[0].detach().clone())  # must return None: a returned value replaces the output
        hs.append(layer.register_forward_hook(hook))
    return hs
if sys.argv[1] == "record":
    model = load_voxcpm2_lite("/home/user/models/VoxCPM2", precision="float32", verbose=False)
    est = model.feat_decoder.estimator
    calls = []
    orig = est.forward
    def logged(x, mu, t, cond, dt):
        store = {}
        hs = hook_layers(est, store)
        out = orig(x, mu, t, cond, dt)
        for h in hs: h.remove()
        calls.append(dict(x=x.clone(), mu=mu.clone(), t=t.clone(), cond=cond.clone(), dt=dt.clone(), out=out.clone(), layers=store))
        return out
    est.forward = logged
    model.generate(target_text="After a long day at work, he likes to cook dinner.", max_len=3, inference_timesteps=10, cfg_value=2.0, seed=5)
    torch.save(calls, P); print("recorded", len(calls), "estimator calls")
else:
    calls = torch.load(P)
    for spec in sys.argv[2:]:
        if spec == "bf16":
            model = load_voxcpm2_lite("/home/user/models/VoxCPM2", precision="bfloat16", verbose=False); dtp = torch.bfloat16
        else:
            _, tp, rr = spec.split(":")
            model = load_voxcpm2_lite("/home/user/models/VoxCPM2-lite", two_pass=tp, reduce_range=bool(int(rr)), verbose=False); dtp = torch.float32
        est = model.feat_decoder.estimator
        print(f"== {spec}")
        with torch.inference_mode():
            for ci in (0, 4, 8, 9, 13, 17):
                c = calls[ci]; store = {}
                hs = hook_layers(est, store)
                out = est(c["x"].to(dtp), c["mu"].to(dtp), c["t"].to(dtp), c["cond"].to(dtp), c["dt"].to(dtp))
                for h in hs: h.remove()
                lay = " ".join(f"{snr(c['layers'][i], store[i]):.0f}" for i in range(12))
                # per-CFG-branch output SNR
                b = c["out"].shape[0] // 2
                print(f"  call {ci:2d} t={c['t'][0].item():.3f}  out SNR cond={snr(c['out'][:b], out[:b]):5.1f} uncond={snr(c['out'][b:], out[b:]):5.1f} dB | per-layer hidden SNR: {lay}")
