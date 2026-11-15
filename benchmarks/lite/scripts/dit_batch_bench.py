"""LocDiT estimator time per evaluation vs batch size (rows = batch * 11 tokens)."""
import sys, time, torch, warnings
warnings.simplefilter("ignore")
from voxcpm.lite import load_voxcpm2_lite
spec = sys.argv[1]
if spec == "bf16":
    model = load_voxcpm2_lite("/home/user/models/VoxCPM2", precision="bfloat16", verbose=False); dt = torch.bfloat16
else:
    model = load_voxcpm2_lite("/home/user/models/VoxCPM2-lite", two_pass=spec, verbose=False); dt = torch.float32
est = model.feat_decoder.estimator
torch.manual_seed(0)
res = []
with torch.inference_mode():
    for b in (1, 2, 4, 6, 8):
        x = torch.randn(b, 64, 4, dtype=dt); mu = torch.randn(b, 2048, dtype=dt); t = torch.rand(b, dtype=dt); cond = torch.randn(b, 64, 4, dtype=dt); d = torch.zeros(b, dtype=dt)
        for _ in range(5): est(x, mu, t, cond, d)
        n = 30; t0 = time.perf_counter()
        for _ in range(n): est(x, mu, t, cond, d)
        res.append(f"b={b}: {(time.perf_counter()-t0)/n*1000:.1f}ms")
print(f"{spec:10s}", "  ".join(res))
