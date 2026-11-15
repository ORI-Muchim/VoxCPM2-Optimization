"""Op-level profile of one LocDiT solve (int8 lite checkpoint)."""
import sys, time, torch, warnings
warnings.simplefilter('ignore')
from voxcpm.lite import load_voxcpm2_lite
tp = sys.argv[1] if len(sys.argv) > 1 else "down_proj"
model = load_voxcpm2_lite("/home/user/models/VoxCPM2-lite", two_pass=tp, verbose=False)
torch.manual_seed(0)
mu = torch.randn(1, 2048); cond = torch.randn(1, 64, 4)
with torch.inference_mode():
    for _ in range(3):
        model.feat_decoder(mu=mu, patch_size=4, cond=cond, n_timesteps=10, cfg_value=2.0)
    t = time.perf_counter()
    for _ in range(5):
        model.feat_decoder(mu=mu, patch_size=4, cond=cond, n_timesteps=10, cfg_value=2.0)
    print(f"two_pass={tp}: DiT solve {(time.perf_counter()-t)/5*1000:.1f} ms")
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            model.feat_decoder(mu=mu, patch_size=4, cond=cond, n_timesteps=10, cfg_value=2.0)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=28, max_name_column_width=44))
