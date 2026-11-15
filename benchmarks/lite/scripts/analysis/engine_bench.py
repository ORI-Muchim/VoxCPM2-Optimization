"""Load time, memory and speed of the int8 model per PyTorch quantized engine.

    python engine_bench.py x86 | fbgemm | onednn
"""
import sys, time, torch, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import lite_dir, model_dir, out_path  # noqa: E402
eng = sys.argv[1]
torch.backends.quantized.engine = eng
from voxcpm.lite import load_voxcpm2_lite
t0 = time.time()
model = load_voxcpm2_lite(lite_dir(), verbose=False)
load = time.time() - t0
def anon():
    for l in open('/proc/self/status'):
        if l.startswith('RssAnon'): return int(l.split()[1]) // 1024
est = model.feat_decoder.estimator
torch.manual_seed(0)
with torch.inference_mode():
    b = 2
    x = torch.randn(b, 64, 4); mu = torch.randn(b, 2048); t = torch.rand(b); cond = torch.randn(b, 64, 4); d = torch.zeros(b)
    for _ in range(5): est(x, mu, t, cond, d)
    n = 40; t0 = time.perf_counter()
    for _ in range(n): est(x, mu, t, cond, d)
    dit = (time.perf_counter() - t0) / n * 1000
    # LM step
    emb = torch.randn(1, 40, 2048)
    out, kv = model.base_lm(inputs_embeds=emb, is_causal=True); model.base_lm.kv_cache.fill_caches(kv)
    h = torch.randn(1, 2048)
    for _ in range(3): model.base_lm.forward_step(h, torch.tensor([model.base_lm.kv_cache.step()]))
    n = 20; t0 = time.perf_counter()
    for _ in range(n): model.base_lm.forward_step(h, torch.tensor([model.base_lm.kv_cache.step()]))
    lm = (time.perf_counter() - t0) / n * 1000
print(f"engine={eng:7s} load {load:5.1f}s  RssAnon {anon()} MB  DiT eval(b=2) {dit:5.1f} ms  base_lm.step {lm:5.1f} ms")
