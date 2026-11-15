"""Tests for the lite (low-memory / int8) VoxCPM2 runtime.

They build a tiny random VoxCPM2 checkpoint in the original on-disk layout and
check that the lite loader reproduces the stock loader.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("pydantic")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TINY_CONFIG = {
    "architecture": "voxcpm2",
    "lm_config": {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_size": 32,
        "intermediate_size": 64,
        "max_position_embeddings": 256,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "kv_channels": 8,
        "rope_scaling": {
            "type": "longrope",
            "long_factor": [1.0, 1.1, 1.2, 1.3],
            "short_factor": [1.0, 1.1, 1.2, 1.3],
            "original_max_position_embeddings": 256,
        },
        "vocab_size": 128,
        "use_mup": False,
        "scale_emb": 12,
        "dim_model_base": 256,
        "scale_depth": 1.4,
    },
    "patch_size": 2,
    "feat_dim": 8,
    "scalar_quantization_latent_dim": 16,
    "scalar_quantization_scale": 9,
    "residual_lm_num_layers": 1,
    "residual_lm_no_rope": True,
    "encoder_config": {"hidden_dim": 32, "ffn_dim": 64, "num_heads": 4, "num_layers": 1, "kv_channels": 8},
    "dit_config": {
        "hidden_dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "num_layers": 1,
        "kv_channels": 8,
        "cfm_config": {"sigma_min": 1e-6, "solver": "euler", "t_scheduler": "log-norm", "inference_cfg_rate": 2.0},
    },
    "audio_vae_config": {
        "encoder_dim": 8,
        "encoder_rates": [2, 2],
        "latent_dim": 8,
        "decoder_dim": 32,
        "decoder_rates": [2, 2],
        "sr_bin_boundaries": [20000, 30000, 40000],
        "sample_rate": 16000,
        "out_sample_rate": 16000,
    },
    "max_length": 64,
    "device": "cpu",
    "dtype": "float32",
}

TEXT = "hello world this is a test"


def _write_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["<unk>", "<s>", "</s>"] + TEXT.split()
    vocab = {w: i for i, w in enumerate(words)}
    for i in range(len(vocab), TINY_CONFIG["lm_config"]["vocab_size"]):
        vocab[f"[tok{i}]"] = i
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "LlamaTokenizerFast", "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"})
    )


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    from safetensors.torch import save_file
    from transformers import LlamaTokenizerFast

    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEV2

    path = tmp_path_factory.mktemp("tiny_voxcpm2")
    (path / "config.json").write_text(json.dumps(TINY_CONFIG))
    _write_tokenizer(path)

    torch.manual_seed(0)
    config = VoxCPMConfig.model_validate(TINY_CONFIG)
    vae = AudioVAEV2(config=config.audio_vae_config)
    model = VoxCPM2Model(config, LlamaTokenizerFast.from_pretrained(str(path)), vae, None, device="cpu")
    state = model.state_dict()
    save_file(
        {k: v.contiguous() for k, v in state.items() if not k.startswith("audio_vae.")},
        str(path / "model.safetensors"),
    )
    torch.save({"state_dict": vae.state_dict()}, str(path / "audiovae.pth"))
    return path


def _stock(path):
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    return VoxCPM2Model.from_local(str(path), optimize=False, device="cpu")


def _gen(model, **kw):
    return model.generate(target_text=TEXT, max_len=6, inference_timesteps=4, cfg_value=2.0, seed=7, **kw)


def test_float32_matches_stock_loader(tiny_ckpt):
    from voxcpm.lite import load_voxcpm2_lite

    ref = _gen(_stock(tiny_ckpt))
    lite = load_voxcpm2_lite(str(tiny_ckpt), precision="float32", verbose=False)
    out = _gen(lite)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)


def test_float32_streaming_matches_stock_loader(tiny_ckpt):
    from voxcpm.lite import load_voxcpm2_lite

    kw = dict(target_text=TEXT, max_len=6, inference_timesteps=4, cfg_value=2.0, seed=7)
    ref = torch.cat(list(_stock(tiny_ckpt).generate_streaming(**kw)), dim=-1)
    lite = load_voxcpm2_lite(str(tiny_ckpt), precision="float32", verbose=False)
    out = torch.cat(list(lite.generate_streaming(**kw)), dim=-1)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)


def test_weight_norm_is_folded_and_decode_matches(tiny_ckpt):
    from voxcpm.lite import load_voxcpm2_lite

    stock = _stock(tiny_ckpt)
    lite = load_voxcpm2_lite(str(tiny_ckpt), precision="float32", verbose=False)
    assert not any(n.endswith(("weight_g", "weight_v")) for n, _ in lite.audio_vae.named_parameters())
    z = torch.randn(1, TINY_CONFIG["feat_dim"], 12)
    with torch.inference_mode():
        assert torch.allclose(lite.audio_vae.decode(z), stock.audio_vae.decode(z), atol=1e-5)


def test_vae_encoder_is_lazy(tiny_ckpt):
    from voxcpm.lite import load_voxcpm2_lite

    stock = _stock(tiny_ckpt)
    lite = load_voxcpm2_lite(str(tiny_ckpt), precision="float32", verbose=False)
    audio = torch.randn(1, 1600)
    with torch.inference_mode():
        assert torch.allclose(lite.audio_vae.encode(audio, 16000), stock.audio_vae.encode(audio, 16000), atol=1e-5)

    no_enc = load_voxcpm2_lite(str(tiny_ckpt), precision="float32", load_vae_encoder=False, verbose=False)
    with pytest.raises(RuntimeError):
        no_enc.audio_vae.encode(audio, 16000)


def test_kv_cache_slicing_matches_masking():
    from voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel
    from voxcpm.modules.minicpm4.model import MiniCPMAttention

    torch.manual_seed(0)
    lm = MiniCPMModel(MiniCPM4Config(**TINY_CONFIG["lm_config"])).eval()
    lm.setup_cache(1, 32, "cpu", torch.float32)
    prefix = torch.randn(1, 5, 32)
    steps = [torch.randn(1, 32) for _ in range(4)]

    def run(sliced):
        for m in lm.modules():
            if isinstance(m, MiniCPMAttention):
                m.slice_kv_cache = sliced
        with torch.inference_mode():
            _, cache = lm(inputs_embeds=prefix, is_causal=True)
            lm.kv_cache.fill_caches(cache)
            return [lm.forward_step(s, torch.tensor([lm.kv_cache.step()])).clone() for s in steps]

    for a, b in zip(run(False), run(True)):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.fixture(scope="module")
def tiny_lite(tiny_ckpt, tmp_path_factory):
    from voxcpm.lite.convert import convert

    dst = tmp_path_factory.mktemp("tiny_voxcpm2_lite")
    stats = convert(str(tiny_ckpt), str(dst), vae_dtype="float32", calibration_precision="float32", verbose=False)
    assert stats["int8_params_m"] > 0
    return dst


def _block_linears(model):
    from voxcpm.lite.loader import block_linear_groups

    return [model.get_submodule(f"{b}.{leaf}") for b, leaves in block_linear_groups(model) for leaf in leaves]


def test_int8_requires_conversion(tiny_ckpt):
    from voxcpm.lite import load_voxcpm2_lite

    with pytest.raises(ValueError, match="voxcpm.lite.convert"):
        load_voxcpm2_lite(str(tiny_ckpt), precision="int8", verbose=False)


def test_int8_checkpoint_loads_and_generates(tiny_lite):
    from voxcpm.lite import Int8Embedding, Int8Linear, Int8LinearSlice, load_voxcpm2_lite

    cfg = json.loads((tiny_lite / "config.json").read_text())
    assert cfg["lite"]["precision"] == "int8"

    lite = load_voxcpm2_lite(str(tiny_lite), verbose=False)
    assert isinstance(lite.base_lm.embed_tokens, Int8Embedding)
    assert all(isinstance(m, (Int8Linear, Int8LinearSlice)) for m in _block_linears(lite))
    # q/k/v and gate/up share one fused integer matmul
    attn = lite.base_lm.layers[0].self_attn
    assert attn.q_proj._shared is attn.k_proj._shared is attn.v_proj._shared
    # two-pass residual quantization on down_proj by default
    assert lite.base_lm.layers[0].mlp.down_proj.two_pass
    assert not lite.base_lm.layers[0].self_attn.o_proj.two_pass
    # ... but not in the LocDiT (see DEFAULT_TWO_PASS)
    assert not lite.feat_decoder.estimator.decoder.layers[0].mlp.down_proj.two_pass
    # everything outside the transformer blocks stays float
    assert isinstance(lite.stop_head, torch.nn.Linear)
    assert isinstance(lite.feat_decoder.estimator.out_proj, torch.nn.Linear)
    assert torch.isfinite(_gen(lite)).all()


@pytest.mark.parametrize("two_pass", ["down_proj", "all", "none"])
@pytest.mark.parametrize("reduce_range", [True, False])
def test_int8_layers_track_float(tiny_ckpt, tiny_lite, two_pass, reduce_range):
    from voxcpm.lite import load_voxcpm2_lite

    f32 = load_voxcpm2_lite(str(tiny_ckpt), precision="float32", verbose=False)
    lite = load_voxcpm2_lite(str(tiny_lite), two_pass=two_pass, reduce_range=reduce_range, verbose=False)
    torch.manual_seed(0)
    for ref_mod, q_mod in zip(_block_linears(f32), _block_linears(lite)):
        x = torch.randn(2, 5, ref_mod.in_features)
        ref = ref_mod(x)
        err = (q_mod(x) - ref).pow(2).mean() / ref.pow(2).mean()
        assert err < 2e-3, (type(q_mod).__name__, float(err))


def test_two_pass_removes_activation_error():
    from voxcpm.lite.quant import Int8Linear, quantize_weight_per_channel

    torch.manual_seed(0)
    w = torch.randn(16, 64)
    q, scale = quantize_weight_per_channel(w)
    w_hat = q.float() * scale[:, None]  # what the int8 weights represent exactly
    x = torch.randn(4, 64) * torch.rand(4, 1) * 10
    ref = x @ w_hat.T

    def rel_err(**kw):
        return float(((Int8Linear(q, scale, **kw)(x) - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())

    one, two = rel_err(two_pass=False), rel_err(two_pass=True)
    assert two < 1e-3 and two < one / 20


@pytest.mark.parametrize("reduce_range", [True, False])
def test_residual_reproduces_kernel_input_quantization(reduce_range):
    from voxcpm.lite.quant import Int8Linear, quantize_weight_per_channel

    for seed in range(40):
        torch.manual_seed(seed)
        rows = [1, 2, 5, 22][seed % 4]
        w = torch.randn(24, 96)
        q, scale = quantize_weight_per_channel(w)
        w_hat = q.float() * scale[:, None]
        x = torch.randn(rows, 96) * (torch.rand(rows, 1) * 10)
        if seed % 3 == 0:
            x = x / x.abs().amax(-1, keepdim=True)  # symmetric [-1, 1] range: zero-point tie
        lin = Int8Linear(q, scale, reduce_range=reduce_range)
        x_q = x - lin._input_quantization_residual(x)  # what the kernel actually multiplies
        assert torch.allclose(lin._kernel(x), x_q @ w_hat.T, rtol=1e-4, atol=1e-4)


def test_outlier_split_is_exact_for_float_channels():
    from voxcpm.lite.quant import Int8Linear, quantize_weight_per_channel, select_outlier_channels

    torch.manual_seed(0)
    w = torch.randn(8, 32)
    chan_absmax = torch.ones(32)
    chan_absmax[[2, 9]] = 100.0
    idx = select_outlier_channels(chan_absmax, ratio=6.0, max_channels=4)
    assert idx.tolist() == [2, 9]
    q, scale = quantize_weight_per_channel(w, zero_cols=idx)
    assert int(q[:, idx.long()].abs().sum()) == 0
    lin = Int8Linear(q, scale, None, float_idx=idx, float_cols=w[:, idx.long()], two_pass=True)
    x = torch.zeros(3, 32)
    x[:, 2], x[:, 9] = 123.0, -77.0  # only outlier channels are active -> pure float path
    assert torch.allclose(lin(x), x @ w.T, rtol=1e-5, atol=1e-4)


def test_keep_float_stack(tiny_ckpt, tmp_path):
    from voxcpm.lite import Int8LinearSlice, load_voxcpm2_lite
    from voxcpm.lite.convert import convert

    dst = tmp_path / "lite_keep_dit"
    convert(str(tiny_ckpt), str(dst), keep_float=["feat_decoder"], calibration_precision="float32", verbose=False)
    lite = load_voxcpm2_lite(str(dst), verbose=False)
    assert isinstance(lite.feat_decoder.estimator.decoder.layers[0].self_attn.q_proj, torch.nn.Linear)
    assert isinstance(lite.base_lm.layers[0].self_attn.q_proj, Int8LinearSlice)
    assert torch.isfinite(_gen(lite)).all()


def test_stock_loader_rejects_lite_directory(tiny_lite):
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    # must fail loudly instead of silently mis-reading int8 weights
    with pytest.raises(FileNotFoundError):
        VoxCPM2Model.from_local(str(tiny_lite), optimize=False, device="cpu")


def test_convert_without_vae_encoder(tiny_ckpt, tmp_path):
    from voxcpm.lite import load_voxcpm2_lite
    from voxcpm.lite.convert import convert

    dst = tmp_path / "lite_noenc"
    convert(str(tiny_ckpt), str(dst), with_vae_encoder=False, calibration_precision="float32", verbose=False)
    lite = load_voxcpm2_lite(str(dst), verbose=False)
    assert torch.isfinite(_gen(lite)).all()
    with pytest.raises(RuntimeError):
        lite.audio_vae.encode(torch.randn(1, 1600), 16000)


def test_two_pass_policy_per_stack(tiny_lite):
    from voxcpm.lite import load_voxcpm2_lite

    lite = load_voxcpm2_lite(str(tiny_lite), two_pass={"feat_decoder": "all", "base_lm": "none"}, verbose=False)
    assert lite.feat_decoder.estimator.decoder.layers[0].self_attn.o_proj.two_pass
    assert not lite.base_lm.layers[0].mlp.down_proj.two_pass
    assert lite.residual_lm.layers[0].mlp.down_proj.two_pass  # untouched default
    with pytest.raises(ValueError):
        load_voxcpm2_lite(str(tiny_lite), two_pass="sometimes", verbose=False)


def test_voxcpm_api_loads_lite_directory(tiny_lite):
    import importlib.util

    import numpy as np

    # tests/test_cli.py replaces ``voxcpm.core`` in sys.modules with a stub for
    # the whole session, so load the real module from its file.
    spec = importlib.util.spec_from_file_location("voxcpm._core_under_test", ROOT / "src" / "voxcpm" / "core.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)

    tts = core.VoxCPM(str(tiny_lite), enable_denoiser=False)
    wav = tts.generate(TEXT, max_len=6, inference_timesteps=4, seed=3, retry_badcase=False)
    assert isinstance(wav, np.ndarray) and wav.ndim == 1 and np.isfinite(wav).all()


def test_rms_norm_fast_path_matches_reference_formula():
    from voxcpm.modules.minicpm4.model import rms_layernorm

    torch.manual_seed(0)
    x, w, eps = torch.randn(3, 5, 32) * 7, torch.randn(32), 1e-5
    ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w
    assert torch.allclose(rms_layernorm(x, w, eps), ref, rtol=1e-5, atol=1e-6)
    # the generic (non-float32) branch is unchanged
    assert torch.allclose(rms_layernorm(x.double(), w.double(), eps), ref.double(), rtol=1e-5, atol=1e-6)


class _LinearField(torch.nn.Module):
    """dx/dt = x: the solver's exact answer is known in closed form."""

    def forward(self, x, mu, t, cond, dt):
        return x


@pytest.mark.parametrize("n_steps", [6, 10])
def test_ab2_solver_is_more_accurate_than_euler(n_steps):
    from voxcpm.modules.locdit import CfmConfig, UnifiedCFM

    def solve(solver):
        cfm = UnifiedCFM(in_channels=4, cfm_params=CfmConfig(solver=solver), estimator=_LinearField())
        torch.manual_seed(0)
        return cfm(mu=torch.zeros(1, 8, dtype=torch.float64), n_timesteps=n_steps, patch_size=2,
                   cond=torch.zeros(1, 4, 2, dtype=torch.float64), cfg_value=2.0)

    torch.manual_seed(0)
    z = torch.randn((1, 4, 2), dtype=torch.float64)
    t = torch.linspace(1, 0, n_steps + 1, dtype=torch.float64)
    t1 = (t + torch.cos(torch.pi / 2 * t) - 1 + t)[1]  # velocity is zero on the first step (CFG-Zero*)
    exact = z * torch.exp(-t1)
    err_euler = (solve("euler") - exact).abs().max()
    err_ab2 = (solve("ab2") - exact).abs().max()
    assert err_ab2 < err_euler / 3

    with pytest.raises(ValueError):
        solve("rk45")


def test_raw_safetensors_reader_matches_safetensors(tmp_path):
    from safetensors.torch import load_file, save_file

    from voxcpm.lite.loader import _RawSafetensorsReader

    tensors = {
        "a": torch.randn(3, 5).to(torch.bfloat16),
        "b": torch.randint(-127, 127, (4, 2), dtype=torch.int8),
        "c": torch.randn(7),
        "d": torch.tensor([1, 2, 3], dtype=torch.int32),
    }
    p = str(tmp_path / "t.safetensors")
    save_file(tensors, p)
    ref = load_file(p)
    reader = _RawSafetensorsReader(p)
    assert set(reader.keys()) == set(ref)
    for k, v in ref.items():
        got = reader.get(k)
        assert got.dtype == v.dtype and torch.equal(got, v)
