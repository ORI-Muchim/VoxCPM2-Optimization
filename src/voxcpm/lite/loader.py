"""Low-footprint loader for VoxCPM2 (CPU-oriented).

Compared with ``VoxCPM2Model.from_local`` this loader

* never materialises randomly-initialised fp32 weights (the stock path peaks
  at ~2.4x the checkpoint size while loading),
* streams tensors one by one straight into their final dtype,
* folds weight-norm out of the AudioVAE (the stock VAE recomputes every conv
  weight on each forward call, i.e. on every streamed chunk),
* loads the AudioVAE encoder lazily (it is only needed for voice cloning),
* only touches the part of the KV cache that is actually used,
* executes the transformer blocks with int8 weights when given a checkpoint
  converted by ``python -m voxcpm.lite.convert`` (see ``quant.py``).

It reads both the original checkpoint layout (float precisions) and the
compact ``*.lite`` layout (int8).
"""

from __future__ import annotations

import contextlib
import json
import os
import struct
import sys
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from .quant import Int8Embedding, Int8Linear, cpu_has_vnni, fuse_int8_linears

LITE_FORMAT_VERSION = 1
LITE_MODEL_FILE = "model.lite.safetensors"
LITE_VAE_FILE = "audiovae.lite.safetensors"
SCALE_SUFFIX = ".__scale"
FIDX_SUFFIX = ".__fidx"
FCOLS_SUFFIX = ".__fcols"

# Transformer stacks whose block Linears are quantized to int8.
GROUPS = ("base_lm", "residual_lm", "feat_encoder", "feat_decoder")
# Projections applied to the same input tensor (fused into one integer matmul)
# and the remaining block Linears.
ATTN_LINEARS = (("q_proj", "k_proj", "v_proj"), ("o_proj",))
MLP_LINEARS = (("gate_proj", "up_proj"), ("down_proj",))
TWO_PASS_MODES = ("down_proj", "all", "none")
# Default: residual pass on the MLP ``down_proj`` of the LMs and the local
# encoder. Not in the LocDiT: it is the most expensive module, and most of its
# deviation from float comes from the (systematic) weight rounding accumulated
# over the solver steps, so the extra pass buys only a few dB there
# (pass ``two_pass={"feat_decoder": "down_proj"}`` to enable it anyway).
DEFAULT_TWO_PASS = {"base_lm": "down_proj", "residual_lm": "down_proj", "feat_encoder": "down_proj", "feat_decoder": "none"}


def resolve_two_pass(two_pass) -> dict:
    """``None`` -> default policy; a mode string -> that mode for every stack;
    a dict -> per-stack overrides of the default policy."""
    policy = dict(DEFAULT_TWO_PASS)
    if two_pass is None or two_pass == "auto":
        return policy
    if isinstance(two_pass, str):
        two_pass = {g: two_pass for g in GROUPS}
    for g, mode in two_pass.items():
        if g not in GROUPS or mode not in TWO_PASS_MODES:
            raise ValueError(f"two_pass: unknown stack/mode {g}={mode} (stacks {GROUPS}, modes {TWO_PASS_MODES})")
        policy[g] = mode
    return policy
_PROJ_MODULES = (
    "fsq_layer",
    "enc_to_lm_proj",
    "lm_to_dit_proj",
    "res_to_dit_proj",
    "fusion_concat_proj",
    "stop_proj",
    "stop_head",
)
_FLOAT_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def group_of(name: str) -> str:
    top = name.split(".", 1)[0]
    if top in ("base_lm", "residual_lm", "feat_encoder", "feat_decoder"):
        return top
    if top in _PROJ_MODULES:
        return "proj"
    if top == "audio_vae":
        return "audio_vae"
    return "other"


# --------------------------------------------------------------------------- #
# Construction helpers
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _no_param_init():
    """Skip random initialisation while building the module skeleton.

    Freshly allocated weights are then never written, so they never become
    resident memory before being replaced by the checkpoint tensors.
    """
    classes = (nn.Linear, nn.Embedding, nn.Conv1d, nn.ConvTranspose1d)
    saved = {c: c.reset_parameters for c in classes}
    for c in classes:
        c.reset_parameters = lambda self: None
    try:
        yield
    finally:
        for c, fn in saved.items():
            c.reset_parameters = fn


@contextlib.contextmanager
def _plain_vae_convs():
    """Build the AudioVAE with plain convolutions (weight-norm folded away)."""
    from ..modules.audiovae import audio_vae_v2 as m

    names = ("WNConv1d", "WNConvTranspose1d", "WNCausalConv1d", "WNCausalTransposeConv1d")
    saved = {n: getattr(m, n) for n in names}
    m.WNConv1d = lambda *a, **k: nn.Conv1d(*a, **k)
    m.WNConvTranspose1d = lambda *a, **k: nn.ConvTranspose1d(*a, **k)
    m.WNCausalConv1d = lambda *a, **k: m.CausalConv1d(*a, **k)
    m.WNCausalTransposeConv1d = lambda *a, **k: m.CausalTransposeConv1d(*a, **k)
    try:
        yield
    finally:
        for n, fn in saved.items():
            setattr(m, n, fn)


def fold_weight_norm(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """``weight_g`` / ``weight_v`` pairs -> plain ``weight`` (same maths as the hook)."""
    out = {}
    for k, v in state.items():
        if k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            out[base + ".weight"] = torch._weight_norm(
                state[base + ".weight_v"].to(torch.float32), v.to(torch.float32), 0
            )
        elif k.endswith(".weight_v"):
            continue
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Checkpoint readers (one tensor at a time)
# --------------------------------------------------------------------------- #
class _SafetensorsReader:
    """mmap-backed reader (tensors are lazily paged views of the file)."""

    def __init__(self, path: str):
        from safetensors import safe_open

        self.path = path
        self._f = safe_open(path, framework="pt", device="cpu")
        self._keys = set(self._f.keys())

    def keys(self) -> Iterable[str]:
        return self._keys

    def __contains__(self, k: str) -> bool:
        return k in self._keys

    def get(self, k: str) -> torch.Tensor:
        return self._f.get_tensor(k)


class _RawSafetensorsReader:
    """Dependency-free safetensors reader using plain file reads.

    Unlike the mmap reader, the file pages never count towards this process'
    resident memory, which keeps the peak low when tensors are converted
    (e.g. bf16 -> int8) or repacked right after being read.
    """

    _DTYPES = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        (n,) = struct.unpack("<Q", self._f.read(8))
        self._meta = json.loads(self._f.read(n))
        self.metadata = self._meta.pop("__metadata__", None)
        self._base = 8 + n

    def keys(self) -> Iterable[str]:
        return self._meta.keys()

    def __contains__(self, k: str) -> bool:
        return k in self._meta

    def get(self, k: str) -> torch.Tensor:
        info = self._meta[k]
        start, end = info["data_offsets"]
        raw = torch.empty(end - start, dtype=torch.uint8)
        if end > start:
            self._f.seek(self._base + start)
            got = self._f.readinto(memoryview(raw.numpy()))
            if got != end - start:
                raise IOError(f"Truncated safetensors file: {self.path}")
        return raw.view(self._DTYPES[info["dtype"]]).reshape(info["shape"])


class _TorchReader:
    """Reads a ``torch.save`` zip checkpoint lazily (mmap)."""

    def __init__(self, path: str):
        self.path = path
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except Exception:  # legacy (non-zip) checkpoints cannot be mmapped
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self._sd = ckpt.get("state_dict", ckpt)

    def keys(self) -> Iterable[str]:
        return self._sd.keys()

    def __contains__(self, k: str) -> bool:
        return k in self._sd

    def get(self, k: str) -> torch.Tensor:
        return self._sd[k]


def _open(path: str, mmap: bool = False):
    if path.endswith(".safetensors"):
        return _SafetensorsReader(path) if mmap else _RawSafetensorsReader(path)
    return _TorchReader(path)


def find_checkpoint_files(path: str):
    """Return ``(model_file, vae_file, is_lite)`` for a checkpoint directory."""
    lite_model = os.path.join(path, LITE_MODEL_FILE)
    if os.path.exists(lite_model):
        return lite_model, os.path.join(path, LITE_VAE_FILE), True
    model = next(
        (p for p in (os.path.join(path, n) for n in ("model.safetensors", "pytorch_model.bin")) if os.path.exists(p)),
        None,
    )
    vae = next(
        (p for p in (os.path.join(path, n) for n in ("audiovae.safetensors", "audiovae.pth")) if os.path.exists(p)),
        None,
    )
    if model is None or vae is None:
        raise FileNotFoundError(f"No VoxCPM2 checkpoint found in {path}")
    return model, vae, False


# --------------------------------------------------------------------------- #
# Assignment helpers
# --------------------------------------------------------------------------- #
def _set_tensor(root: nn.Module, name: str, tensor: torch.Tensor) -> None:
    mod_name, _, leaf = name.rpartition(".")
    mod = root.get_submodule(mod_name) if mod_name else root
    if leaf in mod._parameters:
        expected = mod._parameters[leaf]
        if expected is not None and tuple(expected.shape) != tuple(tensor.shape):
            raise ValueError(f"shape mismatch for {name}: {tuple(tensor.shape)} vs {tuple(expected.shape)}")
        mod._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    elif leaf in mod._buffers:
        mod._buffers[leaf] = tensor
    else:
        raise KeyError(name)


def _replace_module(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, new)


class _VAEWeights:
    """Folded AudioVAE weights, readable per sub-module (decoder / encoder)."""

    def __init__(self, vae_file: str, is_lite: bool):
        self.reader = _open(vae_file)
        self.is_lite = is_lite

    def load_into(self, vae: nn.Module, prefix: str) -> None:
        keys = [k for k in self.reader.keys() if k.startswith(prefix + ".")]
        state = {k: self.reader.get(k) for k in keys}
        if not self.is_lite:
            state = fold_weight_norm(state)
        expected = {n for n, _ in vae.get_submodule(prefix).named_parameters(prefix=prefix)}
        expected |= {n for n, _ in vae.get_submodule(prefix).named_buffers(prefix=prefix)}
        missing = expected - set(state)
        if missing:
            raise KeyError(f"AudioVAE checkpoint is missing {sorted(missing)[:5]} ...")
        for k, v in state.items():
            if k not in expected:
                continue
            # Always copy: tensors may be views into an mmapped checkpoint file.
            v = v.to(torch.float32, copy=True) if v.is_floating_point() else v.clone()
            _set_tensor(vae, k, v)


class _LazyKVCacheMixin:
    """``fill_caches`` without clearing the whole (mostly unused) cache."""

    @staticmethod
    def install(kv_cache) -> None:
        def fill_caches(kv_caches, _c=kv_cache):
            _c.current_length = kv_caches[0][0].size(2)
            for i in range(_c.num_layers):
                _c.kv_cache[0, i, :, :, : _c.current_length, :] = kv_caches[i][0]
                _c.kv_cache[1, i, :, :, : _c.current_length, :] = kv_caches[i][1]

        kv_cache.fill_caches = fill_caches


def _trim_rope_tables(model: nn.Module, lm_positions: int) -> None:
    """Keep only the RoPE rows that can ever be indexed.

    The stock tables cover ``max_position_embeddings`` (32768) positions for
    each of the three RoPE users (~100 MB in total), but the LM can never go
    past its KV-cache capacity and the local encoder / DiT only ever see a
    handful of tokens per patch.
    """
    from ..modules.minicpm4.model import MiniCPMLongRoPE

    local_positions = 4 * model.patch_size + 16
    for name, mod in model.named_modules():
        if not isinstance(mod, MiniCPMLongRoPE):
            continue
        keep = lm_positions if name.startswith("base_lm") else local_positions
        keep = min(keep, mod.cos_cached.size(0))
        mod.cos_cached = mod.cos_cached[:keep].clone()
        mod.sin_cached = mod.sin_cached[:keep].clone()
        mod.max_seq_len_cached = keep


def _release_free_memory() -> None:
    """Hand freed heap pages back to the OS (glibc keeps them otherwise).

    Loading allocates and frees many multi-megabyte temporaries; without this
    the int8 model's resident size is roughly 1.4x what it actually uses.
    """
    import gc

    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def block_linear_groups(model: nn.Module):
    """Yield ``(block_name, (leaf, ...))`` for every group of transformer-block
    Linears (attention and MLP projections) outside the AudioVAE."""
    from ..modules.minicpm4.model import MiniCPMAttention, MiniCPMMLP

    for name, mod in model.named_modules():
        if group_of(name) not in GROUPS:
            continue
        if isinstance(mod, MiniCPMAttention):
            for grp in ATTN_LINEARS:
                yield name, grp
        elif isinstance(mod, MiniCPMMLP):
            for grp in MLP_LINEARS:
                yield name, grp


def _install_int8_blocks(model: nn.Module, reader, reduce_range: bool, two_pass: dict) -> set:
    """Replace quantized block Linears by (fused) ``Int8Linear`` modules.

    Returns the set of checkpoint tensor names consumed.
    """
    from concurrent.futures import ThreadPoolExecutor

    def build(task):
        leaves, parts, idxs, kwargs = task
        if len(leaves) > 1 and all(torch.equal(idxs[0], i) for i in idxs[1:]):
            return fuse_int8_linears(parts, float_idx=idxs[0], **kwargs)
        return [
            Int8Linear(q, scale, None, float_idx=idx, float_cols=cols, **kwargs)
            for (q, scale, cols), idx in zip(parts, idxs)
        ]

    consumed = set()
    pending = []  # (block, task)

    def flush(pool):
        # Packing the int8 weights is single-threaded and dominates load time;
        # do a few groups at a time in parallel (bounded, to keep the peak low).
        for (block, task), new_modules in zip(pending, pool.map(build, [t for _, t in pending])):
            for leaf, new in zip(task[0], new_modules):
                _replace_module(model, f"{block}.{leaf}", new)
        pending.clear()

    workers = max(1, min(8, torch.get_num_threads()))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for block, leaves in block_linear_groups(model):
            names = [f"{block}.{leaf}.weight" for leaf in leaves]
            if not all((n + SCALE_SUFFIX) in reader for n in names):
                continue  # this group was kept in float at conversion time
            parts, idxs = [], []
            for leaf, n in zip(leaves, names):
                if model.get_submodule(f"{block}.{leaf}").bias is not None:
                    raise NotImplementedError("block Linears with bias are not supported by the int8 path")
                q, scale = reader.get(n), reader.get(n + SCALE_SUFFIX)
                has_split = (n + FIDX_SUFFIX) in reader
                idxs.append(reader.get(n + FIDX_SUFFIX).to(torch.long) if has_split else torch.zeros(0, dtype=torch.long))
                parts.append((q, scale, reader.get(n + FCOLS_SUFFIX) if has_split else None))
            mode = two_pass[group_of(block)]
            kwargs = dict(
                reduce_range=reduce_range,
                two_pass=(mode == "all") or (mode == "down_proj" and leaves == ("down_proj",)),
                # q/k/v and gate/up read RMSNorm outputs (rows already on one
                # scale); o_proj / down_proj inputs are not normalised.
                row_scale=len(leaves) == 1,
            )
            pending.append((block, (leaves, parts, idxs, kwargs)))
            consumed.update(names)
            if len(pending) >= 2 * workers:
                flush(pool)
        flush(pool)
    return consumed


def load_voxcpm2_lite(
    path: str,
    precision: Optional[str] = None,
    load_vae_encoder: str | bool = "lazy",
    max_length: Optional[int] = None,
    reduce_range: Optional[bool] = None,
    two_pass=None,
    solver: Optional[str] = None,
    mmap: Optional[bool] = None,
    verbose: bool = True,
):
    """Load VoxCPM2 for CPU inference with a small memory footprint.

    Args:
        path: checkpoint directory. Either the original layout (float
            precisions) or a ``*.lite`` directory written by
            ``python -m voxcpm.lite.convert`` (int8).
        precision: ``"bfloat16"`` or ``"float32"`` for an original checkpoint
            (default ``"bfloat16"``). Ignored for a lite checkpoint, which is
            always int8 weights + float32 activations.
        load_vae_encoder: ``"lazy"`` (load on first voice-cloning call),
            ``True`` (eager) or ``False`` (never; cloning raises).
        max_length: KV-cache capacity in LM tokens (default: config value).
        reduce_range: int8 only. Quantize activations to 7 bits; needed on
            CPUs without VNNI. Default: auto-detect.
        two_pass: int8 only. Which layers use two-pass residual quantization
            (see ``quant.py``): ``"down_proj"``, ``"all"`` (most faithful,
            ~2x the matmul cost) or ``"none"`` (fastest), either for every
            stack or as a ``{stack: mode}`` dict. Default: ``down_proj`` in
            the LMs and the local encoder, ``none`` in the LocDiT.
        solver: ODE solver of the LocDiT, ``"euler"`` (checkpoint default) or
            ``"ab2"``. With ``"ab2"`` 8 steps are at least as accurate as 10
            Euler steps (pass ``inference_timesteps=8`` to ``generate``).
        mmap: map float weights straight from the checkpoint file instead of
            reading them into memory (instant load, pages are shared and
            evictable). Default: on when no dtype conversion is needed.
    """
    from transformers import LlamaTokenizerFast

    from ..model import voxcpm2 as v2
    from ..modules.audiovae import AudioVAEV2
    from ..modules.minicpm4.model import MiniCPMAttention

    model_file, vae_file, is_lite = find_checkpoint_files(path)
    with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    lite_cfg = raw_cfg.pop("lite", None) or {}
    config = v2.VoxCPMConfig.model_validate(raw_cfg)

    if is_lite:
        if precision not in (None, "int8"):
            raise ValueError("A lite checkpoint is int8; drop the `precision` argument.")
        precision = "int8"
        if lite_cfg.get("format") != LITE_FORMAT_VERSION:
            raise ValueError(f"Unsupported lite checkpoint format: {lite_cfg.get('format')}")
    else:
        precision = precision or "bfloat16"
        if precision == "int8":
            raise ValueError(
                "int8 needs a converted checkpoint (weights are calibrated once, offline). Run:\n"
                "    python -m voxcpm.lite.convert <VoxCPM2 dir> <output dir>\n"
                "and load the output directory instead."
            )
        if precision not in _FLOAT_DTYPES:
            raise ValueError(f"Unsupported precision: {precision}")
    float_dtype = torch.float32 if is_lite else _FLOAT_DTYPES[precision]
    if reduce_range is None:
        reduce_range = not cpu_has_vnni()
    two_pass = resolve_two_pass(two_pass)

    config.device = "cpu"
    config.dtype = "float32" if is_lite else precision
    if max_length:
        config.max_length = int(max_length)

    tokenizer = LlamaTokenizerFast.from_pretrained(path)

    with _no_param_init(), _plain_vae_convs():
        audio_vae = AudioVAEV2(config=config.audio_vae_config) if config.audio_vae_config else AudioVAEV2()
        _stderr = sys.stderr
        try:
            if not verbose:
                sys.stderr = open(os.devnull, "w")
            model = v2.VoxCPM2Model(config, tokenizer, audio_vae, None, device="cpu")
        finally:
            if not verbose:
                sys.stderr.close()
                sys.stderr = _stderr

    # ---- transformer stacks ------------------------------------------------
    if mmap is None:
        mmap = precision == "bfloat16" and not is_lite
    reader = _open(model_file, mmap=mmap)

    expected = {n for n, _ in model.named_parameters() if not n.startswith("audio_vae.")}
    assigned = _install_int8_blocks(model, reader, reduce_range, two_pass) if is_lite else set()

    embed_weight = "base_lm.embed_tokens.weight"
    for name in sorted(expected - assigned):
        if name == embed_weight and (name + SCALE_SUFFIX) in reader:
            _replace_module(model, "base_lm.embed_tokens", Int8Embedding(reader.get(name), reader.get(name + SCALE_SUFFIX)))
        else:
            _set_tensor(model, name, reader.get(name).to(float_dtype))
        assigned.add(name)

    missing = expected - assigned
    if missing:
        raise KeyError(f"Checkpoint {model_file} is missing {len(missing)} tensors, e.g. {sorted(missing)[:5]}")

    # ---- AudioVAE ------------------------------------------------------------
    vae_weights = _VAEWeights(vae_file, is_lite)
    vae_weights.load_into(model.audio_vae, "decoder")
    has_encoder = any(k.startswith("encoder.") for k in vae_weights.reader.keys())

    def _load_encoder():
        if not has_encoder:
            raise RuntimeError(
                "This lite checkpoint was exported without the AudioVAE encoder; "
                "voice cloning (prompt / reference audio) is unavailable."
            )
        vae_weights.load_into(model.audio_vae, "encoder")

    if load_vae_encoder is True:
        _load_encoder()
    else:
        orig_encode = model.audio_vae.encode
        state = {"loaded": False}

        def lazy_encode(*a, **k):
            if not state["loaded"]:
                if load_vae_encoder is False:
                    raise RuntimeError("AudioVAE encoder was not loaded (load_vae_encoder=False).")
                _load_encoder()
                state["loaded"] = True
            try:
                return orig_encode(*a, **k)
            finally:
                # Encoding a reference clip allocates several hundred MB of
                # short-lived activations; give them back right away.
                _release_free_memory()

        model.audio_vae.encode = lazy_encode

    # ---- runtime tweaks --------------------------------------------------------
    for m in model.modules():
        if isinstance(m, MiniCPMAttention):
            m.slice_kv_cache = True
    for lm in (model.base_lm, model.residual_lm):
        _LazyKVCacheMixin.install(lm.kv_cache)
    _trim_rope_tables(model, lm_positions=config.max_length)
    if solver is not None:
        if solver not in ("euler", "ab2"):
            raise ValueError(f"Unsupported solver: {solver}")
        model.feat_decoder.solver = solver
    del reader
    _release_free_memory()

    model.lite_info = {
        "precision": precision,
        "reduce_range": bool(reduce_range) if is_lite else None,
        "two_pass": two_pass if is_lite else None,
        "source": "lite" if is_lite else "original",
    }
    return model.eval()
