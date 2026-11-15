"""Int8 building blocks for the lite runtime.

Weights
-------
Stored as int8 with one symmetric scale per output channel
(``w ≈ q * scale[:, None]``, ``q ∈ [-127, 127]``) and executed by PyTorch's
dynamic-quantized linear kernel (FBGEMM): the float input is quantized on the
fly, the matmul runs in integer arithmetic (AVX2 / AVX-512 VNNI) and the
result comes back as float32. Resident weight memory is 1 byte per parameter.

Activations
-----------
Feeding a transformer's hidden states to that kernel as-is is *not* safe: a
few "massive" channels of the residual stream set the min/max of the whole
tensor, everything else collapses onto a handful of quantization levels, and
individual layers drop below 10 dB SNR (VoxCPM2 then stops predicting
end-of-speech reliably). ``Int8Linear`` therefore combines three exact
re-formulations of ``y = W x`` that keep the integer kernel but give it
well-conditioned inputs:

* **outlier-channel split** – a small calibrated set of input channels is
  removed from the integer path and multiplied by float copies of the matching
  weight columns: ``W x = W_int8 x_main + W_float[:, idx] x[idx]``;
* **per-row scaling** – each token is scaled to unit abs-max before the
  kernel and scaled back after (a bias-free linear map is homogeneous);
* **two-pass residual quantization** (optional) – the quantization residual
  of the first pass is fed through the kernel a second time,
  ``W x ≈ K(x) + K(x - Q(x))``, which removes the activation error almost
  entirely (≥ 55 dB) at twice the cost. Used where the input has a broad heavy
  tail that no fixed channel set covers (the MLP ``down_proj``).
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

INT8_QMAX = 127
_F32 = np.float32


def quantize_weight_per_channel(
    weight: torch.Tensor, zero_cols: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 quantization of a 2-D weight.

    ``zero_cols``: input channels handled in float by the caller; they are
    zeroed before quantization so they neither widen the per-row range nor
    contribute to the integer path.

    Returns ``(q, scale)`` with ``q`` int8 of the same shape and ``scale``
    float32 of shape ``(out_features,)``.
    """
    w = weight.detach().to(torch.float32)
    if zero_cols is not None and zero_cols.numel() > 0:
        w = w.index_fill(1, zero_cols.to(torch.long), 0.0)
    absmax = w.abs().amax(dim=1)
    scale = (absmax / INT8_QMAX).clamp_min(1e-12)
    q = torch.round(w / scale[:, None]).clamp_(-INT8_QMAX, INT8_QMAX).to(torch.int8)
    return q, scale


def select_outlier_channels(chan_absmax: torch.Tensor, ratio: float = 6.0, max_channels: int = 64) -> torch.Tensor:
    """Channels whose observed abs-max exceeds ``ratio`` x the median channel."""
    med = chan_absmax.median().clamp_min(1e-12)
    idx = torch.nonzero(chan_absmax > ratio * med).flatten()
    if idx.numel() > max_channels:
        idx = torch.topk(chan_absmax, max_channels).indices
    return torch.sort(idx).values.to(torch.int32)


def cpu_has_vnni() -> bool:
    """True when the CPU has a VNNI flavour, i.e. u8*s8 dot products cannot
    saturate and the full 8-bit activation range is safe."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    return bool({"avx512_vnni", "avx_vnni", "avxvnni"} & flags)
    except OSError:
        pass
    return False


class Int8Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` holding int8 weights.

    Args:
        qweight: int8 ``(out, in)``; columns listed in ``float_idx`` must be 0.
        scale: float ``(out,)`` per-output-channel scale.
        bias: optional float ``(out,)``.
        float_idx: input channels computed in float (outlier-channel split).
        float_cols: float ``(out, len(float_idx))`` weight columns for them.
        reduce_range: quantize activations to 7 bits. Required on CPUs without
            VNNI, where the kernel's 16-bit intermediate may saturate.
        two_pass: add the residual-quantization pass (see module docstring).
        row_scale: scale every row to unit abs-max around the kernel. Worth
            it when rows differ in magnitude (un-normalised inputs); ignored
            with ``two_pass``, which already removes the activation error.

    The forward pass is written to use as few tensor operations as possible:
    the per-step inputs are tiny (1-22 rows), so the cost around the integer
    matmul is dominated by per-operation overhead, not by arithmetic.
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        float_idx: Optional[torch.Tensor] = None,
        float_cols: Optional[torch.Tensor] = None,
        reduce_range: bool = True,
        two_pass: bool = False,
        row_scale: bool = True,
    ):
        super().__init__()
        assert qweight.dtype == torch.int8 and qweight.dim() == 2
        self.out_features, self.in_features = qweight.shape
        self.reduce_range = bool(reduce_range)
        self.two_pass = bool(two_pass)
        self.row_scale = bool(row_scale)

        with warnings.catch_warnings():
            # PyTorch flags its eager quantized tensors as deprecated; the
            # kernels are still shipped and are what makes int8 fast on CPU.
            warnings.simplefilter("ignore")
            qw = torch._make_per_channel_quantized_tensor(
                qweight,
                scale.to(torch.float64),
                torch.zeros(self.out_features, dtype=torch.int64),
                0,
            )
            # The bias is added outside the kernel: the kernel input may be
            # row-scaled, which a bias inside it would not survive.
            self._packed = torch.ops.quantized.linear_prepack(qw, None)

        has_split = float_idx is not None and float_idx.numel() > 0
        self.register_buffer("float_idx", float_idx.to(torch.long) if has_split else None, persistent=False)
        # stored transposed, (k, out), for an in-place rank-k update of the output
        self.register_buffer(
            "float_cols_t", float_cols.to(torch.float32).t().contiguous() if has_split else None, persistent=False
        )
        self.register_buffer("bias", bias.detach().to(torch.float32) if bias is not None else None, persistent=False)

    # -- integer kernel ----------------------------------------------------------
    def _kernel(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_dynamic(x, self._packed, reduce_range=self.reduce_range)

    def _input_quantization_residual(self, x: torch.Tensor) -> torch.Tensor:
        """``x - Q(x)`` where ``Q`` is exactly the kernel's own input
        quantization (same scale / zero point / rounding; ``reduce_range``
        only narrows the range used to pick the scale, the kernel still stores
        uint8 and clamps at 255)."""
        q_scale, q_zero = torch._choose_qparams_per_tensor(x, self.reduce_range)
        f_scale = _F32(q_scale)
        r = x * float(_F32(1.0) / f_scale)
        r.add_(q_zero).round_().clamp_(0, 255).sub_(q_zero).mul_(float(f_scale))
        return torch.sub(x, r, out=r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = self.float_idx is not None
        if split:
            x_out = x.index_select(-1, self.float_idx)
            x = x.index_fill(-1, self.float_idx, 0.0)
        if self.two_pass:
            # K(x) + K(x - Q(x)) ~= W x
            y = self._kernel(x)
            y.add_(self._kernel(self._input_quantization_residual(x)))
        elif self.row_scale:
            row_scale = x.abs().amax(-1, keepdim=True).clamp_min_(1e-20)
            y = self._kernel(x / row_scale).mul_(row_scale)
        else:
            y = self._kernel(x)
        if split:
            y.view(-1, self.out_features).addmm_(x_out.reshape(-1, x_out.shape[-1]), self.float_cols_t)
        if self.bias is not None:
            y.add_(self.bias)
        return y

    def extra_repr(self) -> str:
        k = 0 if self.float_idx is None else self.float_idx.numel()
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, int8, float_channels={k}, "
            f"two_pass={self.two_pass}, row_scale={self.row_scale}, reduce_range={self.reduce_range}"
        )


class _SharedLinear:
    """One fused ``Int8Linear`` serving several projections of the same input.

    ``q_proj`` / ``k_proj`` / ``v_proj`` (and ``gate_proj`` / ``up_proj``) are
    always applied to the same tensor. Running them as a single integer matmul
    quantizes that input once and pays the kernel's fixed per-call cost once,
    which matters because the per-step inputs are tiny (1–22 rows).
    """

    def __init__(self, fused: Int8Linear, n_slices: int):
        self.fused = fused
        self.n_slices = n_slices
        self._x = None
        self._y = None
        self._served = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._x is not x:
            self._x, self._y, self._served = x, self.fused(x), 0
        y = self._y
        self._served += 1
        if self._served >= self.n_slices:  # last consumer: do not keep tensors alive
            self._x = self._y = None
        return y


class Int8LinearSlice(nn.Module):
    """A contiguous slice of the outputs of a fused ``Int8Linear``."""

    def __init__(self, shared: _SharedLinear, start: int, stop: int, owner: bool):
        super().__init__()
        self._shared = shared
        self.start, self.stop = start, stop
        self.in_features = shared.fused.in_features
        self.out_features = stop - start
        if owner:  # register the fused module once so it shows up in .modules()
            self.fused = shared.fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._shared(x)[..., self.start : self.stop]

    def extra_repr(self) -> str:
        return f"out[{self.start}:{self.stop}] of fused int8 linear"


def fuse_int8_linears(parts: Sequence[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]], **kwargs) -> List[Int8LinearSlice]:
    """Fuse ``[(qweight, scale, float_cols), ...]`` sharing one input (and one
    ``float_idx``) into a single ``Int8Linear``; return one slice per part."""
    qweight = torch.cat([p[0] for p in parts], dim=0)
    scale = torch.cat([p[1] for p in parts], dim=0)
    float_cols = None
    if parts[0][2] is not None:
        float_cols = torch.cat([p[2].to(torch.float32) for p in parts], dim=0)
    fused = Int8Linear(qweight, scale, None, float_cols=float_cols, **kwargs)
    shared = _SharedLinear(fused, len(parts))
    slices, start = [], 0
    for i, p in enumerate(parts):
        stop = start + p[0].shape[0]
        slices.append(Int8LinearSlice(shared, start, stop, owner=(i == 0)))
        start = stop
    return slices


class Int8Embedding(nn.Module):
    """Embedding table stored as int8 rows with one float scale per row."""

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        assert qweight.dtype == torch.int8 and qweight.dim() == 2
        self.num_embeddings, self.embedding_dim = qweight.shape
        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("scale", scale.to(torch.float32), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.qweight[ids].to(torch.float32) * self.scale[ids].unsqueeze(-1)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, int8"
