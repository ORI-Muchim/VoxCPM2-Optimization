# Design

Where VoxCPM2 spends time and memory on a CPU, what each optimization changes, and what was tried and dropped. Numbers come from the machine described in [benchmarks.md](benchmarks.md).

## Where the cost is

VoxCPM2 has 2.29 billion transformer parameters plus a 94 million parameter AudioVAE.

| Module                         | Parameters | Role                                              |
| ------------------------------ | ---------- | ------------------------------------------------- |
| Base LM (28 layers)            | 1,321 M    | Text-semantic language model                      |
| Token embedding                | 150 M      |                                                   |
| Residual LM (8 layers)         | 378 M      | Acoustic language model                           |
| LocDiT (12 layers)             | 212 M      | Diffusion decoder for one latent patch            |
| LocEnc (12 layers)             | 208 M      | Encodes the previous patch for the LMs            |
| Projections, stop head         | 21 M       |                                                   |
| AudioVAE encoder / decoder     | 48 M / 46 M | Encoder is only used for voice cloning            |

Each generated patch is 160 ms of audio and costs one base LM step, one residual LM step, one LocEnc pass and one LocDiT solve. With the default sampler the solve is 9 estimator evaluations, each over 22 rows (2 guidance branches × 11 tokens).

With the upstream loader on four cores the time splits like this:

| Module                  | Share of time |
| ----------------------- | ------------- |
| LocDiT solve            | 55 %          |
| Base LM step            | 24 %          |
| Residual LM step        | 7 %           |
| AudioVAE decode         | 6 %           |
| LocEnc                  | 6 %           |

Two things follow from how small the per-step inputs are.

**The work is bound by memory bandwidth, not arithmetic.** A base LM step pushes one token through 1.3 billion weights, and a LocDiT evaluation pushes 22 rows through 212 million weights. Every weight is read from RAM on every pass, so time is roughly weight bytes times number of passes. That is why smaller weights and fewer LocDiT evaluations are the levers that matter.

**Operation count matters, not only operation size.** After a large matmul evicts the caches, each small tensor operation that follows costs 20–50 µs regardless of how little data it touches. A transformer layer has dozens of them, and in the int8 LocDiT they add up to about 40 % of the time.

On memory, the upstream loader peaks at 11.1 GB for a 4.6 GB checkpoint because it first builds the whole model with randomly initialised float32 weights and only then casts and overwrites them.

## Lossless changes

These do not change the model's arithmetic. `tests/test_lite.py` checks that the lite loader in float32 reproduces the upstream loader's audio, streaming included, on a tiny random checkpoint.

**Loader** (`src/voxcpm/lite/loader.py`). Parameter initialisation is skipped while the module skeleton is built, so the freshly allocated weights are never written and never become resident. Tensors are then read one at a time into their final dtype. For bf16 the weights are memory-mapped straight from the safetensors file, which makes loading nearly instant and lets the OS share and evict the pages. For conversions a small non-mmap reader keeps file pages out of the process's resident set. Peak memory drops from 11.1 GB to 5.7 GB and load time from 14–21 s to 1.1 s.

**KV-cache slicing** (`src/voxcpm/modules/minicpm4/model.py`). Upstream attends over all 8192 cache slots on every step and masks the unused ones, because `torch.compile` on CUDA wants static shapes. Grouped-query attention then expands those 8192 keys and values to 16 heads in every layer. On an uncompiled path that is wasted work, so the step now attends to the filled prefix only. The base LM step goes from 138 ms to 85 ms. This is also enabled on the upstream loader whenever `torch.compile` is not in use.

**AudioVAE.** Upstream uses the legacy `weight_norm` hook, which recomputes every convolution weight from its `g` and `v` parts on each forward call, and in streaming mode that means on every chunk. The lite loader folds the weights once at load time. The encoder, 48 million parameters used only to encode cloning audio, is loaded on first use.

**RMSNorm.** In float32 on CPU the six-operation formula is replaced by the single fused `rms_norm` operation, which computes the same thing.

**RoPE tables.** Three modules each kept tables for 32768 positions. The LM can never exceed its KV-cache capacity and the local modules see at most 11 tokens, so the tables are cut down. This saves about 90 MB.

**Lazy imports.** `librosa`, `torchaudio` and `huggingface_hub` are imported where they are used. Text-to-speech then needs only `torch`, `transformers`, `einops`, `pydantic`, `safetensors`, `tqdm` and `numpy`.

## Int8

### Why stock dynamic quantization fails

PyTorch's dynamic-quantized linear kernel quantizes its float input on the fly, using one minimum and maximum for the whole tensor. Applied to every Linear layer of VoxCPM2, that produced a model that mostly could not stop: 10 of the first 19 evaluation utterances ran on until the length limit, turning 4-second sentences into 17 to 30 seconds of audio.

The cause is a few very large channels in the residual stream. Measured on real activations, the inputs of the attention projections and of `gate_proj` / `up_proj` have a peak-to-RMS ratio of up to 45, and in the first base LM layer the largest channels have a range more than 400 times the median channel. Those channels set the quantization range, and everything else collapses onto a handful of levels. The worst layers kept 7–10 dB of signal-to-noise ratio at their outputs.

### The int8 layer

`Int8Linear` (`src/voxcpm/lite/quant.py`) keeps the integer kernel but rewrites `y = W x` into forms that give the kernel well-conditioned inputs. Each is exact in real arithmetic.

**Outlier-channel split.** A small set of input channels, found by calibration, is removed from the integer path and multiplied by float copies of the matching weight columns:

```
W x = W_int8 · x_main + W_float[:, idx] · x[idx]
```

**Per-row scaling.** A bias-free linear map is homogeneous, so each row can be scaled to unit peak before the kernel and scaled back after. This is used where rows differ in magnitude, which is the inputs of `o_proj` and `down_proj`.

**Two-pass residual quantization.** The kernel's own input quantization `Q` is reproduced exactly, and the residual is fed through the kernel a second time:

```
W x ≈ K(x) + K(x − Q(x))
```

The first-pass error cancels and only the much smaller error of quantizing the residual remains. It needs no calibration and doubles the cost of the layer.

Simulated output SNR of the activation quantization on real activations, minimum over the layers of each kind:

| Scheme                                    | q / k / v, gate / up | `o_proj`  | `down_proj` |
| ----------------------------------------- | -------------------- | --------- | ----------- |
| Stock dynamic quantization, 7 bit         | 7–29 dB              | 24–39 dB  | 10–19 dB    |
| Per-row scaling only, 8 bit               | 12–31 dB             | 35–41 dB  | 23–27 dB    |
| Outlier split ¹ + per-row scaling, 8 bit  | 30–46 dB             | 37–45 dB  | 30–31 dB    |
| Two-pass, 8 bit                           | 59–84 dB             | 84–91 dB  | 70–75 dB    |
| Two-pass, 7 bit                           | 46–72 dB             | 72–79 dB  | 58–63 dB    |

¹ Simulated with a fixed 16 channels per layer; the converter picks the number per layer.

The split fixes the projections that read RMSNorm outputs, where the outliers sit in fixed channels. The MLP `down_proj` input has a broad heavy tail that no fixed set of channels covers, which is what the second pass is for.

### Weights

Weights are quantized symmetrically to int8 with one scale per output channel, after zeroing the columns that are handled in float. The weight quantization SNR is about 41 dB for every layer (35–42 dB), which is what uniform 8-bit quantization of a roughly Gaussian row gives. The token embedding is stored as int8 rows with a per-row scale.

Only the attention and MLP projections inside the transformer blocks are quantized. The remaining 26 million parameters (normalization weights, input and output projections, time embeddings, the stop head and the glue between modules) stay in float.

### Fused projections

`q_proj`, `k_proj` and `v_proj` always receive the same tensor, as do `gate_proj` and `up_proj`. Their weights are concatenated into one integer matmul whose output is sliced. The input is quantized once and the kernel's fixed per-call cost is paid once, so a layer makes 4 kernel calls instead of 7.

### Calibration

The converter generates 12 patches for each of three short texts (English, Korean, Chinese) with the float model and records the peak magnitude of each input channel. A channel is handled in float when its peak exceeds 6 times the median channel, up to 64 per layer. For the released weights that selects 3,526 channels across 420 weight matrices. Almost two thirds of them are in the `down_proj` layers (37 each on average); the attention projections need about 5 each, `gate_proj` and `up_proj` about 1, and most `o_proj` layers none. The first base LM layer needs the most.

### Default two-pass policy

By default the second pass runs on `down_proj` in both LMs and the LocEnc, and nowhere in the LocDiT. Measured against float32 with the same inputs ([benchmarks.md](benchmarks.md#fidelity-against-float32)), two passes everywhere bring the LMs and the LocEnc to 41–44 dB, the same level as the upstream bf16 runtime. In the LocDiT the second pass is worth only a few dB. A single LocDiT evaluation is as close to float as the bf16 runtime is, but weight rounding is the same error on every evaluation, so it adds up over the solver steps where bf16's rounding noise averages out. The LocDiT is also the most expensive module.

### Engine and memory

PyTorch offers several quantized engines. On the real model:

| Engine          | Load time | Private memory | LocDiT evaluation | Base LM step |
| --------------- | --------- | -------------- | ----------------- | ------------ |
| `x86` (default) | 20.3 s    | 3.3 GB         | 33.5 ms           | 83.7 ms      |
| `fbgemm`        | 21.8 s    | 3.3 GB         | 32.0 ms           | 76.1 ms      |
| `onednn`        | 10.9 s    | 5.4 GB         | 37.3 ms           | 105.0 ms     |

oneDNN keeps a second copy of every weight and is slower here, so the default engine is used. Load times in this table predate parallel packing: packing the weights into the kernel's layout is single-threaded, so the loader now packs a few groups at a time in a thread pool, which brings loading to 11 s.

glibc keeps freed heap pages, and loading allocates and frees many multi-megabyte temporaries. Without returning them to the OS (`malloc_trim`) the int8 model's resident size was 4.2 GB instead of 3.3 GB.

## Sampler

The LocDiT integrates a velocity field with Euler steps. Its accuracy can be measured directly by comparing the result against a 96-step solve from the same noise:

| Sampler      | Estimator evaluations | SNR vs converged |
| ------------ | --------------------- | ---------------- |
| Euler, 10    | 9                     | 18.0 dB          |
| Euler, 8     | 7                     | 15.6 dB          |
| Euler, 6     | 5                     | 13.8 dB          |
| Euler, 4     | 3                     | 8.4 dB           |
| AB2, 10      | 9                     | 23.9 dB          |
| **AB2, 8**   | **7**                 | **20.3 dB**      |
| AB2, 6       | 5                     | 13.7 dB          |

The `ab2` solver (`src/voxcpm/modules/locdit/unified_cfm.py`) is second-order Adams-Bashforth with variable step size. It combines the current and previous velocity, so it needs no extra evaluations. Eight AB2 steps are closer to the converged solution than the default ten Euler steps, with two fewer evaluations. At six steps the grid is too coarse for the second-order term to help.

This also puts the int8 LocDiT in perspective. Its output deviates from float32 by 19–25 dB depending on the two-pass mode, which is comparable to or smaller than the 18 dB discretization error the default sampler already has.

## Tried and dropped

**Batched two-sub-step sampler.** Predict the state at the next sub-step with the latest velocity and evaluate both sub-steps in one batched call. Six batched evaluations matched the accuracy of the default nine (18.0 dB), but a batch of 4 costs 1.3 times a batch of 2 in bf16 and 1.6–1.7 times in int8, so there was no net gain.

**oneDNN quantized engine.** See the table above.

**Plain dynamic quantization.** See [Why stock dynamic quantization fails](#why-stock-dynamic-quantization-fails).

**Comparison with the community GGUF runtimes.** Not done. Building VoxCPM.cpp was not possible in the environment used for this work.

## What would go further

All of these need a GPU and training data.

1. **Few-step distillation of the LocDiT.** The code already has a MeanFlow training path (`dit_mean_mode`). Distilling to one or two steps would cut the largest cost by 4 to 9 times and is the most direct route to real time.
2. **A smaller backbone.** The base LM is 64 % of the memory. Distil it into fewer layers.
3. **4-bit weights.** PyTorch has no CPU kernel for them, so this means ONNX Runtime or a GGUF runtime. Since the model is bandwidth bound, it would help speed as well as size.

## Code map

| File                                         | Contents                                                                                   |
| -------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `src/voxcpm/lite/loader.py`                  | `load_voxcpm2_lite`, checkpoint readers, weight-norm folding, int8 module installation      |
| `src/voxcpm/lite/quant.py`                   | `Int8Linear`, fused projections, `Int8Embedding`, weight quantization, channel selection    |
| `src/voxcpm/lite/convert.py`                 | Calibration and the `python -m voxcpm.lite.convert` command                                 |
| `src/voxcpm/modules/minicpm4/model.py`       | `slice_kv_cache`, RMSNorm fast path                                                         |
| `src/voxcpm/modules/locdit/unified_cfm.py`   | `ab2` solver                                                                                |
| `tests/test_lite.py`                         | Equivalence with the upstream loader, int8 layer accuracy, converter round trip, solver order |
