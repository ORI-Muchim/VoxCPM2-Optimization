# VoxCPM2-Optimization

Training-free CPU optimizations for [VoxCPM2](https://github.com/OpenBMB/VoxCPM): a low-memory loader, a calibrated int8 runtime and a cheaper sampler, with the measurements behind them.

**English** | [한국어](README_ko.md)

> **Unofficial.** This repository is OpenBMB/VoxCPM at commit [`300e426`](https://github.com/OpenBMB/VoxCPM/commit/300e4260b509066151d623dc9c0aa922cf82232b) plus the changes listed under [Changes from upstream](#changes-from-upstream). The original README is kept as [README_upstream.md](README_upstream.md) ([中文](README_upstream_zh.md)); everything it documents (voice design, cloning, fine-tuning, web demo) still works the same way.

## Results

Same text and seed on a 4-core Xeon (AVX-512 + AMX, no GPU), best of two runs. RTF is compute time divided by audio duration, so lower is faster and 1.0 is real time.

|                                | Upstream (bf16) | Lite bf16 (lossless) | Lite int8   | Lite int8 + AB2, 8 steps |
| ------------------------------ | --------------- | -------------------- | ----------- | ------------------------ |
| Checkpoint on disk             | 4.96 GB         | 4.96 GB              | **2.52 GB** | 2.52 GB                  |
| Load time                      | 14–21 s         | **1.1 s**            | 11 s        | 11 s                     |
| Peak memory (text to speech)   | 11.1 GB         | 5.7 GB ¹             | **4.1 GB**  | 4.1 GB                   |
| RTF                            | 3.52            | 2.64                 | 2.58        | **2.07**                 |
| WER (en) / CER (ko) ²          | 0.00 % / 0.00 % | same maths           | 0.00 % / 0.00 % | 0.00 % / 0.00 %      |
| Speaker similarity (cloning) ² | 0.919           | same maths           | 0.920       | 0.929                    |

¹ 1.3 GB private memory plus 4.4 GB of memory-mapped file pages, which are shared and evictable.
² 22 English and Korean utterances, 6 of them voice cloning; Whisper large-v3-turbo transcripts and WavLM-SV cosine similarity to the reference clip.

Method, per-component timings and the full tables are in [docs/benchmarks.md](docs/benchmarks.md). Audio samples of the same sentences from each configuration are in [benchmarks/lite/samples](benchmarks/lite/samples).

## What is in here

- **Lossless runtime fixes.** A loader that never materialises random fp32 weights and memory-maps the checkpoint, KV-cache slicing instead of masking all 8192 slots on every step, weight-norm folded out of the AudioVAE, and lazy imports so inference needs seven packages instead of the full dependency list.
- **Calibrated int8.** Transformer weights stored and executed as int8. PyTorch's stock dynamic quantization breaks VoxCPM2 (10 of 19 test utterances ran on to the length limit), so the int8 layer adds an outlier-channel split, per-row scaling and two-pass residual quantization.
- **AB2 solver.** A second-order Adams-Bashforth sampler for the LocDiT. It reuses the previous velocity, so it costs nothing extra, and 8 steps track the converged solution more closely than the default 10 Euler steps.

How each piece works, and what was tried and dropped, is in [docs/design.md](docs/design.md).

## Quick start

Install the lite runtime. `requirements-lite.txt` has the seven packages inference needs plus `soundfile` for writing wav files; on a CPU-only machine install the CPU build of PyTorch first.

```bash
git clone https://github.com/ORI-Muchim/VoxCPM2-Optimization && cd VoxCPM2-Optimization
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements-lite.txt && pip install --no-deps -e .
```

Download the original weights.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2', local_dir='pretrained_models/VoxCPM2')"
```

**Lossless loader** on the original checkpoint:

```python
import soundfile as sf
from voxcpm.lite import load_voxcpm2_lite

model = load_voxcpm2_lite("pretrained_models/VoxCPM2")  # bf16; pass precision="float32" on CPUs without AVX512-BF16 / AMX
wav = model.generate(target_text="Hello from the lite runtime.", inference_timesteps=10, cfg_value=2.0)
sf.write("out.wav", wav.squeeze(0).numpy(), model.sample_rate)
```

**Int8.** Convert once (about a minute, includes a short calibration run), then load the converted directory like any other checkpoint:

```bash
python -m voxcpm.lite.convert pretrained_models/VoxCPM2 pretrained_models/VoxCPM2-lite
```

```python
import soundfile as sf
from voxcpm import VoxCPM

tts = VoxCPM("pretrained_models/VoxCPM2-lite", enable_denoiser=False)  # detected as int8 automatically
tts.tts_model.feat_decoder.solver = "ab2"                               # optional: 8 AB2 steps instead of 10 Euler steps
wav = tts.generate("안녕하세요, 경량화된 VoxCPM입니다.", inference_timesteps=8)
sf.write("out.wav", wav, tts.tts_model.sample_rate)
```

All options are described in [docs/usage.md](docs/usage.md).

## Which configuration

| You want                                           | Use                                                    |
| -------------------------------------------------- | ------------------------------------------------------ |
| Identical output, less memory, instant start-up    | `load_voxcpm2_lite` on the original checkpoint         |
| Smallest download and memory                       | A converted int8 checkpoint                            |
| More speed on top of either                        | `solver = "ab2"` with `inference_timesteps=8` (or 6)   |

## Limitations

- **Not real time yet.** RTF is 1.8–2.6 on four cores. The LocDiT takes 55–65 % of the time and is memory-bandwidth bound, so this is about as far as training-free changes go. Getting to real time needs few-step distillation of the LocDiT, which needs a GPU and data.
- **Int8 is barely faster than bf16 on this CPU** (2.64 → 2.58), because AMX accelerates bf16. Here int8 saves memory and disk. CPUs without bf16 acceleration have to fall back to fp32, where int8 should help much more, but that has not been measured.
- **The int8 path depends on PyTorch's eager quantized kernels,** which PyTorch flags as deprecated. It works on PyTorch 2.15, is x86 only, was measured with AVX-512 VNNI only, and does not support LoRA.
- **The quality evaluation is small:** 22 utterances, automatic metrics, one machine. Listen to the samples before relying on it.
- Only VoxCPM2 is covered. VoxCPM-0.5B and VoxCPM1.5 load through the upstream path unchanged.

## Documentation

- [docs/usage.md](docs/usage.md): installation, loader and converter options, tuning, troubleshooting
- [docs/design.md](docs/design.md): where the time and memory go, how each optimization works, experiments that were dropped
- [docs/benchmarks.md](docs/benchmarks.md): environment, full results, method, how to reproduce
- [README_upstream.md](README_upstream.md): the original VoxCPM documentation

## Repository layout

```
src/voxcpm/lite/         loader, int8 layers, checkpoint converter
tests/test_lite.py       26 tests on a tiny random checkpoint
benchmarks/lite/scripts  benchmark, evaluation and analysis scripts
benchmarks/lite/samples  audio from each configuration
docs/                    usage, design, benchmarks
requirements-lite.txt    minimal inference dependencies
```

Everything else is upstream VoxCPM.

## Changes from upstream

| Path                                        | Change                                                                 |
| ------------------------------------------- | ---------------------------------------------------------------------- |
| `src/voxcpm/lite/`                          | New: low-memory loader, int8 layers, converter                         |
| `src/voxcpm/modules/minicpm4/model.py`      | KV-cache slicing option; single-op RMSNorm for float32 on CPU          |
| `src/voxcpm/modules/locdit/unified_cfm.py`  | `ab2` solver                                                           |
| `src/voxcpm/model/voxcpm2.py`               | Lazy `librosa` import; KV-cache slicing whenever `torch.compile` is off |
| `src/voxcpm/model/voxcpm.py`                | Lazy `torchaudio` import                                               |
| `src/voxcpm/core.py`                        | Lazy `huggingface_hub` import; loads converted int8 directories        |
| `tests/test_lite.py`                        | New                                                                    |
| `benchmarks/lite/`, `docs/`, `requirements-lite.txt` | New                                                           |
| `README.md`, `README_ko.md`                 | New; the upstream READMEs were renamed to `README_upstream*.md`        |

## License and credits

Apache-2.0, the same as upstream; see [LICENSE](LICENSE). VoxCPM and its weights are the work of [OpenBMB](https://github.com/OpenBMB/VoxCPM). To cite the model, use the references in [README_upstream.md](README_upstream.md#-citation).
