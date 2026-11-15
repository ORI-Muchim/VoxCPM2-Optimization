# Usage

How to install, load, convert and tune the lite runtime. For what it does internally see [design.md](design.md); for the numbers see [benchmarks.md](benchmarks.md).

## Requirements

- Tested on x86-64 Linux with Python 3.11, PyTorch 2.15 (CPU build) and transformers 5.23. Nothing else has been tested.
- The lossless loader only uses standard PyTorch, so it should run wherever PyTorch runs on CPU.
- The int8 path needs PyTorch's x86 quantized kernels (FBGEMM). It was measured on a CPU with AVX-512 VNNI. CPUs without VNNI are detected and use 7-bit activations automatically; see [7-bit and 8-bit activations](#7-bit-and-8-bit-activations).
- Memory for text-to-speech: about 4 GB for int8, 6 GB for bf16 (most of it reclaimable page cache), 10 GB for float32.

## Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements-lite.txt && pip install --no-deps -e .
```

`--no-deps` skips the full upstream dependency list (gradio, funasr, modelscope, datasets and so on), which inference does not need; a fresh environment ends up with 43 packages. Two features do need extra packages:

| Feature                                             | Install                                   |
| --------------------------------------------------- | ----------------------------------------- |
| Voice cloning from an audio file                    | `pip install librosa`                     |
| Denoiser, text normalization, web demo, fine-tuning | `pip install -e .` (the full upstream set) |

## Loading a model

There are three entry points. All of them return the upstream model classes, so `generate`, `generate_streaming` and the prompt-cache methods behave as documented in [README_upstream.md](../README_upstream.md).

**1. Lossless loader on the original checkpoint.**

```python
from voxcpm.lite import load_voxcpm2_lite

model = load_voxcpm2_lite("pretrained_models/VoxCPM2")                       # bfloat16, memory-mapped
model = load_voxcpm2_lite("pretrained_models/VoxCPM2", precision="float32")  # CPUs without AVX512-BF16 / AMX
```

Use `float32` when the CPU has no bf16 acceleration; bf16 still runs there but is very slow.

**2. Int8 checkpoint through the same loader.**

```python
model = load_voxcpm2_lite("pretrained_models/VoxCPM2-lite")
```

**3. Int8 checkpoint through the high-level `VoxCPM` class.**

```python
from voxcpm import VoxCPM

tts = VoxCPM("pretrained_models/VoxCPM2-lite", enable_denoiser=False)
wav = tts.generate("Hello.", inference_timesteps=10)   # 1-D float32 numpy array
```

`VoxCPM` uses the lite loader only for converted int8 directories. Given an original checkpoint it uses the upstream loader. Pass `enable_denoiser=False` (or `load_denoiser=False` to `VoxCPM.from_pretrained`) unless the full dependency set is installed.

### `load_voxcpm2_lite` arguments

| Argument           | Default      | Meaning                                                                                                                                             |
| ------------------ | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `path`             |              | Original checkpoint directory, or a directory written by the converter.                                                                             |
| `precision`        | `"bfloat16"` | `"bfloat16"` or `"float32"` for an original checkpoint. Ignored for an int8 checkpoint.                                                             |
| `load_vae_encoder` | `"lazy"`     | `"lazy"` loads the AudioVAE encoder on the first voice-cloning call, `True` loads it up front, `False` never loads it (cloning then raises).        |
| `max_length`       | from config  | KV-cache capacity in LM tokens (8192 in the released config). Only the used part is ever touched, so lowering it saves little.                       |
| `reduce_range`     | auto         | Int8 only. `True` quantizes activations to 7 bits. Auto-detected: `False` when the CPU has VNNI.                                                    |
| `two_pass`         | auto         | Int8 only. Which layers use two-pass residual quantization; see [Two-pass modes](#two-pass-modes).                                                  |
| `solver`           | `None`       | `"euler"` or `"ab2"`. `None` keeps the checkpoint's solver (Euler). See [Sampler settings](#sampler-settings).                                      |
| `mmap`             | auto         | Map float weights from the file instead of reading them. On by default for bf16 on an original checkpoint.                                          |
| `verbose`          | `True`       | Print the device and dtype line while building the model.                                                                                           |

The returned model has a `lite_info` dict with the precision, two-pass policy and activation width that were applied.

## Converting to int8

```bash
python -m voxcpm.lite.convert pretrained_models/VoxCPM2 pretrained_models/VoxCPM2-lite
```

The converter loads the original model in float, generates a few short clips to record the range of every transformer-block input, then writes:

```
config.json                 original config plus a "lite" section
model.lite.safetensors      int8 block weights with scales and float columns, int8 embedding, the rest unchanged
audiovae.lite.safetensors   AudioVAE with weight-norm folded, stored as fp16
tokenizer files             copied
```

It took 30–35 seconds on the benchmark machine. Calibration runs in bf16 where the CPU accelerates it and in float32 otherwise, which needs about 10 GB of memory.

| Option                    | Default   | Meaning                                                                                                              |
| ------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------- |
| `--keep-float`            | none      | Comma-separated stacks to leave unquantized: `base_lm`, `residual_lm`, `feat_encoder`, `feat_decoder`.               |
| `--outlier-ratio`         | `6.0`     | An input channel is computed in float when its calibrated range exceeds this many times the median channel.          |
| `--max-float-channels`    | `64`      | Upper bound on float channels per layer.                                                                             |
| `--vae-dtype`             | `float16` | Storage dtype of the AudioVAE (`float16` or `float32`). It always runs in float32.                                    |
| `--no-vae-encoder`        | off       | Drop the AudioVAE encoder. Saves 97 MB on disk and disables voice cloning.                                           |
| `--calibration-precision` | auto      | `bfloat16` or `float32` for the calibration run.                                                                     |

The same function is available from Python as `voxcpm.lite.convert.convert(src, dst, ...)`.

Converted files are named `*.lite.*` on purpose. The upstream loader does not find them and raises `FileNotFoundError` instead of silently reading int8 values as floats.

A fully fine-tuned VoxCPM2 checkpoint in the original layout converts the same way. LoRA adapters are not supported by the int8 path.

## Generating

The lite model is the upstream `VoxCPM2Model`, so the calls are the same.

```python
import soundfile as sf

wav = model.generate(target_text="Text to speak.", inference_timesteps=10, cfg_value=2.0, seed=42)
sf.write("out.wav", wav.squeeze(0).numpy(), model.sample_rate)        # 48 kHz
```

Voice design, by putting a description in parentheses at the start of the text:

```python
wav = model.generate(target_text="(A young woman, gentle and sweet voice)Hello, welcome!")
```

Voice cloning from a reference clip (needs `librosa`). The first call also loads the AudioVAE encoder and encodes the clip, which briefly raises memory use to about 5.3 GB for a 13-second clip on the int8 model:

```python
wav = model.generate(target_text="This is a cloned voice.", reference_wav_path="voice.wav")
```

Streaming, one 160 ms chunk per step:

```python
import torch

chunks = list(model.generate_streaming(target_text="Streaming example.", inference_timesteps=8))
wav = torch.cat(chunks, dim=-1)
```

## Sampler settings

The LocDiT takes most of the time, and its cost is proportional to the number of estimator evaluations, which is one fewer than `inference_timesteps`.

| Solver  | `inference_timesteps` | Evaluations per patch | Accuracy vs converged solution | RTF (int8) |
| ------- | --------------------- | --------------------- | ------------------------------ | ---------- |
| `euler` | 10 (default)          | 9                     | 18.0 dB                        | 2.58       |
| `ab2`   | 8                     | 7                     | 20.3 dB                        | 2.07       |
| `ab2`   | 6                     | 5                     | 13.7 dB                        | 1.78       |
| `euler` | 6                     | 5                     | 13.8 dB                        | not timed  |

`ab2` is the better choice at 8 steps and above. At 6 steps it is no more accurate than Euler. The 22-utterance evaluation found no difference in intelligibility or speaker similarity between these settings; see [benchmarks.md](benchmarks.md#quality).

Set the solver at load time or afterwards:

```python
model = load_voxcpm2_lite("pretrained_models/VoxCPM2-lite", solver="ab2")
model.feat_decoder.solver = "euler"          # switch back at any time
tts.tts_model.feat_decoder.solver = "ab2"    # through the VoxCPM class
```

The solver works with every precision, including the upstream loader.

## Two-pass modes

Two-pass residual quantization runs the integer matmul twice to cancel the activation quantization error. It trades speed for closeness to the float model.

| `two_pass`                       | Where the second pass runs                              | Use it for                                |
| -------------------------------- | ------------------------------------------------------- | ----------------------------------------- |
| not set (default)                | MLP `down_proj` of both LMs and the local encoder       | The measured default                      |
| `"all"`                          | Every quantized layer in every stack                    | Closest match to float, about twice the matmul cost |
| `"none"`                         | Nowhere                                                 | Fastest, lowest fidelity                  |
| `"down_proj"`                    | MLP `down_proj` in every stack, including the LocDiT    | Slightly higher LocDiT fidelity, RTF about 0.6 higher |
| `{"feat_decoder": "down_proj"}`  | A dict overrides the default for the stacks it names    | Per-stack control                         |

Fidelity against float32 for each mode is tabulated in [benchmarks.md](benchmarks.md#fidelity-against-float32). Only the default was evaluated end to end for audio quality.

## 7-bit and 8-bit activations

The integer kernel multiplies unsigned 8-bit activations by signed 8-bit weights. Without VNNI the intermediate sum of two products can exceed 16 bits and saturate, so PyTorch's convention is to use only 7 bits of activation range. The loader checks `/proc/cpuinfo` for a VNNI flag and uses the full 8 bits when it finds one, 7 bits otherwise. Override with `reduce_range=True` or `False`.

On the evaluation set the 7-bit path kept WER and CER at 0 % and speaker similarity at 0.938, with slightly lower estimated PESQ (3.10 against 3.28) and SI-SDR (18.8 against 20.2 dB).

## Threads

PyTorch uses all cores by default. All measurements used 4 threads on 4 cores. To change it:

```python
import torch
torch.set_num_threads(4)
```

## Troubleshooting

| Message or symptom                                                           | Cause and fix                                                                                                                       |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError: int8 needs a converted checkpoint`                              | `precision="int8"` was passed for an original checkpoint. Run the converter and load its output directory.                           |
| `FileNotFoundError: AudioVAE checkpoint not found` on a converted directory  | It was loaded with `VoxCPM2Model.from_local`. Use `load_voxcpm2_lite` or `VoxCPM(...)`.                                              |
| `ModuleNotFoundError: No module named 'librosa'`                             | A prompt or reference audio file was passed. `pip install librosa`.                                                                  |
| `RuntimeError: ... exported without the AudioVAE encoder`                    | The checkpoint was converted with `--no-vae-encoder`. Convert again without it to use voice cloning.                                 |
| `ValueError: LoRA is not supported with lite (int8) checkpoints`             | Use the original checkpoint for LoRA.                                                                                                |
| `ModuleNotFoundError: No module named 'torchaudio'` (or `modelscope`)        | The denoiser was enabled. Pass `enable_denoiser=False`, or install the full dependency set.                                          |
| `FutureWarning` about `torch.jit.script` or `weight_norm`                    | Raised by upstream code at import time. Harmless.                                                                                    |
| Speech keeps going long after the text has ended                             | This is what uncalibrated int8 looks like. It did not happen with converted checkpoints in the evaluation. `two_pass="all"` is the most conservative setting if it does. |
