# Lite runtime benchmarks

Scripts and audio behind the numbers in [docs/benchmarks.md](../../docs/benchmarks.md).

## Samples

`samples/` holds four sentences, each synthesized by four configurations with the same seed:

| File suffix                  | Configuration                          |
| ---------------------------- | -------------------------------------- |
| `__1_original.wav`           | Upstream loader, bf16, 10 Euler steps  |
| `__2_lite_int8.wav`          | Int8, 10 Euler steps                   |
| `__3_lite_int8_ab2_8steps.wav` | Int8, AB2 solver, 8 steps            |
| `__4_lite_int8_ab2_6steps.wav` | Int8, AB2 solver, 6 steps            |

`en0` and `ko0` are plain text-to-speech (the model picks a voice). `clone_en0` and `clone_ko0` clone `examples/reference_speaker.wav`. The same seed does not give the same waveform across configurations, because small numerical differences are amplified by the autoregressive loop; compare voice quality and intelligibility, not sample alignment.

## Scripts

Install the lite runtime first ([docs/usage.md](../../docs/usage.md#installation)), then point the scripts at the checkpoints:

```bash
export VOXCPM2_DIR=/path/to/VoxCPM2 VOXCPM2_LITE_DIR=/path/to/VoxCPM2-lite
```

Outputs go to `benchmarks/lite/out/` (git-ignored) unless `VOXCPM_BENCH_OUT` is set.

| Script                         | What it measures                                                                 | Extra packages                         |
| ------------------------------ | -------------------------------------------------------------------------------- | -------------------------------------- |
| `scripts/bench.py`             | Load time, memory, RTF and per-module time for one configuration                 | `soundfile` (only with `--out`)        |
| `scripts/gen_eval.py`          | Generates the 22-utterance evaluation set                                        | `soundfile`, `librosa`                 |
| `scripts/score_eval.py`        | WER / CER, speaker similarity and SQUIM estimates for generated sets             | `faster-whisper`, `librosa`, `torchaudio` |
| `scripts/fidelity.py`          | Per-module SNR against float32 with identical inputs; sampler accuracy           |                                        |
| `scripts/analysis/`            | One-off analyses: activation statistics, LocDiT probes, engine and batch timing  |                                        |

The exact commands for each table are in [docs/benchmarks.md](../../docs/benchmarks.md#reproducing).
