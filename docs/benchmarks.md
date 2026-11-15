# Benchmarks

What was measured, on what, and how to repeat it. The reasoning behind the optimizations is in [design.md](design.md).

## Environment

| | |
| --- | --- |
| CPU      | Intel Xeon, 4 vCPUs at 2.3 GHz, AVX-512 with VNNI and BF16, AMX (cloud VM) |
| Memory   | 15 GB, no swap |
| GPU      | none |
| Software | Linux, Python 3.11, PyTorch 2.15.0 CPU build (MKL, oneDNN), transformers 5.23, 4 threads |
| Model    | `openbmb/VoxCPM2`, 2.29 B parameters, bf16 checkpoint |

The VM is shared, so absolute timings move by tens of percent between sessions. Each comparison below was run back to back in one session; compare the columns of a table with each other rather than with numbers from elsewhere.

## Speed and memory

One English sentence of 17 words, seed 42, classifier-free guidance 2.0, non-streaming. Each configuration was loaded in a fresh process and run several times; the fastest run is reported.

|                         | Upstream bf16, Euler 10 | Lite bf16, Euler 10 | Lite int8, Euler 10 | Lite int8, AB2 8 | Lite int8, AB2 6 |
| ----------------------- | ----------------------- | ------------------- | ------------------- | ---------------- | ---------------- |
| Checkpoint on disk      | 4.96 GB                 | 4.96 GB             | 2.52 GB             | 2.52 GB          | 2.52 GB          |
| Load time               | 14–21 s                 | 1.1 s               | 11 s                | 11 s             | 11 s             |
| Peak RSS                | 11.1 GB                 | 5.7 GB              | 4.1 GB              | 4.1 GB           | 4.1 GB           |
| RSS while synthesizing  | 6.3 GB                  | 5.7 GB ¹            | 3.5–3.9 GB          | 3.5–3.9 GB       | 3.5–3.9 GB       |
| RTF                     | 3.52                    | 2.64                | 2.58                | 2.07             | 1.78             |

¹ 1.3 GB private memory plus 4.4 GB of memory-mapped checkpoint pages.

Time per call, in milliseconds. One LocDiT solve, one LM step of each kind and one LocEnc pass are needed for every 160 ms of audio; the AudioVAE decodes the whole 5.3-second utterance once.

|                           | Upstream bf16 | Lite bf16 | Lite int8 | Lite int8, AB2 8 | Lite int8, AB2 6 |
| ------------------------- | ------------- | --------- | --------- | ---------------- | ---------------- |
| LocDiT solve              | 303           | 261       | 264       | 194              | 143              |
| Base LM step              | 138           | 85        | 72        | 68               | 67               |
| Residual LM step          | 39            | 23        | 20        | 19               | 19               |
| LocEnc                    | 29            | 25        | 22        | 21               | 22               |
| AudioVAE decode           | 1118          | 775       | 914       | 774              | 835              |

Streaming, time to the first audio chunk: 1.01 s upstream, 0.50–0.57 s for int8 with AB2 at 8 steps.

Voice cloning adds a one-off encoding of the reference clip. With a 12.8-second clip the int8 model's peak RSS was 5.3–5.4 GB.

Converting the checkpoint to int8 took 30–35 seconds, calibration included.

## Quality

Twenty-two utterances: 8 English and 8 Korean sentences synthesized from text alone, plus 3 English and 3 Korean sentences cloning `examples/reference_speaker.wav`. Seeds are fixed per utterance.

| Configuration                        | WER en | CER ko | Speaker similarity | MOS (est.) | Lowest MOS | PESQ (est.) | SI-SDR (est.) | RTF on this set |
| ------------------------------------ | ------ | ------ | ------------------ | ---------- | ---------- | ----------- | ------------- | --------------- |
| Upstream bf16, Euler 10              | 0.00 % | 0.00 % | 0.919              | 3.83       | 2.23       | 3.28        | 20.2 dB       | 3.57            |
| Lite int8, Euler 10                  | 0.00 % | 0.00 % | 0.920              | 3.82       | 2.25       | 3.25        | 20.0 dB       | ²               |
| Lite int8, AB2 8                     | 0.00 % | 0.00 % | 0.929              | 4.10       | 2.77       | 3.13        | 20.9 dB       | 2.46            |
| Lite int8, AB2 6                     | 0.00 % | 0.00 % | 0.919              | 4.03       | 2.86       | 3.20        | 19.8 dB       | 2.04            |
| Lite int8, Euler 10, 7-bit activations | 0.00 % | 0.00 % | 0.938            | 3.95       | 2.12       | 3.10        | 18.8 dB       | 2.86            |

² Not reported: another job was running on the machine during this run.

How the metrics are computed:

- **WER / CER.** Transcripts from Whisper large-v3-turbo (faster-whisper, int8). English is lower-cased with punctuation removed and scored by words; Korean has punctuation and spaces removed and is scored by characters. Whisper writes two spoken Korean numerals as digits ("삼층" as "3층", "삼십 분" as "30분"); those are normalized before scoring.
- **Speaker similarity.** Cosine similarity between `microsoft/wavlm-base-plus-sv` embeddings of the output and of the reference clip, averaged over the 6 cloning utterances.
- **MOS, PESQ, SI-SDR.** Reference-free estimates from torchaudio's SQUIM models. They are estimates of those metrics, not the metrics themselves.

What this does and does not show. The set is easy, so the upstream model already scores 0 % and the test can only reveal gross failures, which is exactly how the uncalibrated int8 model was caught. Differences of a few hundredths in similarity or a few tenths in estimated MOS are within the noise of 22 utterances. No listening test was run.

## Fidelity against float32

A more sensitive check than transcripts: record every sub-module call of a float32 run (two utterances, 16 patches each), then feed the same inputs to another configuration and measure the SNR of its outputs against the float32 outputs.

| Configuration                         | LocEnc  | Base LM prefill | Base LM step | Residual LM prefill | Residual LM step | LocDiT solve |
| ------------------------------------- | ------- | --------------- | ------------ | ------------------- | ---------------- | ------------ |
| Upstream bf16                         | 39.4 dB | 41.1 dB         | 41.3 dB      | 42.8 dB             | 42.7 dB          | 36.4 dB      |
| Int8, two-pass everywhere             | 41.3 dB | 42.7 dB         | 43.3 dB      | 44.0 dB             | 40.9 dB          | 21.7 dB      |
| Int8, two-pass on `down_proj`         | 37.1 dB | 33.6 dB         | 35.4 dB      | 36.2 dB             | 37.0 dB          | 24.7 dB      |
| Int8, no second pass                  | 31.0 dB | 29.0 dB         | 29.9 dB      | 33.6 dB             | 29.1 dB          | 19.2 dB      |
| Int8 7-bit, two-pass everywhere       | 41.4 dB | 42.8 dB         | 43.3 dB      | 44.0 dB             | 40.8 dB          | 20.3 dB      |
| Int8 7-bit, two-pass on `down_proj`   | 33.3 dB | 27.3 dB         | 29.8 dB      | 30.7 dB             | 32.7 dB          | 18.1 dB      |
| Int8 7-bit, no second pass            | 24.3 dB | 23.1 dB         | 24.3 dB      | 27.9 dB             | 23.4 dB          | 18.3 dB      |

In each row the two-pass mode is applied to every stack. The default configuration takes the LM and LocEnc columns from "two-pass on `down_proj`" and the LocDiT column from "no second pass".

The upstream row is the yardstick: the bf16 runtime everyone uses is itself 36–43 dB away from float32. Int8 with two passes everywhere matches it in the LMs and the LocEnc. The LocDiT is different because its weight rounding is the same error on every solver step and accumulates, while bf16 rounding averages out; for scale, the default 10-step Euler sampler is itself only 18 dB from the converged solution.

## Sampler accuracy

The LocDiT inputs of 16 recorded patches, solved from the same noise with different samplers in float32 and compared with a 96-step Euler solve.

| Sampler   | Estimator evaluations | SNR vs 96 steps |
| --------- | --------------------- | --------------- |
| Euler, 4  | 3                     | 8.4 dB          |
| Euler, 5  | 4                     | 12.4 dB         |
| Euler, 6  | 5                     | 13.8 dB         |
| Euler, 8  | 7                     | 15.6 dB         |
| Euler, 10 | 9                     | 18.0 dB         |
| Euler, 16 | 15                    | 27.1 dB         |
| Euler, 32 | 31                    | 32.6 dB         |
| AB2, 4    | 3                     | 7.0 dB          |
| AB2, 5    | 4                     | 11.9 dB         |
| AB2, 6    | 5                     | 13.7 dB         |
| AB2, 8    | 7                     | 20.3 dB         |
| AB2, 10   | 9                     | 23.9 dB         |

## Caveats

- One machine, one PyTorch version. This CPU accelerates bf16 with AMX; CPUs that do not were not measured, and neither were ARM or macOS.
- Peak RSS is `ru_maxrss` of the process. For the memory-mapped bf16 loader it includes file-backed pages the OS can drop.
- Load time for the upstream loader varied from 14 to 21 seconds between runs.
- The speed tables use one sentence. The evaluation set gives somewhat higher RTF because it includes cloning utterances with longer prompts.

## Reproducing

Install the lite runtime ([usage.md](usage.md#installation)), convert a checkpoint, and set:

```bash
export VOXCPM2_DIR=/path/to/VoxCPM2 VOXCPM2_LITE_DIR=/path/to/VoxCPM2-lite
```

```bash
cd benchmarks/lite/scripts
```

Speed and memory (one process per configuration):

```bash
python bench.py $VOXCPM2_DIR --loader upstream --dtype bfloat16 --repeat 2
```

```bash
python bench.py $VOXCPM2_DIR --loader lite --precision bfloat16 --repeat 2
```

```bash
python bench.py $VOXCPM2_LITE_DIR --loader lite --repeat 2
```

```bash
python bench.py $VOXCPM2_LITE_DIR --loader lite --solver ab2 --steps 8 --repeat 2
```

Add `--streaming` for the time to the first chunk. `--loader upstream` is the stock loader with KV-cache slicing turned off, which is how upstream VoxCPM behaves; `--loader stock` leaves it on.

Quality (needs `librosa`, `faster-whisper` and `torchaudio`):

```bash
python gen_eval.py $VOXCPM2_DIR --name upstream_bf16 --loader upstream --dtype bfloat16
```

```bash
python gen_eval.py $VOXCPM2_LITE_DIR --name int8 --variants s10c2.0
```

```bash
python gen_eval.py $VOXCPM2_LITE_DIR --name int8_ab2 --solver ab2 --variants s8c2.0,s6c2.0
```

```bash
python score_eval.py ../out/eval/*
```

Fidelity and sampler accuracy (the float32 model needs about 10 GB of memory):

```bash
python fidelity.py record
```

```bash
python fidelity.py replay stock_bf16 lite:all:0 lite:down_proj:0 lite:none:0
```

```bash
python fidelity.py solver euler:10 euler:6 ab2:8 ab2:6
```

The one-off analyses behind [design.md](design.md) are in `analysis/`: `analyze_act.py` (activation statistics and simulated quantization schemes), `dit_probe.py` (single-evaluation LocDiT fidelity per layer), `engine_bench.py` (quantized engines), `dit_batch_bench.py` (evaluation time against batch size) and `prof_dit.py` (operator profile of one solve).
