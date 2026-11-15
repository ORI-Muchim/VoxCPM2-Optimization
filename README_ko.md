# VoxCPM2-Optimization

[VoxCPM2](https://github.com/OpenBMB/VoxCPM)를 학습 없이 CPU에서 가볍게 돌리기 위한 최적화 모음입니다. 저메모리 로더, 보정된 int8 런타임, 더 싼 샘플러와 그 근거가 되는 측정값을 담고 있습니다.

[English](README.md) | **한국어**

> **비공식 저장소입니다.** OpenBMB/VoxCPM의 커밋 [`300e426`](https://github.com/OpenBMB/VoxCPM/commit/300e4260b509066151d623dc9c0aa922cf82232b)에 [원본 대비 변경 사항](#원본-대비-변경-사항)을 더한 것입니다. 원본 README는 [README_upstream.md](README_upstream.md)([中文](README_upstream_zh.md))로 보존했고, 거기 설명된 기능(음색 디자인, 음성 복제, 파인튜닝, 웹 데모)은 그대로 동작합니다.

## 결과

같은 문장·같은 시드, Xeon 4코어(AVX-512 + AMX, GPU 없음), 2회 중 빠른 쪽입니다. RTF는 계산 시간을 오디오 길이로 나눈 값이라 낮을수록 빠르고, 1.0이 실시간입니다.

|                              | 원본 (bf16)     | Lite bf16 (무손실) | Lite int8       | Lite int8 + AB2 8스텝 |
| ---------------------------- | --------------- | ------------------ | --------------- | --------------------- |
| 체크포인트 크기              | 4.96 GB         | 4.96 GB            | **2.52 GB**     | 2.52 GB               |
| 로딩 시간                    | 14–21초         | **1.1초**          | 11초            | 11초                  |
| 피크 메모리 (텍스트 합성)    | 11.1 GB         | 5.7 GB ¹           | **4.1 GB**      | 4.1 GB                |
| RTF                          | 3.52            | 2.64               | 2.58            | **2.07**              |
| 영어 WER / 한국어 CER ²      | 0.00 % / 0.00 % | 수학적으로 동일    | 0.00 % / 0.00 % | 0.00 % / 0.00 %       |
| 화자 유사도 (음성 복제) ²    | 0.919           | 수학적으로 동일    | 0.920           | 0.929                 |

¹ 개인 메모리 1.3 GB + mmap된 파일 페이지 4.4 GB(공유되고 회수 가능).
² 영어·한국어 22문장(그중 6문장은 음성 복제). Whisper large-v3-turbo 전사, 참조 음성과의 WavLM-SV 코사인 유사도.

측정 방법, 모듈별 시간, 전체 표는 [docs/benchmarks.md](docs/benchmarks.md)에 있습니다. 구성별로 같은 문장을 합성한 음성은 [benchmarks/lite/samples](benchmarks/lite/samples)에 있습니다.

## 들어 있는 것

- **무손실 런타임 개선.** 랜덤 fp32 가중치를 만들지 않고 체크포인트를 mmap하는 로더, 매 스텝 8192칸 전체를 마스킹하는 대신 채워진 구간만 보는 KV 캐시 슬라이싱, AudioVAE의 weight-norm 접기, 그리고 지연 import. 추론에 필요한 패키지가 전체 의존성 대신 7개로 줄어듭니다.
- **보정된 int8.** 트랜스포머 가중치를 int8로 저장하고 실행합니다. PyTorch 기본 동적 양자화는 VoxCPM2를 망가뜨리므로(테스트 19문장 중 10문장이 길이 제한까지 폭주), 이상치 채널 분리·행별 스케일링·2-패스 잔차 양자화를 더했습니다.
- **AB2 솔버.** LocDiT용 2차 Adams-Bashforth 샘플러입니다. 이전 속도를 재사용하므로 추가 비용이 없고, 8스텝이 기본 오일러 10스텝보다 수렴해에 가깝습니다.

각 부분의 동작 원리와 시도했다가 버린 것들은 [docs/design.md](docs/design.md)에 있습니다.

## 빠른 시작

경량 런타임을 설치합니다. `requirements-lite.txt`에는 추론에 필요한 패키지 7개와 wav 저장용 `soundfile`이 들어 있습니다. CPU 전용 머신이라면 PyTorch CPU 빌드를 먼저 설치하세요.

```bash
git clone https://github.com/ORI-Muchim/VoxCPM2-Optimization && cd VoxCPM2-Optimization
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements-lite.txt && pip install --no-deps -e .
```

원본 가중치를 내려받습니다.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2', local_dir='pretrained_models/VoxCPM2')"
```

**무손실 로더** (원본 체크포인트 그대로):

```python
import soundfile as sf
from voxcpm.lite import load_voxcpm2_lite

model = load_voxcpm2_lite("pretrained_models/VoxCPM2")  # bf16. AVX512-BF16 / AMX가 없는 CPU에서는 precision="float32"
wav = model.generate(target_text="Hello from the lite runtime.", inference_timesteps=10, cfg_value=2.0)
sf.write("out.wav", wav.squeeze(0).numpy(), model.sample_rate)
```

**Int8.** 한 번 변환한 뒤(약 1분, 짧은 보정 실행 포함) 변환된 폴더를 일반 체크포인트처럼 불러옵니다.

```bash
python -m voxcpm.lite.convert pretrained_models/VoxCPM2 pretrained_models/VoxCPM2-lite
```

```python
import soundfile as sf
from voxcpm import VoxCPM

tts = VoxCPM("pretrained_models/VoxCPM2-lite", enable_denoiser=False)  # int8 체크포인트임을 자동 인식
tts.tts_model.feat_decoder.solver = "ab2"                               # 선택: 오일러 10스텝 대신 AB2 8스텝
wav = tts.generate("안녕하세요, 경량화된 VoxCPM입니다.", inference_timesteps=8)
sf.write("out.wav", wav, tts.tts_model.sample_rate)
```

모든 옵션은 [docs/usage.md](docs/usage.md)에 정리되어 있습니다.

## 어떤 구성을 쓸까

| 원하는 것                                   | 사용할 것                                              |
| ------------------------------------------- | ------------------------------------------------------ |
| 출력은 그대로, 메모리는 적게, 즉시 로딩     | 원본 체크포인트에 `load_voxcpm2_lite`                  |
| 가장 작은 용량과 메모리                     | 변환된 int8 체크포인트                                 |
| 어느 쪽이든 속도를 더                       | `solver = "ab2"`와 `inference_timesteps=8` (또는 6)    |

## 한계

- **아직 실시간이 아닙니다.** 4코어에서 RTF 1.8–2.6입니다. LocDiT가 시간의 55–65 %를 쓰고 메모리 대역폭에 묶여 있어서, 학습 없이 줄일 수 있는 것은 이 정도가 한계입니다. 실시간까지 가려면 LocDiT를 소수 스텝으로 증류해야 하고, 여기에는 GPU와 데이터가 필요합니다.
- **이 CPU에서는 int8이 bf16보다 거의 빠르지 않습니다**(2.64 → 2.58). AMX가 bf16을 가속하기 때문이고, 여기서 int8의 이점은 메모리와 디스크입니다. bf16 가속이 없는 CPU는 fp32로 돌려야 해서 int8의 이득이 훨씬 클 것으로 예상하지만, 측정하지 못했습니다.
- **int8 경로는 PyTorch의 eager 양자화 커널에 의존합니다.** PyTorch가 deprecated로 표시한 기능입니다. PyTorch 2.15에서 동작하고, x86 전용이며, AVX-512 VNNI에서만 측정했고, LoRA를 지원하지 않습니다.
- **품질 평가가 작습니다.** 22문장, 자동 지표, 머신 한 대입니다. 실제로 쓰기 전에 샘플을 직접 들어보세요.
- VoxCPM2만 다룹니다. VoxCPM-0.5B와 VoxCPM1.5는 원본 경로로 그대로 로드됩니다.

## 문서

- [docs/usage.md](docs/usage.md): 설치, 로더·변환기 옵션, 튜닝, 문제 해결 (영문)
- [docs/design.md](docs/design.md): 시간과 메모리가 어디에 쓰이는지, 각 최적화의 원리, 버린 실험들 (영문)
- [docs/benchmarks.md](docs/benchmarks.md): 환경, 전체 결과, 측정 방법, 재현 방법 (영문)
- [README_upstream.md](README_upstream.md): 원본 VoxCPM 문서

## 저장소 구조

```
src/voxcpm/lite/         로더, int8 레이어, 체크포인트 변환기
tests/test_lite.py       작은 랜덤 체크포인트로 돌리는 테스트 26개
benchmarks/lite/scripts  벤치마크·평가·분석 스크립트
benchmarks/lite/samples  구성별 합성 음성
docs/                    사용법, 설계, 벤치마크
requirements-lite.txt    추론용 최소 의존성
```

나머지는 모두 원본 VoxCPM입니다.

## 원본 대비 변경 사항

| 경로                                        | 변경                                                                  |
| ------------------------------------------- | --------------------------------------------------------------------- |
| `src/voxcpm/lite/`                          | 신규: 저메모리 로더, int8 레이어, 변환기                              |
| `src/voxcpm/modules/minicpm4/model.py`      | KV 캐시 슬라이싱 옵션, CPU float32용 단일 연산 RMSNorm                |
| `src/voxcpm/modules/locdit/unified_cfm.py`  | `ab2` 솔버                                                            |
| `src/voxcpm/model/voxcpm2.py`               | `librosa` 지연 import, `torch.compile`을 쓰지 않을 때 KV 캐시 슬라이싱 |
| `src/voxcpm/model/voxcpm.py`                | `torchaudio` 지연 import                                              |
| `src/voxcpm/core.py`                        | `huggingface_hub` 지연 import, 변환된 int8 폴더 로딩                  |
| `tests/test_lite.py`                        | 신규                                                                  |
| `benchmarks/lite/`, `docs/`, `requirements-lite.txt` | 신규                                                         |
| `README.md`, `README_ko.md`                 | 신규. 원본 README는 `README_upstream*.md`로 이름을 바꿈               |

## 라이선스와 출처

원본과 같은 Apache-2.0입니다([LICENSE](LICENSE)). VoxCPM과 그 가중치는 [OpenBMB](https://github.com/OpenBMB/VoxCPM)의 작업입니다. 모델을 인용할 때는 [README_upstream.md](README_upstream.md#-citation)의 참고 문헌을 사용하세요.
