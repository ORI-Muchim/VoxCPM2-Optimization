# VoxCPM2 경량화 결과

대상: `openbmb/VoxCPM2` (2.29B 파라미터, 한국어 지원 모델은 이것뿐이라 이 버전을 택함)
기준 커밋: `OpenBMB/VoxCPM@300e426` · 측정 환경: Xeon 4 vCPU (AVX-512 + AMX), RAM 15GB, GPU 없음, PyTorch 2.15 CPU

## 결과

학습 없이(training-free) 얻은 수치입니다. 같은 문장·시드, 2회 중 빠른 쪽.

| | 원본 (bf16) | lite bf16 (무손실) | lite int8 | lite int8 + AB2 8스텝 | lite int8 + AB2 6스텝 |
|---|---|---|---|---|---|
| 체크포인트 크기 | 4.96 GB | 4.96 GB | **2.52 GB** | 2.52 GB | 2.52 GB |
| 로딩 시간 | 14–21 s | **1.1 s** | 11 s | 11 s | 11 s |
| 피크 메모리 (텍스트 합성) | 11.1 GB | 5.7 GB ¹ | **4.1 GB** | 4.1 GB | 4.1 GB |
| RTF (한 문장) | 3.52 | 2.64 | 2.58 | **2.07** | 1.78 |
| RTF (22문장 평가셋) | 3.57 | — | 2.86 ² | 2.46 | 2.04 |
| 첫 오디오까지 (스트리밍) | 1.01 s | — | — | 0.5 s | — |
| 영어 WER / 한국어 CER | 0.00 / 0.00 % | 수학적으로 동일 | 0.00 / 0.00 % | 0.00 / 0.00 % | 0.00 / 0.00 % |
| 화자 유사도 (복제 6문장) | 0.919 | 〃 | 0.920 | 0.929 | 0.919 |
| 추정 MOS / PESQ (SQUIM) | 3.83 / 3.28 | 〃 | 3.82 / 3.25 | 4.10 / 3.13 | 4.03 / 3.20 |

¹ 개인 메모리 1.3GB + mmap된 파일 페이지 4.4GB(공유·회수 가능).
² 7비트 활성값 경로로 측정한 값(8비트 기본 경로의 평가셋 측정은 다른 작업과 겹쳐 오염되어 제외).

품질 평가는 영어 8 + 한국어 8 + 음성 복제 6 = 22문장, Whisper large-v3-turbo 전사, WavLM-SV 코사인 유사도, torchaudio SQUIM 추정치입니다. 표본이 작고 자동 지표뿐이라 미세한 음색 차이는 잡지 못합니다. `samples/`에 같은 문장의 원본/경량 버전 음성을 넣어 두었으니 직접 들어보세요.

## 무엇을 했나

**1. 무손실 최적화** (출력이 수치적으로 동일)
- 로더: 랜덤 초기화를 건너뛰고 텐서를 하나씩 최종 dtype으로 적재. 원본은 fp32로 한 번 부풀었다가 줄어들어 피크가 11GB였음.
- KV 캐시: 매 스텝 8192칸 전체를 마스킹하던 것을 채워진 구간만 슬라이싱 → base LM 스텝 138 → 85ms.
- AudioVAE: weight-norm을 접어 넣음(원본은 매 호출·매 스트리밍 청크마다 전체 가중치를 재계산). 인코더는 음성 복제 때만 지연 로딩.
- RMSNorm 단일 연산화, RoPE 테이블 축소, 무거운 import(librosa/torchaudio/huggingface_hub) 지연.
- 추론 의존성: torch, transformers, einops, pydantic, safetensors, tqdm, numpy만으로 동작 확인(복제 시 librosa 추가).

**2. int8 양자화** (`python -m voxcpm.lite.convert`)
- 트랜스포머 블록 가중치 2.26B개를 채널별 int8로 저장하고 PyTorch의 정수 커널(FBGEMM)로 실행.
- PyTorch 기본 동적 int8을 그대로 쓰면 **품질이 붕괴**합니다(레이어 SNR 7–10dB, 22문장 중 다수가 정지 예측에 실패해 4초 문장이 18–30초로 늘어남). 원인은 잔차 스트림의 소수 이상치 채널이었고, 다음 세 가지로 해결했습니다.
  - 이상치 채널 분리: 보정으로 찾은 레이어당 수 개~수십 개 채널만 float로 계산
  - 행별 스케일링
  - 2-패스 잔차 양자화: `K(x) + K(x − Q(x))`로 활성값 양자화 오차를 사실상 제거(MLP `down_proj`에 적용)
- fp32 대비 모듈별 충실도(teacher-forced): 2-패스를 전부 켜면 LM·인코더가 41–44dB로 원본 bf16 런타임(39–43dB)과 동급. 기본 설정은 속도와 절충해 33–37dB.

**3. 샘플러**
- 2차 Adams-Bashforth 솔버 추가(`feat_decoder.solver = "ab2"`). 추가 연산 없이 이전 속도를 재사용하며, 8스텝(평가 7회)이 기본 10스텝 오일러(평가 9회)보다 수렴해에 가깝습니다(20.3 vs 18.0dB).

## 프로파일에서 확인한 사실

- 시간의 55–65%는 LocDiT(패치당 평가 9회), 15–25%는 base LM 스텝.
- 둘 다 **메모리 대역폭 병목**입니다. LocDiT는 평가 1회마다 가중치 212MB(int8 기준)를 전부 읽습니다. 그래서 속도는 "가중치 바이트 × 평가 횟수"로 결정되고, 스텝 수가 가장 큰 레버입니다.
- int8 LocDiT의 fp32 대비 편차는 설정에 따라 19–25dB(기본 설정 19dB)로, 기본 10스텝 샘플러 자체의 적분 오차(18dB)와 비슷하거나 작은 수준입니다. LocDiT에도 2-패스를 켜면(`two_pass={"feat_decoder": "down_proj"}`) 25dB가 되지만 RTF가 약 0.6 늘어납니다.

## 시도했다가 버린 것

- 배치형 2-서브스텝 샘플러: 평가 6회로 기본과 같은 정확도가 나오지만 배치 평가 비용이 1.3–1.7배라 실익 없음.
- oneDNN 양자화 엔진: 가중치를 2벌 보관(5.4GB)하고 실제 모델에서는 더 느림.
- 커뮤니티 C++ 포트(VoxCPM.cpp) 비교 벤치마크: 이 세션의 권한 정책에 막혀 빌드하지 못함.

## 한계

- **아직 실시간이 아닙니다.** 4코어에서 RTF 1.8–2.6. 학습 없이 줄일 수 있는 건 여기까지가 사실상 바닥입니다.
- 이 CPU는 AMX로 bf16이 가속되어 int8의 속도 이득이 작습니다(2.64 → 2.58). bf16 가속이 없는 일반 CPU에서는 fp32와 비교하게 되어 이득이 클 것으로 예상하지만 **측정하지 못했습니다.**
- int8 경로는 PyTorch가 deprecated로 표시한 정수 커널에 의존합니다(2.15에서 동작). x86 전용이며 AVX-512 VNNI에서만 측정했습니다. VNNI 없는 CPU가 쓰는 7비트 경로는 WER/CER 0%, 유사도 0.938이지만 추정 PESQ 3.10, SI-SDR 18.8로 소폭 낮았습니다.
- int8은 LoRA 미지원. 음성 복제 시 13초 참조 음성 인코딩 중 피크 메모리가 약 5.3GB까지 오릅니다.
- 기존 테스트 `test_validate.py::test_cli_validate_exit_code`는 변경 전에도 실패합니다(무관). `test_training_data.py`는 `datasets` 미설치로 실행하지 않았습니다.
- 디스크만 더 줄이고 싶다면 커뮤니티 GGUF Q4_K(1.5GB)가 이미 있습니다(여기서 검증하지 않음).

## 더 줄이려면 (GPU·데이터 필요)

1. **LocDiT 소수 스텝 증류**: 코드에 MeanFlow 학습 경로(`dit_mean_mode`)가 이미 있어 1–2스텝 생성으로 증류하면 DiT 비용이 4–9배 줄어듭니다. 실시간 달성에 가장 직접적입니다.
2. **백본 축소 증류**: base LM(1.47B)이 메모리의 64%. 레이어를 줄인 학생 모델로 증류.
3. **4비트 실행**: PyTorch에는 커널이 없어 ONNX Runtime(MatMulNBits)이나 GGUF 런타임으로 옮겨야 합니다. 대역폭 병목이라 속도도 함께 좋아집니다.

## 사용법

```bash
pip install torch transformers einops pydantic safetensors tqdm numpy soundfile
```

```bash
pip install --no-deps -e .
```

```bash
python -m voxcpm.lite.convert /path/to/VoxCPM2 /path/to/VoxCPM2-lite
```

```python
from voxcpm import VoxCPM

tts = VoxCPM("/path/to/VoxCPM2-lite", enable_denoiser=False)
tts.tts_model.feat_decoder.solver = "ab2"
wav = tts.generate("안녕하세요, 경량화된 VoxCPM입니다.", inference_timesteps=8)
```

무손실 버전만 쓰려면 변환 없이 `from voxcpm.lite import load_voxcpm2_lite; model = load_voxcpm2_lite("/path/to/VoxCPM2")`. 음성 복제에는 `librosa`가 추가로 필요합니다.

## 이 폴더

- `samples/`: 4문장 × (원본 / int8 / int8+AB2 8스텝 / int8+AB2 6스텝)
- `scripts/`: 벤치마크(`bench.py`), 평가셋 생성·채점(`gen_eval.py`, `score_eval.py`), 충실도·샘플러 측정(`fidelity.py`), 활성값 분석(`analyze_act.py`) 등. 측정 당시 컨테이너의 절대 경로(`/home/user/...`)가 그대로 들어 있으므로 실행 전에 모델·저장소 경로를 수정해야 합니다.

변환된 int8 체크포인트(2.5GB)는 저장소에 넣지 않았습니다. 위 변환 명령으로 1분 안에 다시 만들 수 있습니다.
