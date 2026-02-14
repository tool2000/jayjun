# nanochat-ko

```
                                  _           _        _
  _ __   __ _ _ __   ___   ___| |__   __ _| |_     | | _____
 | '_ \ / _` | '_ \ / _ \ / __| '_ \ / _` | __|____| |/ / _ \
 | | | | (_| | | | | (_) | (__| | | | (_| | ||_____|   < (_) |
 |_| |_|\__,_|_| |_|\___/ \___|_| |_|\__,_|\__|    |_|\_\___/
```

> JayJun (제이준) - 준이 아빠가 만든 한국어+영어 이중언어 ChatGPT

이 프로젝트는 Andrej Karpathy의 [nanochat](https://github.com/karpathy/nanochat)을 기반으로 한 **이중언어(한국어+영어) ChatGPT 클론**입니다. A100 GPU 1장으로 약 3일 만에 처음부터 끝까지 학습할 수 있는, 완전한 LLM 파이프라인을 제공합니다.

## 모델 개요

| 항목 | 값 |
|------|---|
| 모델명 | JayJun (제이준) |
| 파라미터 수 | **~482M (0.5B)** |
| 아키텍처 | Transformer, depth=20, GQA 2:1 |
| model_dim | 768 (aspect_ratio=38) |
| 어텐션 헤드 | 6 query / 3 KV (Grouped-Query Attention) |
| head_dim | 128 |
| vocab_size | 65,536 (이중언어 BPE 토크나이저) |
| 컨텍스트 길이 | 2,048 토큰 |
| 학습 토큰 수 | ~19.3B (ratio=40) |
| 학습 언어 비율 | 영어 70% + 한국어 30% |
| 학습 환경 | A100 80GB x 1 |
| 예상 학습 시간 | ~2.5-3.5일 |

## 특징

- **이중언어 지원**: 한국어와 영어를 모두 이해하고 생성
- **GQA (Grouped-Query Attention)**: 20개 레이어를 유지하면서 0.5B로 경량화, 추론 속도 향상
- **풀스택 파이프라인**: 토크나이저 학습 → 사전학습 → 중간학습 → SFT → 평가 → 추론 → 웹 서빙
- **한국어 추론 강화**: mid/SFT 단계에 한국어 QA, KMMLU, CoT 수학 데이터 포함
- **JayJun 정체성**: "준이 아빠가 만든" 친근하고 겸손한 AI 어시스턴트

## 빠른 시작

### 요구사항

- Python 3.10+
- PyTorch 2.9+
- CUDA GPU (A100 80GB 권장)
- [uv](https://github.com/astral-sh/uv) 패키지 매니저

### 디스크 공간 요구사항

학습 전에 충분한 디스크 공간을 반드시 확보하세요. 공간 부족 시 학습이 중단됩니다.

| 항목 | 저장 경로 | 예상 용량 |
|------|-----------|-----------|
| 영어 사전학습 데이터 (615 shards) | `$NANOCHAT_BASE_DIR/base_data/` | ~61.5GB |
| 한국어 사전학습 데이터 | `$NANOCHAT_BASE_DIR/base_data_ko/` | ~18.5GB |
| HuggingFace 캐시 (mid/SFT 데이터셋) | `~/.cache/huggingface/` | ~8-15GB |
| 체크포인트 (base + mid + sft) | `*_checkpoints/` | ~7GB |
| Python 가상환경 | `.venv/` | ~3-5GB |
| 토크나이저 + 정체성 데이터 | `$NANOCHAT_BASE_DIR/` | ~0.1GB |
| **합계** | | **~100-110GB** |

> **권장: 최소 120GB, 안전하게 150GB 이상 확보**
>
> `NANOCHAT_BASE_DIR`은 기본적으로 `/datadrive/nanochat`을 사용합니다.
> 홈 디렉토리 공간이 부족할 경우, HuggingFace 캐시 경로를 변경할 수 있습니다:
> ```bash
> export HF_HOME=/datadrive/hf_cache
> ```

사전 확인:
```bash
df -h /datadrive   # 데이터 드라이브 여유 공간 확인
df -h ~            # 홈 디렉토리 여유 공간 확인
```

### 학습 실행

A100 GPU가 있는 서버에서 다음 스크립트를 실행하면 전체 파이프라인이 자동으로 진행됩니다:

```bash
bash runs/speedrun_jayjun.sh
```

장시간 학습이므로 screen 세션에서 실행하는 것을 권장합니다:

```bash
screen -L -Logfile jayjun.log -S jayjun bash runs/speedrun_jayjun.sh
```

### wandb 로깅

학습 메트릭은 기본적으로 [Weights & Biases](https://wandb.ai/)에 기록됩니다.
처음 실행 시 `wandb login`으로 API 키를 설정하세요:

```bash
pip install wandb
wandb login
```

run 이름을 지정하려면 `WANDB_RUN` 환경변수를 설정합니다:

```bash
WANDB_RUN=jayjun-v1 bash runs/speedrun_jayjun.sh
```

wandb 로깅을 비활성화하려면:

```bash
WANDB_RUN=dummy bash runs/speedrun_jayjun.sh
```

학습이 완료되면 (약 2.5~3.5일) 웹 UI로 대화할 수 있습니다:

```bash
source .venv/bin/activate
python -m scripts.chat_web
```

CLI로도 대화 가능합니다:

```bash
python -m scripts.chat_cli -p '안녕하세요! 당신은 누구예요?'
```

## 학습 파이프라인

`speedrun_jayjun.sh`는 다음 7단계를 순차적으로 실행합니다:

| 단계 | 스크립트 | 설명 |
|------|----------|------|
| 1 | `nanochat.dataset` | 영어(FineWeb-Edu) + 한국어(korean-fineweb-edu) 데이터 다운로드 |
| 2 | `scripts.tok_train` | 이중언어 BPE 토크나이저 학습 (vocab=65536, 영:한=70:30) |
| 3 | `dev.gen_jayjun_identity` | JayJun 정체성 대화 데이터 생성 (1000개 템플릿) |
| 4 | `scripts.base_train` | **사전학습** — d20 GQA 모델, ~19.3B 토큰, batch-size=32 |
| 5 | `scripts.mid_train` | **중간학습** — 대화 형식, 정체성, 한국어 QA, CoT 수학 |
| 6 | `scripts.chat_sft` | **SFT** — 지도 미세조정, 한국어 강화, 추론 데이터 |
| 7 | `nanochat.report` | 평가 리포트 생성 |

### 학습 데이터 구성

**사전학습 (Pretraining):**
- FineWeb-Edu (영어 70%) + korean-fineweb-edu (한국어 30%)
- 총 ~19.3B 토큰, ratio=40

**중간학습 (Mid-training) ~1.1M 대화:**

| 데이터셋 | 수량 | 목적 |
|----------|-----|------|
| SmolTalk | 460K | 일반 영어 대화 |
| MMLU auxiliary | 100K | 객관식 지식 |
| KoreanQA | 100K | 한국어 QA |
| KoreanSmolTalk | ~50K | 한국어 대화 |
| CoT Math | 30K | 단계별 수학 추론 |
| KMMLU | 20K | 한국어 추론/지식 |
| GSM8K | 8K | 수학 + 도구 사용 |
| SimpleSpelling | 200K | 철자 |
| SpellingBee | 80K | 글자 세기 |
| JayJun Identity | 1K x 2 | 정체성 |

**SFT (Supervised Fine-Tuning) ~45K 대화:**

| 데이터셋 | 수량 | 목적 |
|----------|-----|------|
| SmolTalk | 10K | 일반 대화 |
| KoreanQA | 10K | 한국어 QA |
| GSM8K | 8K | 수학 |
| KoreanSmolTalk | 5K | 한국어 대화 |
| CoT Math | 5K | 추론 |
| ARC-Easy/Challenge | 3.4K | 과학 |
| JayJun Identity | 1K x 2 | 정체성 강화 |
| Spelling | 600 | 철자/글자 세기 |

## 모델 아키텍처

```
입력 토큰 → [토큰 임베딩 (65536 x 768)]
              ↓
         [RMS Norm]
              ↓
     ┌── x20 Transformer Block ──┐
     │  ┌─ Attention (GQA 2:1) ─┐│
     │  │  Q: 6 heads x 128 dim ││
     │  │  K: 3 heads x 128 dim ││  ← 2개의 Q 헤드가 1개의 KV 공유
     │  │  V: 3 heads x 128 dim ││
     │  │  + Value Embedding     ││  ← 교대 레이어에 적용
     │  │  + Rotary Embedding    ││
     │  └────────────────────────┘│
     │  ┌─ MLP ─────────────────┐│
     │  │  768 → 3072 → ReLU² → ││
     │  │  3072 → 768           ││
     │  └────────────────────────┘│
     └────────────────────────────┘
              ↓
         [RMS Norm]
              ↓
     [LM Head (768 → 65536)]
              ↓
         출력 로짓
```

**핵심 기술:**
- **GQA 2:1**: Query 6개, KV 3개 — 파라미터 절감 + 추론 속도 향상
- **Muon + AdamW**: 선형 레이어는 Muon, 임베딩은 AdamW 혼합 최적화
- **Value Embedding**: ResFormer 스타일, 교대 레이어에서 입력 의존적 게이트 적용
- **Sliding Window**: `--window-pattern=L` (전체 컨텍스트)

## GPU 최적화

A100 80GB에서 기존 1.4B 모델 대비 최적화된 설정:

| 설정 | 기존 (1.4B) | 현재 (0.5B) | 효과 |
|------|------------|------------|------|
| device-batch-size (사전학습) | 8 | **32** | vocab_size=65536 logits 텐서(B×T×V) 8GiB, A100-80GB에 맞춤 |
| grad_accum_steps | 32 | **8** | total-batch-size 동일, grad accum으로 보정 |
| device-batch-size (중간학습) | 8 | **16** | |
| device-batch-size (SFT) | 4 | **8** | |
| 총 학습 시간 | ~6.5일 | **~3-4일** | |

> **참고**: `vocab_size=65536`의 이중언어 토크나이저를 사용하면 logits 텐서 `(B×T, V)`가
> 기존 nanochat(vocab=50304) 대비 ~30% 커집니다. `device-batch-size`를 줄이고
> gradient accumulation으로 보정하여 동일한 학습 결과를 유지합니다.

## 평가 벤치마크

학습 완료 후 자동으로 다음 벤치마크가 실행됩니다:

- **CORE** — DCLM 논문 기반 종합 메트릭
- **ARC-Easy / ARC-Challenge** — 과학 객관식
- **GSM8K** — 초등 수학
- **HumanEval** — Python 코딩
- **MMLU** — 종합 지식
- **ChatCORE** — 대화형 평가

결과는 `report.md` 파일에 저장됩니다.

## 프로젝트 구조

```
.
├── nanochat/                    # 핵심 라이브러리
│   ├── gpt.py                   # GPT 모델 (GQA 지원)
│   ├── tokenizer.py             # BPE 토크나이저
│   ├── engine.py                # 추론 엔진 (KV Cache)
│   ├── dataloader.py            # 분산 데이터 로더 (이중언어)
│   ├── dataset.py               # 데이터 다운로드/읽기
│   ├── muon.py                  # Muon 옵티마이저
│   ├── adamw.py                 # AdamW 옵티마이저
│   └── ...                      # checkpoint, eval, report 등
│
├── scripts/                     # 학습/평가/서빙 스크립트
│   ├── base_train.py            # 사전학습 (--n-kv-heads로 GQA 설정)
│   ├── mid_train.py             # 중간학습 (한국어+CoT 데이터 포함)
│   ├── chat_sft.py              # SFT (한국어+추론 강화)
│   ├── chat_web.py              # 웹 UI 서버
│   ├── chat_cli.py              # CLI 대화
│   └── ...                      # eval, tokenizer, RL 등
│
├── tasks/                       # 평가/학습 데이터셋
│   ├── korean_chat.py           # KoreanQA, KoreanSmolTalk, KoreanMMLU
│   ├── cot_math.py              # CoT 수학 (OpenMathInstruct-2)
│   ├── gsm8k.py, arc.py, ...   # 기존 벤치마크
│   └── common.py                # Task 베이스 클래스
│
├── runs/                        # 학습 레시피
│   ├── speedrun_jayjun.sh       # JayJun 이중언어 학습 (이 파일!)
│   ├── speedrun.sh              # 원본 nanochat 영어 학습
│   └── ...
│
├── dev/                         # 개발 도구
│   ├── gen_jayjun_identity.py   # JayJun 정체성 데이터 생성
│   └── ...
│
├── tests/                       # 테스트
│   ├── test_engine.py           # 추론 엔진 테스트
│   ├── test_cot_math.py         # CoT 수학 데이터셋 테스트
│   └── test_korean_chat.py      # 한국어 데이터셋 테스트
│
└── docs/plans/                  # 설계 문서
    ├── 2026-02-13-jayjun-0.5b-optimization-design.md
    └── 2026-02-13-jayjun-0.5b-implementation.md
```

## CPU / MPS 실행

macOS나 CPU에서도 실행 가능합니다 (성능은 매우 제한적):

```bash
bash runs/runcpu.sh
```

## 커스터마이징

- **정체성 변경**: `dev/gen_jayjun_identity.py`의 템플릿을 수정하여 AI 이름과 성격 변경
- **언어 비율 조정**: `NANOCHAT_LANG_RATIO` 환경 변수로 영어/한국어 비율 조정 (0.7 = 영어 70%)
- **모델 크기 변경**: `--depth`, `--aspect-ratio`, `--n-kv-heads` 인자로 모델 구조 조정
- **학습량 조정**: `--target-param-data-ratio`로 토큰/파라미터 비율 설정 (기본값: 40)

## 테스트

```bash
python -m pytest tests/ -v
```

## 감사

- [Andrej Karpathy](https://github.com/karpathy)의 [nanochat](https://github.com/karpathy/nanochat) 프로젝트
- [HuggingFace](https://huggingface.co/)의 FineWeb, SmolTalk, 한국어 데이터셋
- [HAERAE-HUB](https://huggingface.co/HAERAE-HUB)의 KMMLU 한국어 벤치마크
- [NVIDIA](https://huggingface.co/nvidia)의 OpenMathInstruct-2

## 인용

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 라이선스

MIT
