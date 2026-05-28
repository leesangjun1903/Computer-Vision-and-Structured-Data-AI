
# LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation

> **📌 논문 정보**
> - **제목:** LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation
> - **arXiv ID:** 2605.18739 (v2)
> - **공개일:** 2026년 5월
> - **저자:** Yukang Chen 외 15인 (NVIDIA NVLabs)
> - **GitHub:** https://github.com/NVlabs/LongLive
> - **프로젝트 페이지:** https://nvlabs.github.io/LongLive/LongLive2/

---

## 1. 핵심 주장 및 주요 기여 요약

LongLive-2.0은 긴 동영상 생성의 전체 학습 및 추론 워크플로우에 걸쳐 속도와 메모리 병목 현상을 해결하는 NVFP4 기반 병렬 인프라를 제시합니다.

저자들의 지식 범위 내에서, LongLive-2.0은 긴 동영상 생성을 위한 **최초의 NVFP4 학습 및 추론 시스템**입니다.

핵심 기여는 크게 세 축으로 요약됩니다:

| 축 | 내용 |
|---|---|
| **① 학습 인프라** | Balanced SP + NVFP4를 결합한 시퀀스 병렬 AR 학습 |
| **② 알고리즘 파이프라인** | ODE 초기화 없이 직접 AR 확산 모델로 파인튜닝 |
| **③ 추론 인프라** | W4A4 NVFP4 추론, KV 캐시 양자화, 비동기 VAE 디코딩 |

실험 결과 학습에서 최대 **2.15배** 속도 향상, 추론에서 **1.84배** 속도 향상을 달성했습니다. LongLive-2.0-5B 모델은 벤치마크에서 강력한 성능을 유지하면서 **45.7 FPS** 추론 속도를 달성합니다.

---

## 2. 해결하고자 하는 문제

### 2.1 기존 한계점

기존 긴 동영상 생성 연구들은 여전히 주목할 만한 한계를 가지고 있습니다. 인프라 수준에서는 학습과 추론 간의 공동 설계(co-design)를 탐구하는 연구가 거의 없으며, 추론의 경우 양자화 기반 방법들은 오직 사후 훈련 양자화(PTQ)만 채택하여 학습과 추론 간의 불일치 및 차선 성능을 유발합니다.

알고리즘 수준에서 Self-Forcing, Causal-Forcing 같은 기존 훈련 파이프라인은 지나치게 복잡합니다. LongLive-2.0은 이전 방법들이 요구하는 ODE 초기화, 중간 DMD 같은 복잡한 다단계 프로세스를 우회합니다.

### 2.2 문제 영역 정리

```
긴 동영상 생성의 병목:
  ├── 메모리 병목: 영상 길이 증가 → VRAM 폭발적 증가
  ├── 속도 병목: GEMM 연산 증가, KV 캐시 비대화
  ├── 알고리즘 복잡도: 다단계 파이프라인 (ODE초기화 → DMD → Long-tuning)
  └── 학습-추론 불일치: PTQ만 사용 시 훈련 분포와 추론 정밀도 불일치
```

---

## 3. 제안 방법 (수식 포함)

### 3.1 Balanced SP (Sequence Parallel) AR 학습

학습을 위해 **Balanced SP**로 인스턴스화된 시퀀스-병렬 자기회귀(AR) 학습을 도입합니다. 이는 각 랭크에서 clean-history와 noisy-target 시간적 청크를 쌍으로 묶어 SP 실행과 효율적인 teacher-forcing 레이아웃을 공동 설계하며, SP 인식 청크 VAE 인코딩을 갖는 자연스러운 teacher-forcing 마스크를 가능하게 합니다.

기존 SP가 clean-context와 noisy-target 잠재 스트림을 일반적인 연결 시퀀스로 처리하는 것과 달리, **Balanced SP는 각 GPU에 동일한 시간적 청크에서 clean 잠재값과 noisy 잠재값을 모두 할당**합니다. 이 쌍 레이아웃은 GPU 간 손실 계산 토큰의 균형을 맞추고, Ulysses All-to-All 통신 후 자연스러운 teacher-forcing 어텐션 마스크를 가능하게 합니다.

**Balanced SP의 핵심 레이아웃 (개념적 수식):**

$T$ 개의 시간적 청크를 $N$ 개의 GPU 랭크에 분배할 때:

$$\text{rank}_i \leftarrow \{z^{\text{clean}}_i, z^{\text{noisy}}_i\}, \quad i = 1, \ldots, N$$

각 랭크 $i$는 동일한 시간 청크 $i$에 대한 clean 잠재값 $z^{\text{clean}}_i$와 noisy 잠재값 $z^{\text{noisy}}_i$를 소유합니다. 어텐션 마스크 $M$은 teacher-forcing 인과 마스크를 자연스럽게 형성합니다:

$$M_{ij} = \begin{cases} 1 & \text{if } j \leq i \text{ (clean-history visible to noisy-target)} \\ 0 & \text{otherwise} \end{cases}$$

이 구조는 teacher-forcing 어텐션 마스크를 매칭하고, 랭크 간 손실 계산 토큰을 균형 있게 분배하며, VAE 인코딩이 모든 GPU에서 복제되는 대신 청크 단위로 샤딩(sharding)될 수 있도록 합니다.

### 3.2 NVFP4 양자화 (W4A4)

NVFP4는 긴 동영상 생성에 매력적인데, 영상 길이가 증가함에 따라 비중이 커지는 저정밀도 GEMM의 메모리 비용을 줄이고 가속하기 때문입니다. 따라서 AR 학습과 추론 모두에 NVFP4를 사용합니다.

FP4 양자화의 기본 수식은:

$$\hat{W} = \text{round}\left(\frac{W}{s}\right) \cdot s, \quad s = \frac{\max(|W|)}{2^{b-1}-1}, \quad b=4$$

여기서 $W$는 원본 가중치 행렬, $s$는 스케일링 팩터, $b=4$는 비트 폭을 나타냅니다. W4A4는 가중치(W)와 활성화(A) 모두 4-bit로 양자화하는 것을 의미합니다.

### 3.3 청정(Clean) 훈련 파이프라인

LongLive-2.0은 긴 동영상 데이터로 양방향 확산 모델을 직접 파인튜닝하여 길고 인터랙티브한 멀티-샷 AR 모델로 변환합니다. 이와 동시에 원래의 확산 모델에 직접 DMD 학습을 통해 독립적인(standalone) LoRA 가중치를 도출합니다.

**2단계 파이프라인:**

```
Stage 1: AR Training
  기본 양방향 Diffusion Model
       ↓ Fine-tune (Long Video Data + Balanced SP + NVFP4)
  Long AR Diffusion Model (Multi-shot, Interactive)

Stage 2: Few-step Distillation (병렬 수행)
  기본 양방향 Diffusion Model
       ↓ DMD Training
  Standalone LoRA Weights → 4~2 denoising steps 실시간 추론
```

ODE 초기화, 단기 비디오 DMD, 추가적인 long-tuning 단계에 의존하는 대신, LongLive-2.0은 기본 양방향 확산 모델을 직접 길고 인터랙티브한 멀티-샷 AR 모델로 변환합니다. 그런 다음 독립적인 LoRA 가중치가 소수-단계 추론 능력을 제공합니다.

### 3.4 추론 인프라

Blackwell GPU에서의 추론을 위해 W4A4 NVFP4 추론을 활성화하고, 메모리 절약을 위해 KV 캐시를 NVFP4로 양자화하며, 비동기 스트리밍 VAE 디코딩으로 엔드-투-엔드 처리량을 향상시킵니다.

비-Blackwell GPU 아키텍처에서는 Blackwell GPU의 속도에 맞추기 위해 SP 추론을 배포하며, 양자화된 KV 캐시가 SP의 GPU 간 통신을 줄일 수 있습니다.

---

## 4. 모델 구조

LongLive-2.0은 알고리즘과 인프라를 하나의 시스템으로 취급합니다.

```
┌─────────────────────────────────────────────────────┐
│              LongLive-2.0 프레임워크 개요             │
├──────────────────────┬──────────────────────────────┤
│    Training Infra     │      Inference Infra         │
├──────────────────────┼──────────────────────────────┤
│ • Balanced SP        │ • W4A4 NVFP4 추론             │
│ • NVFP4 양자화       │ • NVFP4 KV Cache 압축         │
│ • SP-aware VAE 인코딩│ • 병렬 역양자화 커널           │
│ • Multi-shot AR 학습 │ • 비동기 스트리밍 VAE 디코딩   │
│ • Teacher-forcing    │ • SP 추론 (비-Blackwell)       │
├──────────────────────┴──────────────────────────────┤
│         기반 모델: 양방향 Diffusion Model             │
│         (LongLive-2.0-5B: 5B 파라미터)               │
│         독립적 LoRA: 실시간(4→2 denoising steps)     │
└─────────────────────────────────────────────────────┘
```

학습 인프라(좌측): 확산 모델은 긴 동영상에 대한 AR 학습을 통해 파인튜닝되며, Balanced SP와 NVFP4 양자화가 학습 효율을 향상시킵니다. 병렬로, DMD 학습을 통해 독립적인 LoRA 가중치를 도출합니다.

추론 인프라(우측): 전체 NVFP4가 저정밀도 추론(W4A4)과 KV 캐시 압축을 가능하게 합니다. 또한 비동기 디코딩이 유휴 시간을 제거하여 생성 처리량을 극대화합니다.

---

## 5. 성능 향상

시퀀스 병렬성을 추가하면 긴 동영상 학습이 가능해지고 16s/32s 반복 시간이 52.2s와 162.7s로 줄어들며, Balanced SP는 BF16 경로를 세 가지 길이 기준으로 45.8s, 136.8s, 1196.5s로 추가 개선합니다.

| 측면 | 향상 수치 |
|---|---|
| **학습 속도 향상** | 최대 2.15× |
| **추론 속도 향상** | 최대 1.84× |
| **추론 처리량** | 45.7 FPS (LongLive-2.0-5B) |
| **추론 경로 추가 최적화** | 18.6% 처리량 추가 향상 |

최근 업데이트에서는 Triton RoPE/adaLN 커널 융합, KV 캐시 동기화 오버헤드 감소, 제자리(in-place) 양자화 KV 캐시 업데이트, 더 빠른 FP4 KV 역양자화, 핀고정 VAE 전송, LoRA-before-quantization 설정 등을 통해 전체 처리량을 **18.6%** 향상시켰습니다.

---

## 6. 한계점

논문 및 관련 문헌을 종합하면 다음과 같은 한계가 존재합니다:

1. **하드웨어 의존성:** 이는 비싼 구형 가속기 소유자에게 불편하지만, 기술적으로는 놀랍지 않습니다: 새로운 숫자 형식은 실리콘 수준에서 지원할 때만 빠릅니다. 즉, NVFP4의 최대 성능은 Blackwell GPU에서만 완전히 실현됩니다.

2. **학습-추론 정밀도 전환 불일치:** PTQ와 달리 NVFP4로 학습-추론을 통합하지만, 양자화 오차 누적에 대한 이론적 분석이 충분하지 않을 수 있습니다.

3. **비교 기반 모델 고정:** 현재 Wan2.1 계열 기반 모델에 주로 실험이 집중되어, 다른 기반 모델로의 직접 전이 가능성은 추가 검증이 필요합니다.

---

## 7. 모델의 일반화 성능 향상 가능성 🔍

이 논문에서 일반화 성능 향상과 가장 직접적으로 연관된 측면들은 다음과 같습니다.

### 7.1 클린 파이프라인이 일반화에 기여하는 방식

고품질 인프라와 데이터셋이 현저하게 깔끔한 훈련 파이프라인을 가능하게 함을 보입니다. 기존 Self-Forcing 시리즈 방법들이 ODE 초기화와 후속 분포 매칭 증류(DMD)에 의존하는 것과 달리, LongLive-2.0은 확산 모델을 직접 긴, 멀티-샷, 인터랙티브 AR 확산 모델로 튜닝합니다.

이는 단계가 줄어들수록 각 단계에서 발생하는 분포 편이(distribution shift)가 누적되지 않아, **모델이 새로운 도메인이나 프롬프트에 대해 더 안정적인 일반화 성능을 보일 가능성**이 높습니다.

### 7.2 Multi-shot 학습을 통한 일반화

멀티-샷(또는 싱글-샷) 동영상에 대한 AR 학습과 AR 학습 및 소수-단계 증류 모두에 NVFP4(또는 BF16)를 사용합니다.

멀티-샷 학습은 다양한 씬 전환 패턴과 시나리오를 학습하게 되므로, **단일 씬 비디오 생성에 국한되지 않는 광범위한 일반화**가 가능해집니다.

### 7.3 Standalone LoRA의 모듈성

첫 번째 단계는 긴 동영상 데이터를 사용하여 기본 양방향 모델에서 직접 AR 학습을 수행합니다. 독립적인 LoRA 가중치를 삽입하여 소수-단계 추론을 가능하게 함으로써, 독특하게 길고, 인터랙티브하며, 멀티-샷이고, 실시간 생성을 모두 동시에 지원하는 간소화된 파이프라인을 달성합니다.

LoRA 가중치가 독립적(standalone)이기 때문에, **서로 다른 도메인별 LoRA를 교환함으로써** 기반 AR 모델을 재훈련하지 않고도 다양한 특화 시나리오(의료 영상, 게임 시뮬레이션 등)로 일반화할 수 있는 잠재력을 가집니다.

### 7.4 Balanced SP의 일반화 가능성

NVFP4 정밀도와 결합하면 훈련 중 GPU 메모리 비용을 줄이고 GEMM 연산을 가속화하며, 이 비율은 동영상 길이가 길어질수록 증가합니다.

이는 곧 **더 긴 동영상, 더 고해상도, 더 다양한 데이터를 훈련에 포함시킬 수 있음**을 의미하며, 이는 모델의 일반화 성능을 직접적으로 향상시키는 기반이 됩니다.

### 7.5 KV 캐시 상대적 RoPE와 무한 길이 일반화

LongLive는 LongLive의 원래 RoPE를 KV 캐시 상대적 RoPE로 적응시켜 무한히 긴 동영상을 생성하는 것을 지원합니다.

이는 **훈련 시 경험하지 못한 길이의 동영상에도 일반화**할 수 있는 능력을 시사합니다.

---

## 8. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 핵심 기여 | 한계 |
|---|---|---|---|
| **Diffusion Forcing** | NeurIPS 2024 | 다음 토큰 예측 + 전체 시퀀스 확산 결합 | 실시간 성능 부족 |
| **CausVid** | CVPR 2025 | AR 확산 + DMD 결합으로 스트리밍 가능 | 긴 동영상 품질 저하 |
| **Self-Forcing** | NeurIPS 2025 Spotlight | AR 동영상 확산 모델의 새로운 훈련 패러다임 도입. 모델이 학습 중 자신의 불완전한 출력에 조건화되어 생성해야 하는 노출 편향 문제를 해결 | ODE 초기화 필요, 복잡한 파이프라인 |
| **Rolling Forcing** | 2025 | 긴 동영상 생성에서 AR 확산 모델의 우수성을 입증 | 인프라 최적화 미흡 |
| **LongLive 1.0** | ICLR 2026 | 실시간 인터랙티브 긴 동영상 생성, attention sink | NVFP4 미적용 |
| **Causal-Forcing** | ICML 2026 | AR 교사를 사용한 ODE 초기화로 아키텍처 격차 해소 제안 | 여전히 복잡한 다단계 구조 |
| **LongLive-2.0** | 2026.05 | NVFP4 통합, Balanced SP, 단순화 파이프라인 | Blackwell GPU 의존 |

최근 몇 년간 자기회귀 비디오 확산 모델의 급격한 발전이 목격되었으며, AR 비디오 확산은 프레임 수준의 AR 공식화와 각 프레임 내 확산을 채택함으로써 월드 모델링, 게임 시뮬레이션, 체화 지능, 인터랙티브 콘텐츠 생성 등 광범위한 실시간 및 인터랙티브 응용을 가능하게 합니다.

기존 방법들이 오직 사후 훈련 양자화(PTQ)만 채택하여 학습-추론 불일치를 야기하는 것과 달리, Self-Forcing, Causal-Forcing 같은 훈련 파이프라인은 지나치게 복잡하다는 점에서 LongLive-2.0의 통합 접근이 차별화됩니다.

---

## 9. 앞으로의 연구에 미치는 영향 및 고려 사항

### 9.1 미치는 영향

**① 학습-추론 공동 설계의 새로운 표준 제시**

이 접근 방식은 모델, 메모리 경로, 캐시, 디코딩을 별개로 처리하지 않고 연결된 시스템으로 취급합니다. 이는 긴 동영상에 결정적인데, 확산 자체뿐 아니라 컨텍스트, 중간 저장, 동영상 출력도 계속 성장하기 때문입니다.

이는 앞으로 긴 시퀀스를 다루는 모델(긴 텍스트, 3D, 오디오)에서도 **학습과 추론의 정밀도를 통합 설계하는 패러다임**이 주류가 될 것임을 시사합니다.

**② 단순화 파이프라인의 가능성 입증**

고품질 인프라와 데이터셋이 현저하게 깔끔한 훈련 파이프라인을 가능하게 함을 보입니다. 이는 향후 연구에서 복잡한 다단계 증류보다 **데이터 품질과 인프라 설계 자체가 알고리즘 복잡도를 대체할 수 있음**을 보여줍니다.

**③ NVFP4 생태계 확장**

LongLive는 TriAttention을 이용한 KV 캐시 압축을 지원하며, 50% KV 감소와 품질 저하 없음을 달성했습니다. 이러한 기법들은 향후 멀티모달, 긴 컨텍스트 LLM 연구에도 파급 효과가 있을 것입니다.

### 9.2 앞으로 연구 시 고려할 점

**① 비-Blackwell 환경에서의 일반화**

현재 최대 성능은 Blackwell GPU에 종속적입니다. 향후 연구에서는 Intel, AMD, 또는 엣지 디바이스에서도 유사한 효율을 달성하는 **하드웨어-불가지론적(hardware-agnostic) 양자화 프레임워크** 개발이 필요합니다.

**② 양자화 오차의 이론적 분석**

W4A4 학습-추론 통합 시 발생하는 **양자화 오차 누적**에 대한 이론적 보장이 부족합니다. 오차 한계(error bound)를 수식으로 정식화하는 연구가 필요합니다:

$$\mathcal{L}_{\text{quant}} = \mathbb{E}\left[\|f_{\theta}(x) - f_{\hat{\theta}}(x)\|^2\right] \leq \epsilon(b, \sigma^2_w)$$

여기서 $b$는 비트 폭, $\sigma^2_w$는 가중치 분산이며, 이를 이론적으로 제한하는 연구가 필요합니다.

**③ 더 긴 컨텍스트와 무한 길이 일반화**

LongLive는 원래 RoPE를 KV 캐시 상대적 RoPE로 적응시켜 무한히 긴 동영상 생성을 지원하지만, 시간적으로 먼 컨텍스트에서의 의미적 일관성 유지 메커니즘에 대한 추가 연구가 필요합니다.

**④ 멀티모달 확장 가능성**

현재는 텍스트-투-비디오에 집중되어 있으나, 음성, 3D 포인트 클라우드, 깊이 정보 등과의 **멀티모달 AR 확산 모델**로의 확장이 유망한 연구 방향입니다. Balanced SP 설계는 시퀀스 차원이 존재하는 모든 모달리티에 이론적으로 적용 가능합니다.

**⑤ 데이터 품질과 모델 성능 간 스케일링 법칙**

고품질 인프라와 데이터셋이 현저하게 깔끔한 훈련 파이프라인을 가능하게 한다는 점에서, 데이터 품질과 양이 AR 비디오 확산 모델 성능에 미치는 스케일링 법칙을 규명하는 연구가 중요해집니다.

**⑥ 소수-단계 증류와 품질 간 트레이드오프**

독립적인 LoRA 가중치로 실시간 생성(4에서 2 디노이징 스텝)으로 변환 가능하지만, 스텝 감소에 따른 품질 저하를 최소화하면서 더 적은 스텝(1스텝)으로 가능한지 연구가 필요합니다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 출처 |
|---|---|---|
| 1 | **LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation** | arXiv:2605.18739 (https://arxiv.org/abs/2605.18739) |
| 2 | LongLive-2.0 논문 HTML 전문 | https://arxiv.org/html/2605.18739v2 |
| 3 | LongLive-2.0 HuggingFace 논문 페이지 | https://huggingface.co/papers/2605.18739 |
| 4 | LongLive GitHub 공식 저장소 (NVLabs) | https://github.com/NVlabs/LongLive |
| 5 | LongLive-2.0 공식 프로젝트 페이지 | https://nvlabs.github.io/LongLive/LongLive2/ |
| 6 | ResearchGate — LongLive-2.0 PDF 페이지 | https://www.researchgate.net/publication/404992205 |
| 7 | Igor's Lab — LongLive 2.0 분석 기사 | https://www.igorslab.de/en/longlive-2-0-nvfp4-long-ai-videos-faster-more-memory-efficient/ |
| 8 | **LongLive 1.0: Real-time Interactive Long Video Generation** | arXiv:2509.22622 |
| 9 | **Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion** (NeurIPS 2025 Spotlight) | OpenReview: mSiN7i0BYH |
| 10 | **Causal Forcing: Autoregressive Diffusion Distillation Done Right** (ICML 2026) | arXiv:2602.02214 |
| 11 | **Rolling Forcing: Autoregressive Long Video Diffusion in Real Time** | arXiv:2509.25161 |
| 12 | Medium 튜토리얼 — From Teacher Forcing to Self-Forcing++ | https://medium.com/@aminfadaeinejad.edu |
| 13 | **LoRA: Low-Rank Adaptation of Large Language Models** | arXiv:2106.09685 |

> ⚠️ **정확도 주의:** 본 답변은 공개된 arXiv 초록, GitHub, 프로젝트 페이지를 기반으로 작성되었습니다. 논문 본문의 일부 세부 수식(특히 Balanced SP의 정확한 수식 표기)은 공개된 HTML 버전에서 완전히 파싱되지 않은 부분이 있어, 개념적 재구성을 포함했음을 밝힙니다. 완전한 수식 확인을 위해서는 arXiv PDF 원문을 직접 참조하시기 바랍니다.
