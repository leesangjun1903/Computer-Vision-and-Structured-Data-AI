# Segment Anything in High Quality

---

## 1. 핵심 주장 및 주요 기여 요약

**핵심 주장:** Segment Anything Model (SAM)은 강력한 zero-shot 세그멘테이션 능력을 보유하지만, 복잡한 구조를 가진 객체의 마스크 경계가 거칠고, 얇은 구조(thin structures)에서 오류가 빈번하다. HQ-SAM은 SAM의 원래 설계(promptable design), 효율성, zero-shot 일반화 성능을 완전히 보존하면서도, **최소한의 추가 파라미터(<0.5%)와 연산만으로** 고품질 마스크 예측 능력을 부여한다.

**주요 기여:**
1. **HQ-Output Token:** SAM의 mask decoder에 주입되는 학습 가능한 고품질 출력 토큰을 설계하여, 고품질 마스크 예측을 전담
2. **Global-local Feature Fusion:** ViT 인코더의 초기 레이어(local boundary details)와 최종 레이어(global semantic context) 특징을 mask decoder 특징과 융합하여 HQ-Features 생성
3. **HQSeg-44K 데이터셋 구성:** 6개 기존 데이터셋에서 44,320개의 극도로 정밀한 마스크 어노테이션을 수집한 학습 데이터셋
4. **효율적 학습:** 8 GPU에서 단 4시간 학습으로 SAM 대비 현저한 마스크 품질 향상 달성
5. **광범위한 zero-shot 검증:** 10개 다양한 세그멘테이션 벤치마크(8개 zero-shot)에서 일관된 성능 향상 입증

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

SAM은 SA-1B 데이터셋(11M 이미지, 1.1B 마스크)으로 학습되어 강력한 zero-shot 세그멘테이션을 제공하지만, 두 가지 핵심 문제가 존재한다:

1. **거친 마스크 경계(Coarse mask boundaries):** 얇은 객체 구조를 무시하거나 정밀하지 않은 경계 생성
2. **잘못된 예측(Incorrect predictions):** 얇은 구조 오해석으로 인한 깨진 마스크(broken masks), 구멍(holes), 대규모 오류 발생

이러한 문제는 자동 어노테이션, 이미지/비디오 편집 등 높은 마스크 정확도가 요구되는 응용에서 SAM의 활용을 심각하게 제한한다.

### 2.2 제안하는 방법

#### 2.2.1 SAM 기본 구조 (Preliminaries)

SAM은 세 모듈로 구성된다:
- **(a) Image Encoder:** ViT 기반 백본으로 $64 \times 64$ 크기의 이미지 임베딩 추출
- **(b) Prompt Encoder:** 포인트/박스/마스크의 상호작용적 위치 정보 인코딩
- **(c) Mask Decoder:** 2-layer transformer 기반 디코더가 이미지 임베딩과 출력/프롬프트 토큰을 결합하여 마스크 예측

#### 2.2.2 High-Quality Output Token

SAM의 원래 mask decoder에서 output token은 DETR의 object query와 유사하게, dynamic MLP 가중치를 예측한 후 mask features와 point-wise product를 수행하여 마스크를 생성한다.

HQ-SAM에서는 새로운 **HQ-Output Token** $\mathbf{t}\_{\text{HQ}} \in \mathbb{R}^{1 \times 256}$을 도입하여, SAM의 기존 output tokens $\mathbf{T}\_{\text{out}} \in \mathbb{R}^{4 \times 256}$ 및 prompt tokens $\mathbf{T}\_{\text{prompt}} \in \mathbb{R}^{N_{\text{prompt}} \times 256}$과 연결(concatenation)하여 mask decoder에 입력한다:

$$\mathbf{T}_{\text{input}} = [\mathbf{T}_{\text{out}}; \mathbf{t}_{\text{HQ}}; \mathbf{T}_{\text{prompt}}]$$

각 attention layer에서 HQ-Output Token은:
1. 다른 토큰들과 **self-attention** 수행
2. **Token-to-image attention** 및 **image-to-token attention**을 통한 특징 업데이트

업데이트된 HQ-Output Token $\mathbf{t}_{\text{HQ}}^{\prime}$으로부터 **3-layer MLP**가 dynamic convolutional kernel을 생성하고, 이를 HQ-Features와 spatially point-wise product하여 고품질 마스크를 생성한다:

$$\mathbf{M}\_{\text{HQ}} = \text{MLP}(\mathbf{t}_{\text{HQ}}^{\prime}) \otimes \mathbf{F}_{\text{HQ}}$$

여기서 $\otimes$는 point-wise product, $\mathbf{F}_{\text{HQ}}$는 융합된 HQ-Features ($256 \times 256$ 해상도)이다.

#### 2.2.3 Global-local Feature Fusion

정밀한 세그멘테이션을 위해 세 가지 다른 단계의 특징을 융합하여 HQ-Features를 구성한다:

1. **Early-layer local feature** $\mathbf{F}_{\text{early}} \in \mathbb{R}^{C \times 64 \times 64}$: ViT 인코더의 첫 번째 global attention block 출력 (경계/에지 디테일)
2. **Final-layer global feature** $\mathbf{F}_{\text{final}} \in \mathbb{R}^{C \times 64 \times 64}$: ViT 인코더의 마지막 블록 출력 (글로벌 컨텍스트)
3. **Mask decoder feature** $\mathbf{F}_{\text{mask}} \in \mathbb{R}^{C \times 256 \times 256}$: SAM의 mask decoder에서 생성된 특징 (마스크 형상 정보)

Early-layer와 Final-layer 특징은 transposed convolution으로 $256 \times 256$으로 업샘플링한 후, element-wise summation으로 융합한다:

$$\mathbf{F}_{\text{HQ}} = \text{Conv}(\mathbf{F}_{\text{mask}}) + \text{ConvT}(\mathbf{F}_{\text{early}}) + \text{ConvT}(\mathbf{F}_{\text{final}})$$

여기서 $\text{ConvT}(\cdot)$은 transposed convolution (크기 $2 \times 2$, stride 2)을 통한 업샘플링, $\text{Conv}(\cdot)$은 간단한 convolutional processing이다.

#### 2.2.4 학습 및 추론

**손실 함수:** BCE Loss와 Dice Loss의 조합으로 HQ-Output Token의 마스크 예측을 지도학습한다:

$$\mathcal{L} = \mathcal{L}_{\text{BCE}}(\mathbf{M}_{\text{HQ}}, \mathbf{M}_{\text{GT}}) + \mathcal{L}_{\text{Dice}}(\mathbf{M}_{\text{HQ}}, \mathbf{M}_{\text{GT}})$$

**학습 전략:**
- SAM의 모든 사전학습 파라미터는 **완전히 동결(freeze)**
- 학습 대상: HQ-Output Token, 3-layer MLP, 3개 convolution (HQ-Features 융합용)
- 학습 파라미터: 약 5.1M (SAM-L 대비 <0.5%)
- 학습률: 0.001, 12 epochs (10 epoch 이후 학습률 감소)
- 8 RTX 3090 GPU, 배치 크기 32, 약 4시간 (16.6K iterations)
- 다양한 프롬프트 타입(bounding box, random points, coarse mask) 혼합 학습
- Large-scale jittering으로 다양한 객체 스케일에 대한 일반화

**추론 시 Error Correction:** SAM의 Output Token이 예측한 마스크 로짓과 HQ-Output Token이 예측한 마스크 로짓을 $256 \times 256$ 해상도에서 element-wise summation한 후, $1024 \times 1024$로 업샘플링:

$$\mathbf{M}_{\text{final}} = \text{Upsample}(\mathbf{L}_{\text{SAM}} + \mathbf{L}_{\text{HQ}})$$

여기서 $\mathbf{L}\_{\text{SAM}}$은 SAM Output Token의 로짓, $\mathbf{L}_{\text{HQ}}$는 HQ-Output Token의 로짓이다.

### 2.3 모델 구조

HQ-SAM의 전체 구조는 Figure 3에 기반하며, 다음과 같이 요약된다:

| 구성요소 | 설명 |
|---|---|
| **Image Encoder** | SAM의 ViT (ViT-B/L/H) — 동결 |
| **Prompt Encoder** | SAM의 프롬프트 인코더 — 동결 |
| **Mask Decoder** | SAM의 2-layer transformer decoder — 동결, HQ-Output Token 추가 주입 |
| **HQ-Output Token** | $1 \times 256$ 학습 가능 토큰 (새로 도입) |
| **3-layer MLP** | 업데이트된 HQ-Output Token → dynamic kernel 생성 (새로 도입) |
| **Global-local Fusion Block** | 3개 convolution으로 early/final ViT feature + mask decoder feature 융합 (새로 도입) |

**파라미터 효율성 (ViT-L 기준):**

| | SAM | HQ-SAM |
|---|---|---|
| 전체 파라미터 | 1191M | 1196.1M |
| 학습 파라미터 | 1191M | **5.1M** |
| 추론 FPS | 5.0 | 4.8 |
| GPU 메모리 | 7.6G | 7.6G |

### 2.4 성능 향상

#### 2.4.1 고품질 세그멘테이션 데이터셋 (4개 HQ 데이터셋 평균, ViT-L)

| 모델 | mIoU | mBIoU |
|---|---|---|
| SAM (baseline) | 79.5 | 71.1 |
| **HQ-SAM** | **89.1** | **81.8** |

특히 DIS 데이터셋에서 mBIoU가 52.8 → 70.4로 **+17.6 포인트** 향상되었다.

#### 2.4.2 Zero-shot COCO 성능

| 모델 | $\text{AP}_B$ | AP |
|---|---|---|
| SAM | 33.3 | 48.5 |
| **HQ-SAM** | **34.4** | **49.5** |

#### 2.4.3 다양한 벤치마크에서의 일관된 향상

- **UVO:** $\text{AP}_B^{\text{strict}}$ 8.6 → 9.9 (+1.3)
- **BIG (Box):** mBIoU 70.4 → 75.3 (+4.9)
- **BIG (Mask):** mBIoU 41.8 → 75.1 (+33.3)
- **HQ-YTVIS (비디오):** $\text{AP}^B$ 30.2 → 34.0 (+3.8)
- **LVIS:** $\text{AP}_B^{\text{strict}}$ 32.1 → 32.5
- **DAVIS 2017:** $\mathcal{J}, \mathcal{F}$ 82.0 → 83.2
- **SGinW:** Grounded-HQ-SAM이 49.6 mean AP로 zero-shot 트랙 **1위** 달성

#### 2.4.4 대안 전략과의 비교 (Table 4)

| 전략 | 4 HQ mBIoU | COCO $\text{AP}_B$ |
|---|---|---|
| SAM baseline | 71.1 | 33.3 |
| SAM 전체 학습 | 12.2 | 0.2 |
| SAM decoder 미세조정 | 79.5 | 9.0 |
| CascadePSP 후처리 | 74.6 | 2.8 |
| SAM output token 미세조정 | 79.7 | 33.7 |
| **HQ-SAM** | **81.8** | **34.4** |

SAM 전체를 미세조정하거나 decoder만 미세조정하면 **catastrophic forgetting**이 발생하여 COCO zero-shot 성능이 급락한다. 후처리 네트워크 추가도 심각한 과적합을 초래한다. HQ-SAM만이 HQ 성능과 zero-shot 성능을 **동시에** 향상시킨다.

### 2.5 한계

논문에서 명시적 또는 암시적으로 드러난 한계점은 다음과 같다:

1. **극단적 환경에서의 실패:** 극도로 어두운 환경이나 매우 작은 금속 막대 등에서는 HQ-SAM도 여전히 정확한 마스크를 생성하지 못한다 (Figure 12 failure cases)
2. **Heavy ViT Encoder 의존:** HQ-SAM은 SAM의 무거운 ViT 인코더를 공유하므로, 비디오 처리에서 실시간 속도를 달성하기 어렵다. Light HQ-SAM (TinyViT)으로 41.2 FPS를 달성하지만, 성능 저하가 동반된다
3. **학습 데이터 편향 가능성:** HQSeg-44K는 6개 특정 데이터셋으로 구성되어 있어, 이 데이터셋들이 커버하지 않는 도메인(의료, 위성 등)에서의 일반화 한계가 존재할 수 있다
4. **SAM의 인코더 표현력에 의존:** 인코더를 동결하므로, SAM 인코더가 원래 잘 표현하지 못하는 패턴에 대해서는 개선이 제한적이다
5. **단일 마스크 출력 모드 의존:** Box prompt 기반 평가에서 SAM의 single mask output mode를 사용하여, 모호한 상황에서의 다중 마스크 예측 능력은 충분히 검증되지 않았다

---

## 3. 모델의 일반화 성능 향상 가능성

HQ-SAM의 일반화 성능은 이 논문의 가장 핵심적인 설계 원칙에서 비롯된다.

### 3.1 일반화를 위한 설계 원칙

**(1) 최소 적응(Minimal Adaptation) 전략:**

SAM의 사전학습 가중치를 완전히 동결하고, <0.5% 파라미터만 새로 학습함으로써:
- **Catastrophic forgetting 방지:** Table 4에서 SAM 전체 학습 시 COCO AP가 48.5 → 5.5로 급락하는 반면, HQ-SAM은 49.5로 오히려 향상
- **과적합(Overfitting) 방지:** Encoder adapter (LoRA 포함) 적용 시에도 COCO $\text{AP}_B$가 33.3 → 28.6~29.6으로 하락 (Table 9), 인코더 동결이 핵심

**(2) 토큰 학습의 일반화 특성:**

HQ-Output Token과 MLP는 특정 데이터셋의 어노테이션 편향에 과적합되지 않는다. 이는 토큰이 SAM decoder 내에서 self-attention, token-to-image attention을 통해 **이미지 콘텐츠에 적응적으로** 작동하기 때문이다. Context Token (CoOp 방식)보다 HQ-Output Token이 일관되게 우수한 성능을 보인다 (Table 2: 평균 mBIoU 77.0 vs 81.8).

### 3.2 Zero-shot 일반화 성능의 실증적 증거

**(1) 광범위한 벤치마크 커버리지:**

8개 zero-shot 벤치마크에서 일관된 향상을 보이며, 이는 다양한 도메인과 태스크를 포함한다:
- 이미지 인스턴스 세그멘테이션 (COCO, LVIS)
- 오픈월드 세그멘테이션 (UVO)
- 비디오 인스턴스 세그멘테이션 (HQ-YTVIS, YTVIS)
- 비디오 객체 세그멘테이션 (DAVIS)
- 고해상도 세그멘테이션 (BIG)
- in-the-wild 세그멘테이션 (SGinW — 25개 데이터셋)

**(2) 학습/테스트 데이터 분리 실험 (Table 16):**

DIS와 ThinObject-5K의 학습 분할을 제거해도 여전히 SAM 대비 큰 향상을 유지:

| 설정 | DIS-mIoU | ThinObject-mIoU |
|---|---|---|
| SAM baseline | 62.0 | 73.6 |
| HQ-SAM (DIS & ThinObject 모두 제거) | 72.9 (+10.9) | 82.7 (+9.1) |
| HQ-SAM (기본 HQSeg-44K) | 78.6 (+16.6) | 89.5 (+15.9) |

이는 HQ-SAM이 학습 데이터에 과적합되지 않고, **일반적인 고품질 세그멘테이션 능력**을 학습함을 입증한다.

**(3) 입력 노이즈에 대한 강건성 (Table 13):**

$$\text{Noise scale } 0.4 \text{에서: SAM mBIoU } 39.8 \;\text{vs}\; \text{HQ-SAM mBIoU } 60.3 \;(\uparrow 20.5)$$

노이즈가 증가할수록 HQ-SAM과 SAM의 성능 격차가 확대되어, HQ-SAM이 불완전한 프롬프트에도 더 강건함을 보인다.

**(4) 다양한 백본에서의 일관성 (Table 10):**

ViT-B, ViT-L, ViT-H, TinyViT 모든 백본에서 일관된 향상을 보이며, 이는 방법론의 범용성을 입증한다.

### 3.3 일반화 성능 향상의 추가 가능성

1. **학습 데이터 확장:** HQSeg-44K를 의료, 위성, 산업 등 도메인별 고품질 마스크로 확장하면 해당 도메인에서의 일반화가 향상될 것으로 기대
2. **다중 해상도 학습:** 현재 $1024 \times 1024$ 고정 해상도를 다중 해상도로 확장하면 매우 고해상도 이미지에서의 세밀한 구조 포착 능력 향상 가능
3. **SAM 2와의 통합:** Meta의 SAM 2 (2024)가 비디오 도메인으로 확장된 만큼, HQ-SAM의 접근법을 시간축으로 확장하면 비디오 세그멘테이션의 일반화 향상 가능

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 연구에 미치는 영향

**(1) Foundation Model 적응 패러다임 정립:**

HQ-SAM은 대규모 foundation model을 미세조정하지 않고, **최소한의 학습 가능 모듈만 추가**하여 특정 능력(고품질 마스크)을 향상시키는 효율적 적응 패러다임을 제시하였다. 이는 NLP의 LoRA, Adapter와 유사하지만, 세그멘테이션 태스크에 특화된 "Output Token" 기반 적응이라는 새로운 방식을 개척했다.

**(2) 데이터 효율적 학습의 가능성 입증:**

1.1B 마스크로 학습된 SAM을 44K 마스크만으로 품질 측면에서 크게 능가할 수 있음을 보여, **데이터 양보다 데이터 품질**의 중요성을 강조하였다. 이는 향후 foundation model 학습 전략에 중요한 시사점을 제공한다.

**(3) Plug-and-Play 확장성:**

HQ-SAM의 설계는 SAM 위에 "플러그인"처럼 작동하므로, Grounding-DINO + HQ-SAM (Grounded-HQ-SAM), XMem + HQ-SAM 등 **다양한 파이프라인에 즉시 통합** 가능하다. 이는 실용적 응용 관점에서 큰 영향력을 가진다.

**(4) 벤치마크 및 평가 체계 기여:**

Boundary IoU (BIoU) 등 경계 품질 중심 평가 지표의 중요성을 부각시키고, HQSeg-44K라는 고품질 학습 데이터셋을 공개하여 후속 연구의 표준 벤치마크로 활용 가능하다.

### 4.2 향후 연구 시 고려할 점

1. **인코더 적응의 안전한 방법 탐색:** 현재 인코더를 완전히 동결하지만, 과적합 없이 인코더의 일부 표현을 안전하게 개선할 수 있는 방법(예: 선택적 layer 미세조정, feature modulation) 탐색이 필요
2. **3D/점군 세그멘테이션으로의 확장:** 2D 이미지 세그멘테이션을 넘어 3D point cloud, depth-aware segmentation으로 확장 가능성 탐색
3. **자동 프롬프트 생성과의 시너지:** HQ-SAM과 자동 프롬프트 생성기(예: Grounding-DINO, SEEM)의 결합에서 프롬프트 품질이 최종 마스크 품질에 미치는 영향에 대한 체계적 연구
4. **극한 조건에서의 강건성 향상:** 극도로 어두운 환경, 매우 작은 객체, 심한 폐색 등 failure case에 대한 추가적인 해결 방안 연구
5. **실시간 응용을 위한 경량화:** Light HQ-SAM (TinyViT) 이상의 경량화를 통해 모바일/에지 디바이스에서의 실시간 고품질 세그멘테이션 달성
6. **학습 데이터의 도메인 다양성:** HQSeg-44K가 주로 자연 이미지 중심이므로, 의료(병변 경계), 위성(건물 윤곽), 산업(결함 검출) 등 특수 도메인의 고품질 어노테이션 추가가 일반화에 미치는 영향 연구

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 기여 | HQ-SAM과의 관계 |
|---|---|---|---|
| **SAM** (Kirillov et al.) [21] | 2023 | SA-1B 기반 promptable segmentation foundation model | HQ-SAM의 기반 모델. HQ-SAM이 해결하는 품질 문제의 원천 |
| **PointRend** (Kirillov et al.) [22] | 2020 | 불확실한 영역에서 adaptive point sampling으로 고해상도 마스크 렌더링 | 특정 태스크(instance seg.)에 한정, zero-shot 불가. HQ-SAM은 태스크 불문 범용 |
| **CascadePSP** (Cheng et al.) [6] | 2020 | Global-local refinement을 통한 class-agnostic 고해상도 세그멘테이션 | 후처리 기반으로 과적합 심함 (COCO $\text{AP}_B$ 2.8). HQ-SAM은 통합적 예측 |
| **Mask Transfiner** (Ke et al.) [19] | 2022 (CVPR) | Transformer 기반 고품질 인스턴스 세그멘테이션 | Closed-world, 특정 태스크 전용. HQ-SAM은 open-world zero-shot |
| **Video Mask Transfiner** (Ke et al.) [20] | 2022 (ECCV) | 비디오 인스턴스 세그멘테이션의 시간적 일관성 + 고품질 마스크 | 비디오 특화, HQ-SAM은 이미지/비디오 모두 범용 |
| **SegGPT** (Wang et al.) [43] | 2023 | In-context learning 기반 범용 세그멘테이션 | 프롬프트 방식이 다름 (in-context example vs. point/box). HQ-SAM은 SAM 호환 프롬프트 유지 |
| **SEEM** (Zou et al.) [59] | 2023 (NeurIPS) | 다양한 프롬프트(text, visual, audio)를 통합한 범용 세그멘테이션 | 다중 모달 프롬프트에 초점. HQ-SAM은 마스크 품질 향상에 초점 |
| **MobileSAM** (Zhang et al.) [52] | 2023 | TinyViT 기반 SAM 경량화 | HQ-SAM과 상호보완적 (Light HQ-SAM으로 결합됨) |
| **LoRA** (Hu et al.) [17] | 2022 (ICLR) | Low-rank adaptation으로 대규모 모델 효율적 미세조정 | SAM 인코더에 적용 시 COCO $\text{AP}_B$ 33.3 → 28.6으로 하락. HQ-SAM이 우수 |
| **CoOp/CoCoOp** (Zhou et al.) [56] | 2022 | Vision-language model을 위한 learnable prompt | Context token으로 SAM에 적용 시 HQ-SAM 대비 열등 (mBIoU 77.0 vs 81.8) |
| **Grounding-DINO** (Liu et al.) [32] | 2023 | Open-set 객체 탐지 | HQ-SAM과 결합하여 Grounded-HQ-SAM 구성, SGinW 1위 달성 |
| **DIS** (Qin et al.) [35] | 2022 (ECCV) | 이분화(dichotomous) 이미지 세그멘테이션을 위한 고정밀 데이터셋/모델 | HQSeg-44K 구성 데이터 중 하나. 특정 태스크 전용 vs HQ-SAM의 범용성 |
| **High Quality Seg.** (Shen et al.) [37] | 2022 (CVPR) | 초고해상도 이미지의 고품질 세그멘테이션 | 후처리 기반 refinement. COCO $\text{AP}_B$ 15.9로 과적합 심함 |
| **SAM 2** (Meta, Ravi et al.) | 2024 | 비디오 프롬프터블 세그멘테이션으로 SAM 확장 | HQ-SAM의 후속 연구 방향과 직접 관련. 비디오에서의 고품질 마스크 문제 여전히 존재 |
| **EfficientSAM** (Xiong et al.) | 2024 | SAM의 경량화를 위한 masked image pretraining 기반 효율적 인코더 | HQ-SAM의 경량화 방향(Light HQ-SAM)과 보완적 |

### 핵심 차별점 요약

기존 고품질 세그멘테이션 연구들은 대부분 **closed-world**, **태스크 특화**, **후처리 기반**이라는 한계를 가진다. HQ-SAM은 이러한 한계를 극복하여:

1. **Open-world zero-shot** 환경에서 작동
2. **Foundation model의 내부 구조를 재사용**하여 효율적 적응
3. **후처리가 아닌 통합적 예측**으로 과적합 방지
4. **<0.5% 파라미터, 4시간 학습**이라는 극도의 효율성 달성

---

## 참고자료

1. Ke, L., Ye, M., Danelljan, M., Liu, Y., Tai, Y.-W., Tang, C.-K., & Yu, F. (2023). "Segment Anything in High Quality." *NeurIPS 2023*. arXiv:2306.01567v2.
2. Kirillov, A., Mintun, E., Ravi, N., et al. (2023). "Segment Anything." *ICCV 2023*.
3. Kirillov, A., Wu, Y., He, K., & Girshick, R. (2020). "PointRend: Image Segmentation as Rendering." *CVPR 2020*.
4. Cheng, H.K., Chung, J., Tai, Y.-W., & Tang, C.-K. (2020). "CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement." *CVPR 2020*.
5. Ke, L., Danelljan, M., Li, X., Tai, Y.-W., Tang, C.-K., & Yu, F. (2022). "Mask Transfiner for High-Quality Instance Segmentation." *CVPR 2022*.
6. Zhou, K., Yang, J., Loy, C.C., & Liu, Z. (2022). "Learning to Prompt for Vision-Language Models." *IJCV 2022*.
7. Hu, E.J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
8. Zhang, C., et al. (2023). "Faster Segment Anything: Towards Lightweight SAM for Mobile Applications." arXiv:2306.14289.
9. Wang, X., et al. (2023). "SegGPT: Segmenting Everything in Context." arXiv:2304.03284.
10. Zou, X., et al. (2023). "Segment Everything Everywhere All at Once." *NeurIPS 2023*.
11. Shen, T., et al. (2022). "High Quality Segmentation for Ultra High-Resolution Images." *CVPR 2022*.
12. Qin, X., et al. (2022). "Highly Accurate Dichotomous Image Segmentation." *ECCV 2022*.
13. HQ-SAM 공식 GitHub: https://github.com/SysCV/SAM-HQ
