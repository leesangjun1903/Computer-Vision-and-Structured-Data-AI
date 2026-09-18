
# Marigold V2: Revisiting Diffusion Transformers for Monocular Depth Estimation

> **논문 정보**
> - **제목:** Marigold V2: Revisiting Diffusion Transformers for Monocular Depth Estimation
> - **저자:** Igor Pavlovic\*, Thiemo Wandel\*, Anton Obukhov§, Luca Bartolomei, Andrey Davydov, Fabio Tosi, Matteo Poggi, Sabine Süsstrunk, Dengxin Dai
> - **소속:** EPFL · HUAWEI Bayer Lab · University of Bologna
> - **게재:** ACM Transactions on Graphics 45(6), SIGGRAPH Asia 2026 (2026년 12월 예정)
> - **arXiv:** [2609.08084](https://arxiv.org/abs/2609.08084) (2026-09-08 공개)
> - **GitHub:** [huawei-bayerlab/marigold-v2](https://github.com/huawei-bayerlab/marigold-v2)

---

## 1. Executive Summary (10문장 이내)

단안 깊이 추정(Monocular Depth Estimation)은 장면 재구성, 계산 사진술, 로보틱스 등 다양한 분야에서 핵심적으로 활용되는 컴퓨터 비전 과제이지만, 기존 모델들은 분포 외(out-of-distribution) 입력에 대한 일반화 및 선명하고 세밀한 깊이 맵 생성에 여전히 어려움을 겪고 있다.  
이 논문은 현대적인 이미지 생성·편집 모델을 DiT(Diffusion Transformer) 아키텍처 기반으로 최고 수준의 단안 깊이 추정기로 재활용하는 기법 집합인 Marigold를 재검토한다.  
Marigold V2는 오픈소스 이미지 편집 DiT인 Qwen-Image-Edit를 단안 깊이 추정기로 재활용하는 모델과 비용 효율적인 파인튜닝 프로토콜을 제시한다.  
핵심 접근법은 사전학습된 멀티스텝 flow-matching 모델에서 단일 스텝 추론을 목표로 하며, 필요 시 양자화를 통해 모델 용량을 유지하면서도 실행 비용을 낮춘다.  
학습 과정에서 발생하는 나이브한 아티팩트를 분석하고, 두 가지 효과적인 해결책을 제안한다:  
정답 깊이로부터 추출한 의미론적 피처와 모델 내부 표현을 정렬하는 것(iREPA-depth), 그리고 새로운 Sinkhorn 기반 손실 함수를 중심으로 한 2단계 파인튜닝 프로토콜(SinkLoss)이다.  
Marigold V2는 단일 소비자 GPU에서 일주일 이내에 파인튜닝이 완료되며, 결과는 날카로운 엣지, 털, 머리카락 수준의 세부 묘사까지 충실히 재현하는 최첨단 성능을 달성한다.  
결과적으로 KITTI와 ETH3D에서 이전 최고 성능 대비 AbsRel 기준 16~26% 향상을 달성하였으며, 털, 나뭇잎, 머리카락처럼 얇은 엣지를 기존 모델들이 해결하지 못했던 수준으로 복원한다.  
나아가 Marigold V2는 표면 법선 추정(surface normals estimation)과 내재적 이미지 분해(intrinsic image decomposition) 등 다른 밀집 회귀 태스크에서도 최첨단 성능을 달성한다.  
단일 스텝 추론은 1024² 해상도에서 1.9초, 17 GB 메모리로 기존 확산 기반 방법보다 빠르고 가볍다.  
본 논문은 ACM Transactions on Graphics에 게재 예정이며 SIGGRAPH Asia 2026에서 발표될 예정이다.

---

### 1-1. 연구의 목적과 필요성

단안 깊이 추정은 어디에서나 필요하지만 매우 불량 조건(ill-posed) 문제로, 최근 모델들도 분포 외 입력에 대한 일반화와 선명하고 세밀한 깊이 맵 생성에 어려움을 겪고 있다.

기존 확산 기반 모델들의 미해결 한계로는 예측 깊이 맵에서의 세밀한 디테일 손실과 과도하게 부드러운(oversmoothed) 경계선, 점 구름(point cloud)으로 투영 시 발생하는 flying pixel 문제가 대표적이다.

대형 사전학습 DiT를 4비트로 양자화하고 rank-128 QLoRA를 추가하여 소비자용 32 GB GPU 단 한 장으로 파인튜닝하고, 몇 시간 내에 유의미한 깊이를 얻고 며칠 내에 완료할 수 있어, 80 GB 카드나 8장의 GPU 스택이 필요 없다.

> 💡 **용어 설명 — ill-posed problem (불량 조건 문제)**
> 단 한 장의 이미지에서 깊이를 추정할 때, 수학적으로 유일한 해가 존재하지 않고 무수한 3D 해석이 가능한 문제를 말합니다. 즉, 2D 이미지 → 3D 깊이의 역변환은 정보가 부족하여 원칙적으로 풀 수 없는 문제입니다.

> 💡 **용어 설명 — Flying Pixels**
> 깊이 맵을 3D 점 구름(point cloud)으로 변환할 때, 경계선 근처에서 깊이가 부정확하게 추정되어 허공에 점이 떠 있는 것처럼 보이는 아티팩트입니다.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|:---|:---|:---|
| DiT 기반 단일 스텝 추론으로 고품질 깊이 추정 가능 | Flow-matching 사전학습 모델을 단일 스텝으로 변환, 추론 속도 1.9초@1024² | Abstract, Fig. 2 |
| iREPA-depth가 세부 묘사 및 수렴 속도를 향상시킴 | GT 깊이의 DINOv3 피처와 DiT 내부 표현 정렬 → KITTI AbsRel 7.84→6.72 (30k steps) | Sec. 3, Table 3 |
| SinkLoss가 엣지 품질을 개선함 | 5×5 블록 단위 Sinkhorn 최적 수송 매칭 → SEE 지표 최고 달성 | Sec. 3, Table 2,4 |
| 소비자 GPU 1장으로 일주일 이내 파인튜닝 가능 | Hypersim + Virtual KITTI 2 (약 74K 이미지), 4-bit QLoRA rank-128 | Sec. 4, Table 1 |
| KITTI·ETH3D에서 AbsRel 16~26% 개선 | 이전 최고 대비 KITTI 6.5→5.4, ETH3D 3.8→2.8 | Table 1 |
| 다른 밀집 예측 태스크로 레시피 전이 가능 | 표면 법선, 알베도, 깊이 완성, See-through Depth에서 최고 수준 성능 | Sec. 5, Table 6 |
| SinkLoss는 다른 백본으로도 전이됨 | Stable Diffusion 1.5, FLUX.2에서도 엣지 지표 개선 확인 | Table 5 |
| 로그 깊이 표현이 최적 파라미터화 | 선형·역수 표현 대비 정량·정성 지표 모두 우수 | Ablation, Table 3 |

---

### 2-1. 해결 문제 / 제안 방법 / 모델 구조 / 성능 향상 및 한계

---

#### 🔴 해결하고자 하는 문제

기존 확산 기반 깊이 추정 모델의 주요 미해결 한계는 예측 깊이 맵에서의 세밀한 디테일 손실, 과도하게 부드러운 경계선, 그리고 점 구름으로 투영 시의 flying pixel 문제이다.

구체적으로는 다음 세 가지 문제입니다:

1. **VAE의 저주파 편향**: 이미지 VAE를 깊이 표현에 그대로 사용하면 공간적 고주파 성분(엣지, 털)이 사라집니다.
2. **Ground-truth 노이즈 문제**: 합성 GT 데이터(예: Hypersim)의 풀잎 수준까지 확대하면 깊이가 잘못 할당되어 있으므로, 노이즈 GT를 허용할 수 있는 손실 함수가 필요합니다 — 이것이 SinkLoss의 동기입니다.
3. **멀티스텝 추론의 비효율**: 기존 확산 기반 방법은 수십 스텝의 반복 디노이징이 필요합니다.

---

#### 🟢 제안 방법 (수식 포함)

##### (A) Flow-Matching 기반 단일 스텝 추론

기존 멀티스텝 확산 모델을 단일 스텝으로 증류합니다.

Flow-Matching의 조건부 벡터장(Conditional Vector Field):

$$v_\theta(x_t, t, c) = \frac{x_1 - x_0}{1} = x_1 - x_0$$

여기서:
- $x_t = (1-t)x_0 + t x_1$: 시각 $t$에서의 잠재 변수 (노이즈와 깊이 사이 선형 보간)
- $x_0$: 순수 가우시안 노이즈 ( $x_0 \sim \mathcal{N}(0, I)$ )
- $x_1$: 목표 깊이 잠재 벡터
- $c$: RGB 이미지 조건
- $t \in [0, 1]$: 시간 스텝
- $\theta$: 학습 파라미터

단일 스텝 추론: $t=0$에서 출발하여 $t=1$로 직행합니다.

$$\hat{x}_1 = x_0 + v_\theta(x_0, 0, c)$$

> 💡 **용어 설명 — Flow Matching**
> 노이즈($x_0$)에서 데이터($x_1$)로 향하는 확률적 흐름(flow)을 신경망으로 학습하는 방법입니다. 기존 DDPM처럼 수천 스텝을 밟을 필요 없이 직선 경로를 학습하면 단 1스텝 만에 샘플링이 가능합니다.

---

##### (B) iREPA-depth: 표현 정렬 손실

iREPA-depth는 DiT 중간 피처에 적용되는 표현 정렬 손실입니다. 기존 방법들은 RGB 입력에서 타겟 피처를 추출했으나, Marigold V2는 정답 깊이 맵에서 DINOv3 피처를 추출하여 모델의 내부 표현과 정렬합니다. 이는 RGB 피처가 외관 의미론을 인코딩하는 반면, 깊이 유래 피처는 구조적 배치를 더 직접적으로 드러내기 때문에 실질적인 차이를 만들어냅니다.

$$\mathcal{L}_{\text{iREPA}} = \left\| \phi_{\text{DiT}}^{(l)}(x_t, t, c) - \text{sg}(\phi_{\text{DINO}}(d_{\text{GT}})) \right\|_2^2$$

여기서:
- $\phi_{\text{DiT}}^{(l)}(\cdot)$: DiT의 $l$번째 레이어에서 추출한 중간 특징 벡터
- $\phi_{\text{DINO}}(\cdot)$: 동결된(frozen) DINOv3 인코더에서 추출한 피처
- $d_{\text{GT}}$: 정답(ground-truth) 깊이 맵
- $\text{sg}(\cdot)$: Stop-Gradient 연산 (DINO 피처는 역전파하지 않음)
- $\|\cdot\|_2^2$: L2 노름의 제곱

> 💡 **용어 설명 — DINOv3 (DINO = Self-Distillation with No Labels)**
> 레이블 없이 자기지도(self-supervised) 학습으로 강력한 시각적 표현을 학습하는 Vision Transformer 계열 모델입니다. 이미지의 의미론적·구조적 특징을 매우 잘 포착합니다.

> 💡 **용어 설명 — Stop-Gradient**
> 역전파 시 해당 경로의 그래디언트를 차단하는 연산입니다. DINO 인코더 가중치가 학습 중 변하지 않도록 보호합니다.

---

##### (C) SinkLoss: Sinkhorn 기반 블록 매칭 손실

이미지를 5×5 블록으로 타일링하여 25개의 예측 깊이와 25개의 정답 깊이를 Sinkhorn 알고리즘으로 소프트 일대일 매칭합니다. 매칭은 소프트하며, 손실은 수송 비용(transport cost)입니다.

$$\mathcal{L}_{\text{Sink}} = \sum_{b \in \text{blocks}} \langle M^*_b, C_b \rangle_F$$

여기서:
- $b$: 이미지 내 각 5×5 블록 인덱스
- $M^*_b \in \mathbb{R}^{25 \times 25}$: Sinkhorn-Knopp 알고리즘으로 구한 소프트 수송 행렬 (각 행/열의 합 = 1)
- $C_b \in \mathbb{R}^{25 \times 25}$: 비용 행렬, $C_b[i,j] = |d_{\text{pred}}^{(i)} - d_{\text{GT}}^{(j)}|$
- $\langle \cdot, \cdot \rangle_F$: Frobenius 내적 (행렬 원소곱 후 합산)
- $d_{\text{pred}}^{(i)}$: 블록 $b$ 내 $i$번째 예측 깊이 값
- $d_{\text{GT}}^{(j)}$: 블록 $b$ 내 $j$번째 정답 깊이 값

Sinkhorn-Knopp 반복 ($\epsilon$-정규화 최적 수송):

$$M^* = \arg\min_{M \in \mathcal{U}} \langle M, C \rangle_F - \epsilon H(M)$$

여기서:
- $\mathcal{U} = \{M \geq 0 \mid M\mathbf{1} = \mathbf{1}/25,\ M^\top\mathbf{1} = \mathbf{1}/25\}$: 이중확률행렬 집합
- $H(M) = -\sum_{ij} M_{ij} \log M_{ij}$: 엔트로피 정칙화항
- $\epsilon > 0$: 정칙화 강도 하이퍼파라미터

> 💡 **용어 설명 — Sinkhorn-Knopp 알고리즘 / 최적 수송(Optimal Transport)**
> "A 집합의 물건들을 B 집합으로 가장 적은 비용으로 옮기는 방법"을 찾는 이론이 최적 수송(OT)입니다. Sinkhorn 알고리즘은 엔트로피 정칙화를 통해 이 OT 문제를 빠르게 소프트하게 근사합니다. 픽셀-투-픽셀 L1 대신 블록 내에서 "비슷한 깊이끼리 유연하게 매칭"하므로 GT 노이즈에 강건합니다.

---

##### (D) 전체 학습 손실 (Stage 1)

$$\mathcal{L}_{\text{Stage1}} = \mathcal{L}_{\text{FM}} + \lambda_1 \mathcal{L}_{\text{pixel}} + \lambda_2 \mathcal{L}_{\text{grad}} + \lambda_3 \mathcal{L}_{\text{iREPA}}$$

여기서:
- $\mathcal{L}_{\text{FM}}$: Flow-Matching 잠재 공간 MSE 손실
- $\mathcal{L}_{\text{pixel}}$: 픽셀 공간 L1 손실 (디코딩 후 비교)
- $\mathcal{L}_{\text{grad}}$: 픽셀 공간 그래디언트 손실 (엣지 보존)
- $\mathcal{L}_{\text{iREPA}}$: 표현 정렬 손실 (DINOv3 피처 기반)
- $\lambda_1, \lambda_2, \lambda_3$: 각 손실의 가중치 하이퍼파라미터

**Stage 2 추가 손실:**

$$\mathcal{L}_{\text{Stage2}} = \mathcal{L}_{\text{Stage1}} + \lambda_4 \mathcal{L}_{\text{Sink}}$$

---

#### 🔵 모델 구조

추론 시 배포 모델은 VAE를 통한 단 한 번의 순전파와 DiT를 통한 순전파만으로 동작합니다. 아키텍처는 동결된 VAE 인코더, QLoRA 효율 학습으로 훈련되는 DiT, 그리고 Stage 1에서는 동결되고 Stage 2에서는 훈련되는 VAE 디코더로 구성됩니다. 복수의 손실 함수로 Sinkhorn과 iREPA가 포함됩니다.

```
[RGB 이미지]
    │
    ▼
[동결 VAE 인코더] ──────────────────────────────────────────┐
    │ RGB 잠재 벡터 (c)                                      │
    ▼                                                        │
[DiT (Qwen-Image-Edit-2509)]                                 │
 ├─ 4-bit 양자화된 동결 가중치                               │
 ├─ rank-128 QLoRA 어댑터 (학습 대상)                        │
 ├─ 입력: [가우시안 노이즈 x_0 + RGB 조건 c]                 │
 └─ 출력: 깊이 잠재 벡터 x_1                                 │
    │                                                        │
    ▼                                                        │
[VAE 디코더]                                                 │
 ├─ Stage 1: 동결                                            │
 └─ Stage 2: 학습됨 (SinkLoss 적용)                          │
    │                                                        │
    ▼                                                        │
[깊이 맵 출력 (로그-정규화, affine-invariant)]               │
    │                                                        │
    └── [DINOv3 인코더 (동결)] ◄── GT 깊이 맵 (학습 시만)   │
              iREPA-depth 정렬 ──────────────────────────────┘
```

QLoRA 기반으로 사전학습 모델 가중치를 4비트로 양자화하고 rank-128 LoRA 어댑터를 학습합니다. 배치 크기는 1로 설정하고 그래디언트 클리핑을 사용하여 최적화를 안정화하고 품질이 낮거나 노이즈 있는 학습 샘플의 영향을 줄입니다.

> 💡 **용어 설명 — QLoRA (Quantized Low-Rank Adaptation)**
> 대형 모델의 가중치를 4비트로 압축(양자화)한 뒤, 소수의 추가 학습 파라미터(Low-Rank 행렬)만 학습하는 파라미터 효율적 파인튜닝 기법입니다. 메모리를 대폭 절감하면서도 높은 성능을 유지합니다.

> 💡 **용어 설명 — VAE (Variational Autoencoder)**
> 이미지를 저차원 잠재 공간(latent space)으로 압축(인코딩)했다가 다시 이미지로 복원(디코딩)하는 신경망입니다. 확산 모델에서 픽셀 대신 잠재 공간에서 노이즈 제거를 수행하기 위해 사용됩니다.

---

#### 🟠 성능 향상

가장 강력한 수치적 성능 향상은 KITTI와 ETH3D에서 나타납니다. KITTI에서 AbsRel이 가장 강력한 비교 기준치인 6.5에서 5.4로 약 17% 감소하였습니다. ETH3D에서는 3.8에서 2.8로 약 26% 향상되었으며, 이는 두 벤치마크에서 이전 최고 대비 16~26% 향상을 보고한 초록의 수치를 뒷받침합니다.

NYUv2, ScanNet, DIODE에서도 제로샷으로 AbsRel = 3.6~5.4, δ₁ = 97~98%를 달성합니다.

HyperSim에서의 엣지 품질(Soft Edge Error, SEE 지표)에서도 테스트된 모든 세 가지 패치 크기에서 최고 점수를 기록하며, Pixel-Perfect Depth와 InfiniDepth를 능가합니다.

| 벤치마크 | 지표 | 이전 최고 | Marigold V2 | 개선율 |
|:---|:---|:---|:---|:---|
| KITTI | AbsRel ↓ | 6.5 | 5.4 | ~17% |
| ETH3D | AbsRel ↓ | 3.8 | 2.8 | ~26% |
| NYUv2/ScanNet/DIODE | AbsRel ↓ | - | 3.6~5.4 | - |
| NYUv2/ScanNet/DIODE | δ₁ ↑ | - | 97~98% | - |
| HyperSim (SEE) | SEE ↓ | - | 최고 | 전 크기 최고 |

> 💡 **용어 설명 — AbsRel (Absolute Relative Error)**
> $\text{AbsRel} = \frac{1}{N}\sum_i \frac{|d_i - \hat{d}_i|}{d_i}$ 로 정의되는 깊이 추정 표준 지표입니다. 낮을수록 좋습니다.

> 💡 **용어 설명 — δ₁ (Threshold Accuracy)**
> $\max\!\left(\frac{\hat{d}_i}{d_i}, \frac{d_i}{\hat{d}_i}\right) < 1.25$를 만족하는 픽셀의 비율입니다. 높을수록 좋습니다.

---

#### 🔴 한계

반사, 모션 블러, 초점 흐림(defocus) 영역은 여전히 모호하며, 저자들은 이를 미해결 문제로 명시합니다. 깊이 출력은 아핀 불변(affine-invariant, 상대적)이며, 희소 깊이 앵커를 이용한 테스트 시간 LoRA 피팅 절차를 추가하지 않으면 메트릭(절대) 깊이가 아닙니다.

---

## 3. 각 주장의 위치 (페이지 / Figure / Table 번호)

| 주장 | 위치 |
|:---|:---|
| 단안 깊이 추정의 일반화 문제 제시 | Sec. 1 (Introduction), p.1 |
| iREPA-depth 제안 및 설명 | Sec. 3.2, Fig. 2 |
| SinkLoss 제안 및 설명 | Sec. 3.3, Fig. 3 |
| 2단계 학습 프로토콜 (Stage 1/2) | Sec. 3.1, Fig. 2 |
| 로그 깊이 파라미터화 최적성 | Sec. 3.1, Ablation Table 3 |
| 벤치마크 비교 (NYUv2/KITTI/ETH3D/ScanNet/DIODE) | Sec. 4, **Table 1** |
| Ablation Study (iREPA vs LPIPS 등) | Sec. 4.2, **Table 3** |
| SinkLoss 백본 전이 가능성 | Sec. 4.3, **Table 5** |
| 다른 밀집 예측 태스크 전이 | Sec. 5, **Table 6** |
| 정성적 비교 (털, 나뭇잎, 엣지) | **Fig. 1, 4, 5, 6** |
| 아키텍처 개요 | **Fig. 2** |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 📌 저자가 직접 보고한 결과

> **연구 주제:**
> 나이브 학습의 아티팩트를 분석하고 두 가지 효과적인 해결책을 찾아냈습니다: 정답에서 추출한 의미론적 피처와 모델 내부 표현 정렬, 그리고 새로운 Sinkhorn 기반 손실로 구성된 2단계 파인튜닝 프로토콜. Marigold 계열을 이미지 편집 패러다임으로 확장하면서, VAE를 사용하는 확산 기반 깊이 추정기의 세밀한 디테일 손실과 과도하게 부드러운 경계선 문제에 대한 원리적 해결책을 제시합니다.

> **방법:**
> 사전학습된 멀티스텝 flow-matching 모델에서 단일 스텝 추론을 레시피로 목표하며, 필요 시 양자화를 통해 모델 용량을 유지하면서 실행 비용을 낮춥니다.

> **결과:**
> KITTI와 ETH3D에서 이전 최고 대비 AbsRel 기준 16~26% 향상된 선명하고 깔끔한 깊이 맵을 생성하며, 기존 모델들이 해결하지 못했던 털, 나뭇잎, 머리카락 수준의 엣지를 복원합니다. 표면 법선 추정, 내재적 이미지 분해 등 다른 밀집 회귀 태스크에서도 최첨단 성능을 달성합니다.

---

### 🔍 분석자(본 분석)의 해석

- **iREPA-depth의 역할 재해석:** iREPA-depth의 이점은 160,000 스텝 이후 표준 지표에서는 덜 두드러지지만, 정성적 개선은 가시적으로 남습니다. 이는 iREPA-depth가 수렴 가속기이자 지각적 정칙화기(perceptual regularizer)로 기능함을 의미합니다. 즉, 단순한 정확도 지표 이상으로 인간 시지각에 유의미한 개선을 제공하는 역할을 합니다.

- **SinkLoss의 범용성:** 저자들은 백본을 Stable Diffusion 1.5와 FLUX.2-klein-4B로 교체했을 때도 SinkLoss가 전이됨을 보입니다. 성능 향상이 Qwen이라는 특정 모델이 아닌 레시피 자체에서 비롯됨을 비교적 직접적으로 보여주는 근거입니다. 이는 향후 더 강력한 기반 모델이 등장해도 동일 레시피 적용이 가능함을 시사합니다.

- **소규모 데이터 효율성의 시사점:** 학습 데이터는 Hypersim과 Virtual KITTI 2만을 사용하여 약 74K 이미지에 불과합니다. 이는 수천만 장의 데이터를 사용하는 판별 모델 대비 극도로 효율적인 데이터 활용이며, 사전학습 DiT의 강력한 세계 모델 능력이 도메인 전이에 핵심적임을 시사합니다.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치 ⚠️

| 문제 유형 | 내용 | 비고 |
|:---|:---|:---|
| ⚠️ **선택적 벤치마크 강조** | 16~26% AbsRel 향상은 5개 벤치마크 중 2개(KITTI, ETH3D)에만 해당되며, NYUv2·ScanNet·DIODE에서의 향상은 더 작습니다. | 대표성 편향 가능성 |
| ⚠️ **데이터 규모 불균형 비교** | 500만 장 이상의 이미지로 학습한 방법들(MoGe-2, π³ 등)은 회색으로 표시되어 랭킹에서 제외됩니다. | 공정 비교 범위 제한 |
| ⚠️ **평가 프로토콜 불일치** | InfiniDepth, Lotus-2, FE2E와 자신들의 결과만 동일 프로토콜로 재현하고, 나머지는 PPD 논문에서 가져옵니다. | 측정 환경 차이 |
| ⚠️ **iREPA-depth 장기 효과 약화** | 160,000 스텝 이후에는 iREPA-depth의 이점이 표준 지표에서 덜 두드러집니다. | 장기 학습 시 효과 모호 |
| ⚠️ **정성적 개선의 객관화 한계** | 털, 나뭇잎, 엣지 재현은 선택적 시각화 예시에 의존 | 체계적 정량화 없음 |
| ⚠️ **ETH3D 해상도 조건 상이** | ETH3D는 2048×1360으로 업샘플링하여 평가하며 이는 PPD의 평가 절차를 따른 것이지만, 다른 데이터셋과 해상도 조건이 다릅니다. | 비교 조건 불균일 |

---

## 6. 논문이 답하지 않는 질문 ❓

| # | 미답 질문 |
|:---|:---|
| 1 | 비디오(temporal) 깊이 추정으로의 확장 가능성과 시간적 일관성은? |
| 2 | 야간, 열화상, 의료 영상 등 극단적 도메인에서의 일반화는? |
| 3 | 반사, 모션 블러, 초점 흐림 영역에서 왜 여전히 모호하며, 이를 해결하기 위한 구체적 방향은? |
| 4 | QLoRA rank와 양자화 비트 수의 최적 조합에 대한 체계적 탐색 결과는? |
| 5 | SinkLoss의 블록 크기(5×5)와 정칙화 강도 $\epsilon$의 민감도 분석은? |
| 6 | 합성 데이터만으로 학습했을 때, 특정 실제 도메인(의료·위성·수중)에서의 한계는? |
| 7 | 절대 메트릭 깊이를 위한 테스트 시간 LoRA 피팅의 연산 비용 및 일반화 범위는? |
| 8 | iREPA 적용 레이어 번호(layer $l$)의 최적 선택 기준은? |
| 9 | 멀티뷰 일관성 보장 방법은? (단일 이미지 특성상 뷰 간 불일치 가능) |
| 10 | 더 큰 DiT 백본(예: 더 큰 Qwen 버전)으로 스케일업 시 성능 향상 예측은? |

---

## 7. 가장 중요한 그림 5개 해석 🖼️

### Figure 1 — 정성적 비교 (Qualitative Comparison)

KITTI와 ETH3D에서 이전 대비 AbsRel 16~26% 개선된 선명한 깊이 맵을 보여주며, 털·나뭇잎·머리카락처럼 얇은 엣지를 기존 모델들이 해결하지 못했던 수준으로 복원합니다. 이 그림은 정량적 수치를 지각적으로 체감하게 해주는 핵심 근거로, 엣지 경계의 "flying pixel" 아티팩트가 Marigold V2에서 현저히 감소함을 시각적으로 보여줍니다.

**해석:** 단순한 AbsRel 개선을 넘어, 실용적 3D 재구성에서 치명적인 경계선 흐림 문제가 해결되었음을 직관적으로 증명합니다.

---

### Figure 2 — 2단계 학습 프로토콜 아키텍처

Marigold V2 학습 프로토콜을 도식화합니다. Stage 1에서는 QLoRA 어댑터 가중치만 iREPA-depth 정칙화와 함께 파인튜닝하여 이미 강력한 모델을 만들고, Stage 2에서 SinkLoss를 추가하고 VAE 디코더를 해동(unfreezing)하여 더욱 개선합니다. 추론 시에는 VAE와 DiT를 통한 단 한 번의 순전파만 필요합니다.

**해석:** 두 단계의 역할 분리(Stage 1: 의미론적 정확도, Stage 2: 엣지 품질)가 명확히 시각화되어 있어 방법론의 직관적 이해와 재현이 용이합니다.

---

### Figure 3 — SinkLoss 개념 도식

SinkLoss는 블록 단위 매칭 손실입니다. 얇은 물체에 대한 GT 깊이는 진정한 노이즈가 있어 — 렌더링된 머리카락 가닥이 전경 또는 배경 깊이에 거의 무작위로 할당될 수 있습니다. 엄격한 픽셀-투-픽셀 L1은 그 노이즈를 정확히 재현하지 못한다고 모델을 벌칙합니다. 대신, SinkLoss는 이미지를 5×5 블록으로 타일링하고, 각 블록 내에서 Sinkhorn-Knopp 알고리즘을 사용하여 25개의 예측 깊이와 25개의 GT 깊이 사이의 소프트 일대일 매칭을 찾습니다.

**해석:** 픽셀-투-픽셀 엄밀 매칭의 GT 노이즈 문제를 블록 수준 소프트 매칭으로 우아하게 해결하는 핵심 설계입니다. 이 그림은 왜 SinkLoss가 엣지 품질을 올리면서도 전체 AbsRel을 해치지 않는지를 직관적으로 보여줍니다.

---

### Figure 4 / Figure 5 — 실내외 및 야생(in-the-wild) 장면 정성적 비교

실내, 실외, 야생 장면 전반에 걸쳐 Marigold V2는 얇은 구조와 로컬 디테일을 보존하면서 flying pixel 아티팩트를 제한합니다.

**해석:** 모델이 특정 데이터셋(예: 실내 NYUv2)에 과적합하지 않고, 학습 분포 외 장면에서도 일관된 성능을 유지함을 보여줍니다. 특히 학습에 사용하지 않은 도메인(야생 사진 등)에서의 성능이 일반화 능력의 핵심 근거입니다.

---

### Figure — Ablation Study (iREPA-depth vs. 대안들) [Table 3 / Fig. 대응]

30,000 스텝 실험에서 깊이 타겟 iREPA 변형이 NYUv2, KITTI, ETH3D, ScanNet, DIODE 전반에 걸쳐 가장 강력한 전체 성능을 제공합니다. 예를 들어 KITTI AbsRel을 픽셀-및-그래디언트 기준선의 7.84에서 6.72로 줄이며, 동시에 세부 묘사도 향상합니다.

iREPA-depth는 RGB 피처 기반 iREPA와 직접적인 LPIPS 지각 손실 모두를 능가하며, 특히 나뭇잎과 같이 복잡한 영역에서 차이가 두드러집니다. Stage 2에서 추가된 SinkLoss는 평균 AbsRel이나 δ₁을 해치지 않으면서 엣지 지표(SEE)를 크게 향상시킵니다.

**해석:** 각 구성 요소의 기여를 독립적으로 측정하여, 성능 향상이 특정 요소에 의한 것임을 명확히 입증합니다. iREPA-depth와 SinkLoss가 서로 보완적으로 작용함을 보여주는 핵심 실험입니다.

---

## 8. 결론 — 시사점, 후속 연구 계획, 추가 방향

### 8-1. 모델의 일반화 성능 향상 가능성

Marigold V2는 정량적 정확도와 지각적 품질을 동시에 추구하는 실용적인 확산 기반 단안 깊이 추정기입니다. iREPA 정칙화와 새로운 SinkLoss에 의해 조향되는 2단계 파인튜닝 프로세스를 통해 Qwen-Image-Edit에서 파생됩니다.

**일반화 성능 향상의 핵심 동인:**

1. **대규모 생성 사전 지식(Generative Prior) 활용**: Marigold V1은 사전학습 확산 모델이 밀집 예측을 위한 강력한 의미론적·기하학적 사전 지식을 담고 있음을 입증했습니다. V2는 이를 DiT 규모로 확장하여 더 강력한 일반화 기반을 마련합니다.

2. **최소 데이터로 최대 효과**: 약 74K 이미지(Hypersim + Virtual KITTI 2)만으로 소비자 GPU 한 장에서 학습하면서 날카로운 엣지와 머리카락 수준의 세부 묘사까지 재현하는 최첨단 성능을 달성합니다. 이는 데이터 효율적 일반화의 가능성을 열어줍니다.

3. **다중 태스크 전이 가능성**: 레시피는 표면 법선(4개 데이터셋에서 최고 또는 근접), 알베도(HyperSim에서 최고 PSNR·LPIPS), 희소 측정을 이용한 테스트 시간 LoRA 메트릭 깊이 완성, LayeredDepth-Syn 학습으로 AbsRel 13.7→8.2 개선된 유리 뒤 See-through 깊이로 확장됩니다.

4. **백본 독립적 레시피**: 다양한 DiT 백본 변경과 SinkLoss의 효과를 연구하며, 추가 학습 스텝의 영향과 2단계에서의 SinkLoss 영향도 분석합니다. SinkLoss가 Stable Diffusion과 FLUX에서도 전이됨은 레시피 자체의 범용성을 입증합니다.

**⚠️ 일반화의 남은 한계:**

- 반사·모션 블러·초점 흐림은 미해결 문제로 남아 있으며, 이러한 도전적 조건에서의 일반화는 후속 연구 과제입니다.
- 학습 데이터가 합성(synthetic) 데이터에 한정되어 있어, 특수 실제 도메인(의료 내시경, 위성, 수중)에서의 일반화 여부는 검증되지 않았습니다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

#### 주요 관련 연구 계보

| 연구/모델 | 연도 | 방법 | 특징 |
|:---|:---|:---|:---|
| MiDaS | 2020 | DPT (판별 모델) | 다중 데이터셋 혼합 학습으로 일반화 |
| DPT | 2021 | ViT 기반 밀집 예측 | Transformer를 깊이 추정에 적용 |
| Depth Anything V1/V2 | 2024 | 판별 ViT, 대규모 비레이블 데이터 | 5M+ 이미지 학습, 강한 일반화 |
| **Marigold V1** | 2024 | Stable Diffusion 기반 | 생성 사전 지식, 합성 데이터만 사용 |
| DepthFM | 2024 | Flow Matching | 빠른 추론, Marigold보다 적은 스텝 |
| BetterDepth | 2024 | 확산 기반 정제 | 거친 예측을 확산 모델로 개선 |
| Pixel-Perfect Depth (PPD) | 2025 | - | 현재 Marigold V2의 주요 비교 대상 |
| Lotus-2 | 2025 | DiT 기반 | V2의 비교 대상 |
| **Marigold V2** | **2026** | **DiT + Flow Matching + QLoRA** | **본 논문** |

Marigold V1 계열과 동시대에 생성적 확산 모델에서 파생된 깊이 추정 접근법이 등장하였으며, 이 파인튜닝에 사용된 깊이 주석 학습 데이터 양이 매우 적음에도 인상적인 결과를 보여줍니다.

---

#### 이 논문이 앞으로의 연구에 미치는 영향

1. **"소규모 데이터 + 강력한 생성 사전" 패러다임의 확립:** 수천만 장 데이터를 요구하는 판별 모델과 달리, 74K 합성 이미지만으로 최첨단 성능을 달성한 것은 데이터 효율적 밀집 예측 연구의 새로운 기준점을 제시합니다.

2. **SinkLoss의 광범위한 적용 가능성:** SinkLoss는 엣지 지표(SEE)를 크게 향상시키면서도 AbsRel이나 δ₁을 해치지 않습니다. 저자들은 백본을 Stable Diffusion 1.5와 FLUX.2-klein-4B로 교체해도 SinkLoss가 전이됨을 보입니다. 이는 SinkLoss가 다른 밀집 예측 작업이나 생성 모델 파인튜닝에도 폭넓게 적용될 수 있는 범용 손실 함수임을 시사합니다.

3. **접근성 민주화:** 소비자 GPU 한 장에서 일주일 이내 파인튜닝이 가능하여 개인 연구자나 소규모 연구실의 수준에서도 최첨단 결과를 달성할 수 있습니다. 이는 대형 연구 기관 중심이던 대규모 생성 모델 연구의 접근 장벽을 낮춥니다.

4. **SIGGRAPH Asia 게재의 파급력:** ACM Transactions on Graphics 45(6)에 게재되고 SIGGRAPH Asia 2026에서 발표됩니다. SIGGRAPH는 컴퓨터 그래픽스 분야의 최고 학술대회로, 3D 재구성·렌더링·시각 효과 연구자들에게도 큰 영향을 미칠 것으로 예상됩니다.

---

#### 앞으로 연구 시 고려할 점

| 고려 사항 | 내용 |
|:---|:---|
| **GT 노이즈 처리** | SinkLoss의 블록 크기와 $\epsilon$ 하이퍼파라미터 탐색이 중요. 도메인마다 GT 노이즈 특성이 다를 수 있음. |
| **메트릭 깊이 확장** | 현재 출력은 affine-invariant 상대 깊이이므로, 로봇·자율주행 적용 시 절대 깊이 변환 파이프라인 설계 필요. |
| **비디오 연속성** | 단일 이미지 모델은 시간적 일관성 보장 불가. 비디오 기반 확산 모델(예: SVD)로의 확장 시 시간적 정칙화 기법 필요. |
| **도메인 특화 파인튜닝** | 의료·위성·수중 등 특수 도메인에서는 해당 도메인의 합성 데이터 생성 후 Marigold V2 레시피 적용 가능. |
| **SinkLoss 일반화** | SinkLoss를 깊이 이외의 광학 흐름(optical flow), 표면 법선, 알베도 등 다른 밀집 예측에도 적용할 때 블록 크기 및 비용 함수 설계 필요. |
| **확장 백본 탐색** | 다양한 DiT 백본의 효과를 연구하며, 향후 더 강력한 기반 모델(예: 차세대 Qwen, FLUX 대형 버전)이 출시될 때 레시피 재적용 가능성이 높음. |
| **iREPA 레이어 선택** | iREPA를 어느 DiT 레이어에 적용하느냐가 성능에 영향을 미치므로, 레이어 민감도 분석이 추가적으로 필요. |

---

### 추가 후속 연구 방향 제안

1. **비디오 Marigold V3:** SVD(Stable Video Diffusion) 등 비디오 생성 DiT를 기반으로 시간적 일관성 있는 깊이 추정으로 확장.
2. **능동 학습(Active Learning) 통합:** 소량의 실제 도메인 레이블 데이터를 효율적으로 활용하여 합성→실제 도메인 갭을 줄이는 연구.
3. **SinkLoss의 이론적 분석:** 최적 수송 관점에서 SinkLoss가 엣지 품질을 향상시키는 이론적 근거 및 최적 블록 크기 결정 이론 수립.
4. **의료 영상 응용:** 복강경·내시경 영상에서의 깊이 추정에 Marigold V2 레시피를 적용, 합성 의료 데이터(예: 시뮬레이터 기반 GT)를 활용하는 연구.
5. **불확실성 정량화:** 각 픽셀에서의 깊이 예측 불확실성을 추정하여 자율주행·로봇의 안전 임계 응용에 신뢰성 있는 깊이를 제공하는 연구.

---

## 📚 참고 자료 (출처)

1. **arXiv 원문:** Pavlovic et al., "Marigold V2: Revisiting Diffusion Transformers for Monocular Depth Estimation," arXiv:2609.08084, 2026. https://arxiv.org/abs/2609.08084
2. **arXiv HTML 전문:** https://arxiv.org/html/2609.08084v1
3. **Hugging Face Papers 페이지:** https://huggingface.co/papers/2609.08084
4. **GitHub 공식 저장소:** https://github.com/huawei-bayerlab/marigold-v2
5. **Hugging Face 모델 허브:** https://huggingface.co/huawei-bayerlab/marigold-v2-0
6. **Emergent Mind 분석:** https://www.emergentmind.com/papers/2609.08084
7. **rcap.io 분석:** https://rcap.io/posts/marigold-v2-revisiting-diffusion-transformers-for-monocular-2026-09-08
8. **AI Weekly 분석:** https://aiweekly.co/alerts/marigold-v2-cuts-kitti-eth3d-absrel-16-26-with-qlora-tuned-dit
9. **Marigold V1 (원조):** Ke et al., "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation," arXiv:2312.02145, CVPR 2024. https://arxiv.org/abs/2312.02145
10. **Marigold-DC:** "Zero-Shot Monocular Depth Completion with Guided Diffusion," arXiv:2412.13389. https://arxiv.org/pdf/2412.13389
11. **데모 공간:** https://huggingface.co/spaces/huawei-bayerlab/marigold-v2-web

> ⚠️ **정확도 고지:** 본 분석은 공개된 arXiv 논문 HTML 전문, GitHub 저장소, Hugging Face 페이지, 및 관련 분석 사이트를 기반으로 작성되었습니다. 일부 수식(특히 SinkLoss의 세부 수식과 Stage 1 손실 가중치)은 논문 원문 PDF의 정확한 표기를 완전히 확인하지 못한 부분이 있을 수 있으므로, 연구 목적의 정밀 인용 시 원문 PDF를 직접 확인하시기 바랍니다.
