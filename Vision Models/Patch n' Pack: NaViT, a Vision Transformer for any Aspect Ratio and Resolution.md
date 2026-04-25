# Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

NaViT(Native Resolution Vision Transformer)는 **고정 해상도로 이미지를 리사이즈하는 관행이 최적이 아님**을 실증적으로 보여주며, Vision Transformer의 시퀀스 기반 모델링 특성을 활용하여 **임의의 해상도 및 종횡비(aspect ratio)를 가진 이미지를 그대로 처리**할 수 있음을 제안합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **Patch n' Pack** | NLP의 example packing을 Vision Transformer에 적용 |
| **학습 효율성** | 동일 컴퓨팅 예산에서 ViT 대비 최대 4× 빠른 학습 |
| **유연한 추론** | 단일 모델로 다양한 해상도에서 비용-성능 트레이드오프 조절 |
| **일반화 성능** | OOD 벤치마크(ImageNet-A, ObjectNet) 성능 개선 |
| **공정성 향상** | 공정성 관련 신호 주석 정확도 향상 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

기존 컴퓨터 비전 모델들은 다음과 같은 문제를 가지고 있었습니다:

1. **고정 해상도 요구**: CNN 기반 구조의 특성상 고정된 배치 크기와 이미지 크기가 필요
2. **종횡비 왜곡**: 정사각형으로 리사이즈 시 원본 정보 손실
3. **학습-추론 불일치**: 특정 해상도에서만 학습하면 다른 해상도에서 성능 저하
4. **비효율적 패딩**: 가변 길이 시퀀스를 처리하기 위해 패딩 사용 시 연산 낭비

논문에서 분석한 실제 데이터셋의 종횡비 현황:
- **ImageNet**: 85.9%가 비정사각형
- **LVIS**: 92.2%가 비정사각형
- **WebLI**: 57.3%가 비정사각형

### 2.2 제안하는 방법

#### 2.2.1 Patch n' Pack 핵심 메커니즘

여러 이미지의 패치를 단일 시퀀스로 패킹합니다:

$$\text{Sequence} = [\underbrace{p_1^{(1)}, \ldots, p_{n_1}^{(1)}}_{\text{Image 1}}, \underbrace{p_1^{(2)}, \ldots, p_{n_2}^{(2)}}_{\text{Image 2}}, \ldots, \underbrace{\text{PAD}}_{\text{패딩}}]$$

#### 2.2.2 마스킹된 Self-Attention

서로 다른 이미지의 패치 간 attention을 차단합니다. Attention mask $M$은:

$$M_{ij} = \begin{cases} 0 & \text{if token } i \text{ and } j \text{ belong to the same image} \\ -\infty & \text{otherwise} \end{cases}$$

Attention 출력:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top + M}{\sqrt{d_k}}\right)V$$

#### 2.2.3 팩토리즈 위치 임베딩 (Factorized Positional Embeddings)

기존 1D 위치 임베딩 대신 $x$, $y$ 좌표를 독립적으로 분리하여 임베딩:

$$\text{pos emb}(x, y) = \phi_x(x) + \phi_y(y)$$

- **절대 좌표(Absolute)**: $\phi(p): [0, \text{maxLen}] \rightarrow \mathbb{R}^D$, 절대 패치 인덱스의 함수
- **분수 좌표(Fractional)**: $\phi(r): [0, 1] \rightarrow \mathbb{R}^D$, where $r = p / \text{side-length}$, 상대적 위치의 함수

분수 좌표의 경우 이미지 크기에 무관하게 임베딩 파라미터를 공유할 수 있어 **미학습 해상도로의 외삽(extrapolation)에 유리**합니다.

#### 2.2.4 연속 토큰 드롭핑 (Continuous Token Dropping)

$n$번째 처리 이미지에 적용되는 토큰 드롭 스케줄:

$$\rho(n; \rho_{\min}, \rho_{\max}, \mu, \tau) = \rho_{\min} + (\rho_{\max} - \rho_{\min}) \cdot \sigma\!\left(\frac{n - \mu}{\tau}\right)$$

여기서 $\sigma$는 sigmoid 함수, $\rho_{\min} = 0.2$, $\rho_{\max} = 0.8$.

베타 분포를 이용한 이미지별 드롭률 샘플링:

$$u \sim \mathcal{B}(\alpha, \beta), \quad d = u \times d_{\max}$$

$$\text{where} \quad u_\mu = \frac{d_\mu}{d_{\max}}, \quad \sigma^2 = 0.3 \cdot u_\mu(1 - u_\mu)$$

#### 2.2.5 해상도 샘플링 전략

각 이미지에 대해 $u \sim \mathcal{D}$를 샘플링하고 $[-1, 1]$에서 $[64, 384]$로 선형 스케일링:

$$R \sim \mathcal{U}(64, R_{\max}) \quad \text{또는} \quad R \sim \mathcal{N}_t(-0.5, 1) \text{(저해상도 편향)}$$

실험 결과, 면적(area) 대신 **side length를 직접 샘플링**하고 **저해상도 방향으로 편향된 truncated normal 분포**를 사용하는 것이 최적.

### 2.3 모델 구조

NaViT는 기본 ViT 구조를 유지하면서 다음 요소를 수정합니다:

```
Input Images (다양한 해상도/종횡비)
        ↓ Patchify (패치 크기 P × P)
        ↓ Token Drop (연속적 드롭률)
 ──────────────────────────────────
| Packed Sequence                  |
| [Img1 patches | Img2 patches |..] |
 ──────────────────────────────────
        ↓ Factorized Positional Embedding
        ↓ Masked Self-Attention (이미지 간 격리)
        ↓ MLP / LayerNorm / Residual (변경 없음)
        ↓ Masked Pooling (이미지별 독립 풀링)
        ↓ Classification Head / Contrastive Loss
```

**추가된 ViT 개선사항:**
- Query-Key Normalization
- Bias 제거
- Attention Pooling

### 2.4 성능 향상

#### 학습 효율

| 모델 | TPU Hours | 학습 이미지 수 |
|------|-----------|---------------|
| ViT-L/16 | $9.8 \times 10^{12}$ | $4.0 \times 10^9$ |
| NaViT-L/16 | $9.8 \times 10^{12}$ | $1.9 \times 10^{10}$ (약 **5배**) |

- 동일 컴퓨팅 예산에서 최고 성능 ViT와 동등한 결과를 **4× 적은 연산**으로 달성

#### 다운스트림 태스크 성능

| 태스크 | ViT | NaViT | 개선 |
|--------|-----|-------|------|
| ImageNet Top-1 (zero-shot) | 68.3% | 72.9% | +4.6%p |
| LVIS AP | 23.3% | 28.3% | +5.0%p |
| LVIS AP rare | 17.2% | 24.3% | +7.1%p |
| ADE20k mIoU (R384) | ~47 | ~51 | +4 |

#### OOD 성능

- **ImageNet-A**: NaViT가 ViT 대비 일관되게 우수 (극단적 종횡비 이미지 포함)
- **ObjectNet**: 비슷한 수준 (aspect-preserving center crop 적용 시)

### 2.5 한계점

1. **Self-Attention의 $O(n^2)$ 비용**: 여러 이미지를 패킹하면 시퀀스가 길어져 메모리 부담 증가 (단, 모델 차원 확장 시 비율 감소)
2. **패딩 토큰**: Greedy 패킹으로도 ~2% 패딩 불가피
3. **Contrastive Loss의 복잡성**: Chunked contrastive loss 등 별도 처리 필요
4. **$E_{\max}$ 설정**: 시퀀스당 최대 이미지 수 결정이 필요하며, 초과 시 이미지 손실
5. **기존 벤치마크와의 불일치**: 고정 해상도 정사각형 벤치마크(224×224)는 NaViT의 유연성을 완전히 평가하지 못함

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 팩토리즈 위치 임베딩의 일반화

기존 방법들과의 비교:

| 임베딩 방식 | 학습 범위 내 정확도 | 고해상도 외삽 |
|-------------|-------------------|--------------|
| Learned 1D (ViT) | 53.8% | 69.3% |
| Learned 2D (Pix2struct) | 63.5% | 71.3% |
| **Factorized (+) Abs.** | **64.9%** | **71.5%** |
| Fourier (fractional) | 63.9% | 70.8% |

Pix2struct의 Learned 2D 임베딩은 학습 중 보지 못한 $(x, y)$ 쌍이 증가함에 따라 **고해상도 외삽에 실패**하는 반면, 팩토리즈 임베딩은 $x$와 $y$를 독립적으로 학습하므로 학습 범위를 벗어난 해상도에서도 강건합니다.

$$\phi_x(x) + \phi_y(y) \quad \text{vs} \quad \phi_{xy}(x, y) \quad \Rightarrow \text{팩토리즈 방식이 외삽에 유리}$$

### 3.2 혼합 해상도 학습을 통한 일반화

**Variable-resolution pre-training** 효과:

$$R \sim \mathcal{U}(64, R_{\max})$$로 학습 시, 고정 해상도 $R = R_{\max}$로 학습한 모델보다 **모든 평가 해상도에서 동등하거나 우수**한 성능을 보임.

이는 모델이 다양한 "시점"에서 이미지를 학습함으로써 **특정 해상도에 과적합되지 않는** 표현을 학습하기 때문입니다.

### 3.3 OOD(Out-of-Distribution) 일반화

**ImageNet-A** (극단적 종횡비 이미지 포함):

$$\text{NaViT의 ImageNet-A 성능} \gg \text{ViT의 ImageNet-A 성능}$$

- NaViT-L/16 vs ViT-L/16 (Resize 전략): NaViT $\approx$ 59.9% vs ViT $\approx$ 38.5% (compute $2.5 \times 10^{12}$)

이는 **종횡비 보존 학습**이 실세계의 다양한 이미지 형태에 대한 강건성을 높임을 의미합니다.

### 3.4 저해상도 파인튜닝 → 고해상도 평가 일반화

NaViT를 저해상도(64)에서 파인튜닝해도 고해상도에서 평가 시 좋은 성능을 유지:

- **NaViT-64:512** (variable resolution fine-tuning): 단일 해상도 NaViT와 동등한 성능
- 비용 절감 + 해상도 유연성의 두 마리 토끼를 동시에

### 3.5 공정성(Fairness) 관련 일반화

FairFace 및 CelebA 데이터셋 실험:
- NaViT가 ViT보다 높은 공정성 신호 주석 정확도 달성 ($p = 3 \times 10^{-4}$, Wilcoxon signed-rank test)
- Native aspect ratio 사용 시 추가 성능 향상 ($p = 0.02$, 95% 신뢰 수준)

### 3.6 모델 캘리브레이션의 안정성

시퀀스 길이 128~1024 범위에서 Expected Calibration Error가 $(0.045, 0.047)$ 구간에서 **매우 안정적**:

| Sequence Length | 128 | 256 | 384 | 512 | 640 | 768 | 1024 |
|----------------|-----|-----|-----|-----|-----|-----|------|
| Calibration Error | 0.047 | 0.046 | 0.048 | 0.047 | 0.047 | 0.046 | 0.045 |

이는 해상도 변화에 관계없이 **모델 불확실성 추정의 일관성**을 의미하며, 실제 응용에서의 신뢰성을 높입니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | 종횡비 보존 | 다중 해상도 | 효율성 |
|------|------|----------|------------|------------|--------|
| **ViT** (Dosovitskiy et al.) | 2021 | 고정 패치 분할 | ✗ | ✗ | 기준 |
| **FlexiViT** (Beyer et al.) | 2023 | 다중 패치 크기 지원 | ✗ | ○ (patch size 변경) | ○ |
| **Pix2struct** (Lee et al.) | 2022 | 2D 위치 임베딩 + AR 보존 | ✓ | ✗ | △ |
| **MAE** (He et al.) | 2022 | 랜덤 마스킹으로 효율화 | ✗ | ✗ | ○ |
| **Swin Transformer V2** (Liu et al.) | 2022 | 계층적 표현, 고해상도 | ✗ | ○ (계층적) | ○ |
| **MViTv2** (Li et al.) | 2022 | 멀티스케일 ViT | ✗ | ✓ (계층적) | ○ |
| **DeiT III** (Touvron et al.) | 2022 | FixRes 전략 활용 | ✗ | △ | ○ |
| **NaViT** (Dehghani et al.) | 2023 | Patch n' Pack + 팩토리즈 임베딩 | ✓ | ✓ | ◎ |

### 주요 차별점 분석

**vs FlexiViT:**
- FlexiViT는 **패치 크기**를 변경하여 해상도를 조절하나, 종횡비는 보존하지 않음
- NaViT는 **원본 종횡비를 보존**하면서 실제 픽셀 수 기반으로 처리

**vs Pix2struct:**
- Pix2struct는 문서/차트 이해 태스크에 특화, 2D 위치 임베딩 사용
- NaViT는 범용 비전 태스크에 적용 가능하며, **미학습 해상도 외삽에 더 강건**

**vs Multigrid Training (Wu et al., 2020):**
- Multigrid는 복잡한 계층적 스케줄 필요
- NaViT는 **단순한 Patch n' Pack으로 동일 효과** 달성

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 데이터 파이프라인의 패러다임 전환
NaViT는 CNN 시대부터 이어진 "모든 이미지를 정사각형으로 리사이즈" 관행에 도전합니다. 향후 연구에서는:
- **Native resolution 처리**가 표준이 될 가능성
- 이미지-텍스트 멀티모달 모델(CLIP 등)의 학습 파이프라인 혁신

#### (2) 대규모 언어-비전 통합 모델 (VLM)에의 적용
NaViT의 Patch n' Pack 아이디어는 이미 후속 멀티모달 연구에 영향을 미치고 있습니다:
- LLaVA, Flamingo 등의 VLM에서 가변 해상도 처리 도입 촉진
- 의료 영상, 위성 이미지 등 **도메인 특화 모델**에서의 종횡비 보존 필요성 강조

#### (3) 적응형 계산(Adaptive Computation) 연구 촉진
고정 배치 형태 제약이 해소됨으로써:
- **이미지 난이도에 따른 동적 토큰 할당** 연구 활성화
- 캐스케이드(cascade) 추론 전략 발전

#### (4) 비디오 처리 연구
NaViT-L이 ViViT-L과 경쟁적 성능을 **약 6배 적은 에폭**으로 달성:
- 시간적 차원을 포함한 **3D Patch n' Pack** 연구 가능성
- 다양한 프레임 레이트/해상도 비디오 처리

#### (5) 효율적 Attention 연구와의 시너지
FlashAttention(Dao et al., 2022) 등 메모리 효율적 attention과 결합:

$$\text{NaViT} + \text{FlashAttention} \Rightarrow \text{긴 시퀀스 처리 효율 극대화}$$

### 5.2 앞으로 연구 시 고려할 점

#### (1) 패킹 효율 최적화
- 현재 Greedy 패킹 방식은 약 2% 패딩 발생
- **Bin packing 알고리즘** 적용 또는 **강화학습 기반 동적 패킹** 탐색 필요
- 해상도 분포 최적화와 패킹 효율의 공동 최적화

#### (2) 대규모 Long-Context 처리
- 여러 이미지를 하나의 시퀀스로 패킹 시 $O(n^2)$ attention 비용 증가
- **Sparse attention**, **Linear attention** 또는 **FlashAttention**과의 결합 필수 고려
- 특히 고해상도 의료 이미지(예: 4096×4096)에서의 적용 가능성

#### (3) 계층적 구조와의 결합
NaViT는 계층적 표현을 생성하지 않으므로, **Swin Transformer** 계열의 계층적 특성맵 생성과 결합 시:
- 객체 탐지, 인스턴스 세그멘테이션에서 추가 성능 향상 가능성
- **FPN(Feature Pyramid Network)** 등과의 통합 방법론 탐색

#### (4) 자기지도 학습(Self-Supervised Learning)과의 결합
- **MAE(Masked Autoencoder)**와 Patch n' Pack 결합 시 재구성 타겟 설계 방법 연구 필요
- 가변 해상도에서의 **Contrastive Learning** 긍정/부정 쌍 구성 전략 최적화

#### (5) 평가 프로토콜 재설계
현재 벤치마크(ImageNet 등)는 고정 해상도 정사각형 이미지 기준:
- **가변 해상도 및 다양한 종횡비를 포함하는 새로운 벤치마크** 설계 필요
- 실제 웹 이미지 분포를 반영한 평가 지표 개발

#### (6) 실제 배포 환경에서의 고려사항
- **동적 배치 크기**: 추론 시 시퀀스 길이가 가변적이므로 하드웨어 최적화 복잡
- **레이턴시 예측 불확실성**: 이미지 해상도에 따른 처리 시간 편차 관리
- **분산 학습 시 로드 밸런싱**: 다양한 시퀀스 길이의 균등한 분산 처리

---

## 참고자료 (출처)

1. **Dehghani et al. (2023)** - "Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution" - arXiv:2307.06304v1 *(본 논문, 제공된 PDF)*

2. **Dosovitskiy et al. (2021)** - "An image is worth 16x16 words: Transformers for image recognition at scale" - ICLR 2021

3. **Beyer et al. (2023)** - "FlexiViT: One model for all patch sizes" - CVPR 2023

4. **Lee et al. (2022)** - "Pix2struct: Screenshot parsing as pretraining for visual language understanding" - arXiv:2210.03347

5. **He et al. (2022)** - "Masked autoencoders are scalable vision learners" - CVPR 2022

6. **Dao et al. (2022)** - "FlashAttention: Fast and memory-efficient exact attention with IO-awareness" - NeurIPS 2022

7. **Liu et al. (2022)** - "Swin Transformer V2: Scaling Up Capacity and Resolution" - CVPR 2022

8. **Li et al. (2022)** - "MViTv2: Improved multiscale vision transformers for classification and detection" - CVPR 2022

9. **Touvron et al. (2022)** - "DeiT III: Revenge of the ViT" - ECCV 2022

10. **Wu et al. (2020)** - "A multigrid method for efficiently training video models" - CVPR 2020

11. **Krell et al. (2021)** - "Efficient sequence packing without cross-contamination: Accelerating large language models without impacting performance" - arXiv:2107.02027

12. **Zhai et al. (2022)** - "Scaling vision transformers" - CVPR 2022

13. **Mustafa et al. (2023)** - "On efficient losses for distributed contrastive learning" - arXiv preprint

14. **Tancik et al. (2020)** - "Fourier features let networks learn high frequency functions in low dimensional domains" - NeurIPS 2020

15. **Geirhos et al. (2021)** - "Partial success in closing the gap between human and machine vision" - NeurIPS 2021
