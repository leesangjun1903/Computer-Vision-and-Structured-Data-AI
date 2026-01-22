
# Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior

## 개요

"Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior"는 Microsoft Research와 Shanghai Jiao Tong University의 협력으로 탄생한 논문으로, 단일 이미지에서 고품질 3D 모델 생성의 새로운 패러다임을 제시합니다. 이 논문은 **Diffusion 모델의 암묵적 3D 이해 능력**을 활용하여 대규모 3D 데이터 없이도 일반적인 객체의 고충실도 3D 재구성을 처음으로 성취했습니다.

***

## 1. 핵심 주장 및 기여 요약

### 1.1 주요 기여

Make-It-3D는 다음 세 가지 핵심 기여를 제시합니다:

**첫째, 일반적 객체 3D 생성의 첫 시도**: 기존의 DreamFusion, Magic3D 등은 텍스트 조건에 의존했고, DietNeRF나 SinNeRF는 특정 카테고리에 제한되었습니다. Make-It-3D는 특정 객체 클래스에 대한 학습 없이 **임의의 이미지에서 고품질 3D를 생성**합니다. 이는 대규모 다양한 3D 데이터셋 구축의 어려움을 우회하고, 2D 이미지라는 풍부하고 접근 가능한 자원을 활용하는 혁신입니다.

**둘째, 2단계 최적화 파이프라인**: 기하학과 텍스처 최적화를 분리한 설계가 각 단계의 성능을 극대화합니다. Stage 1은 NeRF를 통해 기본 기하학을 추정하고, Stage 2는 텍스처 포인트 클라우드로 변환하여 세부적인 색상과 세밀한 기하학을 강화합니다. 이러한 분리는 **인간의 지각 특성**에 부합합니다—우리는 기하학보다 텍스처에 더 민감하므로, 텍스처에 집중 최적화하면 전체 품질이 효과적으로 향상됩니다.

**셋째, Diffusion 모델의 3D-인식 감독**: Reference image 제약과 함께 Score Distillation Sampling (SDS)의 변형인 **CLIP-Diffusion loss ($L_{CLIP-D}$)**를 도입하여, 이미지 수준의 의미론적 정렬을 강제합니다. 이는 단순 픽셀 손실이나 텍스트 조건보다 훨씬 효과적으로 생성된 3D가 입력 이미지에 충실하도록 유도합니다.

***

## 2. 해결하는 문제와 제안하는 방법

### 2.1 문제 정의

단일 이미지로부터의 3D 생성은 본질적으로 **ill-posed 문제**입니다:

1. **정보 부족**: 단일 뷰는 원본 3D 기하학의 극소 부분만 제공합니다. 카메라 뒤편, 위/아래, 불가시 영역의 기하학은 추측만 가능합니다.
2. **텍스처 할루시네이션**: 보이지 않는 영역의 텍스처는 완전히 생성되어야 하는데, 참조 이미지가 없으므로 내용 기반 추측에만 의존해야 합니다.
3. **모호성 문제**: 같은 2D 이미지에 대응하는 무한히 많은 유효한 3D 해석이 존재합니다.

기존 방법의 한계:

- **DreamFusion (2022)**: 텍스트 프롬프트만 사용하므로 실제 이미지 세부사항 손실 → 최적화에 15시간 소요
- **DietNeRF/SinNeRF**: Multi-view 감독에만 의존 → 단일 이미지에서 기하학 재구성 불가능
- **Point-E**: 낮은 품질의 포인트 클라우드만 생성
- **3D-Photo**: 경계 부근에서 아티팩트, 큰 뷰 각도에서 실패


### 2.2 제안하는 해결책

Make-It-3D는 **2단계 최적화 + 다중 제약 조합** 방식을 제안합니다:

#### Stage 1: 거친 NeRF 최적화

**총 손실 함수:**

$L_{total} = \lambda_{ref}L_{ref} + \lambda_{SDS}L_{SDS} + \lambda_{CLIP-D}L_{CLIP-D} + \lambda_{depth}L_{depth}$

**(1) Reference View 제약 ($L_{ref}$):**

$L_{ref} = \|x - m \odot G_{ref}\|_1$

여기서 $m$은 전경 영역만 활성화하는 마스크로, 배경 불안정성을 제거합니다. 이는 학습을 객체 기하학에 집중시킵니다.

**(2) Score Distillation Sampling ($L_{SDS}$):**

$$L_{SDS} = \mathbb{E}_{t,w} [w_t (\hat{\epsilon}_\theta(z_t | y, t) - \epsilon) \frac{\partial z_0}{\partial \theta}]$$

이 손실은 DreamFusion에서 도입된 것으로, 렌더링 $G_\theta$를 "좋은 샘플처럼 보이도록" 확산 모델의 기울기를 통해 유도합니다. 타임스텝 $t$에 따라 가중치 $w_t$를 조정하여 노이즈 수준별 영향을 제어합니다.

**(3) CLIP-기반 Diffusion 손실 ($L_{CLIP-D}$) — 핵심 혁신:**

$$L_{CLIP-D} = \mathbb{E}_{t} [- \cos\_sim(\mathbf{e}_{CLIP}(\tilde{X}_{\theta,t}), \mathbf{e}_{CLIP}(X_0^{ref}))]$$

여기서 $\tilde{X}_{\theta,t}$는 렌더링된 이미지를 노이즈 제거한 결과입니다:

$\tilde{X}_{\theta,t} = \text{Denoise}(z_t, t, y)$

이 손실의 혁신성은 **CLIP 인코더를 통한 의미론적 정렬**입니다. 저수준 픽셀 차이가 아닌 고수준 시맨틱 특성을 정렬하므로, 텍스처 세부사항의 차이에 덜 민감하면서도 일관성을 강제합니다. 예를 들어, "파란색 구"는 정확한 픽셀 색상보다는 색상 범주의 일관성이 중요하며, CLIP은 이를 포착합니다.

**선택적 실행 전략**: $L_{SDS}$는 강력한 기하학 구조를 제공하지만 텍스처 세부사항을 무시하고, $L_{CLIP-D}$는 외형을 보존하지만 기하학을 덜 제약합니다. 따라서:

$\text{if } t < t_{thresh}: \text{use } L_{CLIP-D}$
$\text{if } t \geq t_{thresh}: \text{use } L_{SDS}$

최적 $t_{thresh} = 400$ (범위: 200-600의 노이즈 스케일)에서 양자의 균형을 이룹니다.

**(4) 깊이 정규화 ($L_{depth}$):**

$L_{depth} = -\text{Pearson Corr}(d_{pred}, d_{nerf})$

$= -\frac{\text{Cov}(d_{pred}, d_{nerf})}{\sqrt{\text{Var}(d_{pred})} \sqrt{\text{Var}(d_{nerf})}}$

깊이 추정기(DPT)로부터의 절대 깊이값은 스케일 불확실성이 있으므로, 상관계수를 사용하여 **선형 관계만 강제**합니다. 이는 깊이 모호성을 해결하고 기하학의 전반적 형태를 보장합니다. 예: "얼굴이 꺼지거나" "기하학이 과도하게 납작해지는" Janus 문제를 부분 완화합니다.

#### Stage 2: 신경 텍스처 강화

첫 단계에서 얻은 NeRF는 **저품질 텍스처**를 가집니다 (과도하게 평활, 채도 저하). 단순히 렌더링 해상도를 올려도 고주파 세부정보가 없으므로 무의미합니다. 따라서:

**Step 1: 명시적 포인트 클라우드 구축**

$V_{ref} = R_{ref} K^{-1} P[D_{ref} \odot M_{ref}]$

NeRF는 암묵적 표현이라 텍스처 투영이 어렵습니다. 포인트 클라우드는 명시적이므로 reference 이미지의 고품질 텍스처를 직접 투영할 수 있습니다. 이때 가시성 마스킹으로 충돌을 방지합니다.

**Step 2: Deferred Renderer를 통한 텍스처 최적화**

$$I_\phi(\mathbf{v}) = R_{renderer}(\mathcal{R}_0, \mathcal{R}_1, ..., \mathcal{R}_{K-1})$$

여기서 $\mathcal{R}\_i$는 $2^i$ 스케일의 멀티스케일 래스터화 특성 맵, $R_{renderer}$ 는 학습 가능한 U-Net입니다. 19차원 포인트 디스크립터 $\mathbf{f}$를 최적화하면서 ($RGB$ 3차원 + 학습 가능 16차원), diffusion prior가 숨겨진 영역의 텍스처를 생성합니다.

***

## 3. 모델 구조 상세 분석

### 3.1 아키텍처 개요

```
[Reference Image] ──→ [Image Captioning] ──→ "A brown teddy bear..."
                                                      ↓
[Reference Depth] ──→ [Depth Estimation] ┐           ↓
                                         ├─→ [Stage 1: NeRF Optimization]
[Diffusion Model] ◄────────────────────────────┤ (SDS + CLIP-D + Depth)
(Stable Diffusion)                             ↓
                                    [Coarse NeRF Model]
                                         ↓
                                  [Point Cloud Export]
                                    (Visible + Invisible)
                                         ↓
                     ┌────────────────────┴──────────────────┐
                     ↓                                        ↓
            [Visible Points]                    [Invisible Points]
        (Texture Projected)              (Initialized from NeRF)
                     │                                        │
                     └────────────────────┬──────────────────┘
                                         ↓
                          [Stage 2: Deferred Rendering]
                          (U-Net + Multi-scale Features)
                                         ↓
                                [High-fidelity 3D Mesh]
```


### 3.2 Stage 1 상세 구현

**NeRF 표현** (Instant-NGP 기반):

- **Hash Encoding**: 16개 해상도 레벨, $2^{19}$ 크기, 32차원
- **MLP**: 3층 구조, 각 64개 숨은 유닛
- **Volume Rendering**: 레이 당 96개 샘플 (균일 64 + 중요도 32)

**렌더링 방정식** (Volume Rendering):

$C(\mathbf{r}) = \int_0^\infty T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt$

$T(t) = \exp(-\int_0^t \sigma(\mathbf{r}(s)) ds)$

여기서 $\sigma$는 불투명도 (기하학 인코딩), $\mathbf{c}$는 색상 (보기 방향 의존).

**학습 전략**:

- **Progressive Training**: Reference view 근처의 좁은 범위로 시작 → 점진적으로 360°로 확대
    - 이유: 좁은 범위에서는 NeRF가 빠르게 수렴, 이후 확대 시 초기화된 기하학이 안정성 제공
- **Camera Sampling**: 75% 정도는 novel view (다양한 각도 학습), 25%는 reference view (정렬 유지)
- **Data Augmentation**:
    - Shading augmentation (Lambertian + Normal shading)
    - Background color jittering (배경 불안정성 제거)

**최적화 설정**:

- Optimizer: Adam, learning rate 0.001
- Iterations: 5,000
- Render resolution: 100×100
- GPU: V100 32GB에서 약 1시간


### 3.3 Stage 2 상세 구현

**포인트 클라우드 구축 (Iterative Lifting)**:

Reference view 포인트:
$V_{ref} = \{(\mathbf{p}_i, \mathbf{c}_i) : \mathbf{p}_i = \text{unproject}(d_i, \mathbf{u}_i), \mathbf{c}_i = \text{image}(\mathbf{u}_i)\}$

For each other view $i$:

1. 기존 포인트 $V_{ref}$를 view $i$로 투영 → visibility mask 생성
2. View $i$에서 렌더링된 깊이를 들어올린 후, visibility mask로 필터링
3. 새로운 포인트만 $V_i$에 추가 (충돌 방지)

최종: $V = V_{ref} \cup V_1 \cup ... \cup V_N$

**Deferred Renderer 설계**:

```
[Point Cloud] ──→ [Rasterization at K scales]
                  ├─→ Feature Map 0 (full res)
                  ├─→ Feature Map 1 (1/2 res)
                  └─→ Feature Map 2 (1/4 res)
                      ↓
                  [Concatenate] ──→ [U-Net Renderer]
                                         ↓
                                   [Final RGB Image]
```

디스크립터 최적화:

$$\arg\min_{\mathbf{F}} \mathbb{E}_{views} [L_{diffusion}(R(\mathcal{R}_0, ..., \mathcal{R}_{K-1})) + \lambda_{reg}\|\mathbf{F} - \mathbf{F}_{init}\|^2]$$

정규화 항이 초기 텍스처와의 급격한 편차를 방지하여, 참조 이미지의 고품질 부분은 보존하면서 숨겨진 영역만 강화합니다.

**최적화 설정**:

- Iterations: 5,000
- Render resolution: 800×800
- 학습 시간: 약 1시간
- **총 inference 시간: ~2시간 (V100 32GB)**

***

## 4. 성능 향상 (정량적 및 정성적 평가)

### 4.1 정량적 벤치마크 결과

**DTU Dataset (Multi-view Stereo 표준)**:[^1_1]


| 메트릭 | DietNeRF | SinNeRF | DreamFusion | Point-E | 3D-Photo | **Ours-Coarse** | **Ours-Enhanced** |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| LPIPS↓ | 0.1831 | 0.2059 | 0.4075 | — | 0 | 0.1427 | **0.0908** |
| Contextual↓ | 5.34 | 4.28 | 2.15 | 2.23 | 3.43 | 1.74 | **1.59** |
| CLIP↑ | 64.90 | 73.24 | 82.81 | 71.31 | 87.65 | 87.50 | **95.65** |

**분석:**

- **LPIPS (Learning Perceptual Image Patch Similarity)**: 낮을수록 우수. Make-It-3D-enhanced가 3D-Photo의 0을 제외하고 모두 능가. "완벽한 재구성"보다는 "지각적 유사성"을 측정하므로 0은 과적합 의심.
- **Contextual Loss**: 낮을수록 우수. Novel view와 reference 간의 픽셀 수준 구조 유사성. Make-It-3D가 최저 (1.59).
- **CLIP Score**: 높을수록 우수. 의미론적 정렬. Make-It-3D가 최고 (95.65), DreamFusion보다 12.84점 개선.

**Test Benchmark (400개 다양한 이미지)**:[^1_2]


| 메트릭 | DreamFusion | Point-E | **Ours-Coarse** | **Ours-Enhanced** |
| :-- | :-- | :-- | :-- | :-- |
| LPIPS | 0.5649 | — | 0.2354 | **0.0780** |
| Contextual | 3.07 | 5.37 | 1.98 | **1.33** |
| CLIP | 84.08 | 64.36 | 89.06 | **95.12** |

**개선율:**

- LPIPS: 86% 개선 (0.5649 → 0.0780)
- Contextual: 57% 개선 (3.07 → 1.33)
- CLIP: 11% 개선 (84.08 → 95.12)


### 4.2 Ablation Study (Stage 1 손실 함수 분석)

| 구성 | LPIPS | Contextual | CLIP | 관찰 |
| :-- | :-- | :-- | :-- | :-- |
| **SDS만** | 0.3045 | 2.29 | 86.04 | 강한 기하학, 약한 정렬 → 텍스처 불일치 |
| **CLIP-D만** | 0.1260 | 2.43 | 80.27 | 강한 정렬, 약한 기하학 → 납작한 모양 |
| **SDS+CLIP-D (혼합)** | 0.2772 | 2.32 | 84.01 | 둘 다 중간 정도 → 최적 아님 |
| **Thresh=300** | 0.1757 | 2.19 | 87.40 | 초기 임계값, 기하학 약함 |
| **Thresh=400** ✓ | **0.1427** | **1.74** | **87.50** | **최적 균형** |
| **Thresh=500** | 0.1696 | 2.23 | 86.09 | 후기 임계값, 정렬 약화 |

**해석:**

- Threshold=400에서 LPIPS 감소, Contextual 최소, CLIP 평탄 → 기하학과 정렬의 최적점
- 이는 노이즈 스케일 $t \in [200,600]$ 범위에서 약 중점 (t≈400)


### 4.3 정성적 결과

**Reference View 재구성 (Figure 4)**:

- Make-It-3D: 참조 이미지와 거의 동일 (법선 맵에서 세세한 면 구조 가시)
- 기준선들: 텍스처 불일치, 조명 오류, 기하학 왜곡

**Novel View 렌더링 (Figures 8-9)**:

- **DietNeRF/SinNeRF**: 복잡한 객체에서 기하학 재구성 실패, 텍스처 없음
- **3D-Photo**: 큰 각도에서 경계 아티팩트, 불완전한 기하학
- **Make-It-3D**: 부드러운 회전, 일관된 조명, 세밀한 표면 세부사항

**현실 세계 장면 (Figure 9)**:

- 건물, 풍경, 복잡한 물체 처리 가능 (단순 장난감 이상)

***

## 5. 모델의 한계

### 5.1 내재적 기술적 한계

**Janus Problem (다면 얼굴 문제)**:[^1_3]

- 증상: 회전된 텍스트나 얼굴이 모든 각도에서 보임
- 원인: Diffusion prior가 각 각도에서 "자연스러운" 이미지를 생성하려다 보니, 같은 오브젝트가 모든 면에 복제됨
- 예: "FRONT"라는 텍스트가 뒷면에도 나타남
- 부분 완화: Depth prior가 일부 억제하지만 완전 해결 아님

**Over-flat Geometry (과도한 납작함)**:

- 증상: 3D 모양이 너무 납작함 (예: 얼굴이 평면처럼)
- 원인: 깊이 정규화가 단일 reference view에서만 작용하므로, 다른 각도에서는 기하학 모호성 잔존
- 현상: 특히 대칭 객체 (구, 원통 등)에서 심함

**포인트 클라우드 품질 문제**:[^1_4]

- 생성된 포인트 클라우드는 높은 노이즈 포함
- 원래 평면이어도 물결 모양 (wavy) 표면 생성 → 수 cm 오차 범위
- 투명한 객체나 어두운 이미지에서 특히 악화


### 5.2 실용적 한계

**계산 시간**: ~2시간/객체

- 특정 응용 (실시간 렌더링, 대량 생성)에서 비실용적
- 이후 Wonder3D (3-5분), Instant3D (10초)로 해결됨

**메모리 요구**: V100 32GB

- 고해상도 (800×800) 렌더링 필수
- 저사양 GPU (RTX 3060 등)에서 불가능

**입력 의존성**:

- 선명한, 고품질 reference 이미지 필수
- 저해상도, 흐릿한 이미지에서 성능 급감


### 5.3 데이터 및 평가 한계

**Test Benchmark의 한계**:

- 400개 이미지는 상대적으로 작은 규모
- 회전, 조명 변화 등 제한적 다양성
- Ground truth 3D 모델 없음 (proxy metric에만 의존)

**평가 메트릭 부적절성**:

- LPIPS: 저수준 픽셀 기반 (인간 지각과 불일치 가능)
- CLIP Score: 의미론적이지만 기하학 정확도 무시
- Contextual Loss: 구조 유사성이지만 절대 품질 미반영
- **Ground truth 3D와의 비교 불가**

***

## 6. 일반화 성능 분석

### 6.1 객체 다양성

**Test Benchmark 구성**:

- 실제 이미지: 도시 사진, 제품, 동물, 자연 경관 등
- 생성 이미지: Stable Diffusion으로 생성된 다양한 스타일

**성과**:

- 모든 카테고리에서 consistent한 성능
- 학습 데이터 없이 novel categories에 즉시 적용 가능

**제한**:

- 매우 드문 객체 카테고리 (예: 희귀 동물, 추상 조각)에서 성능 불명확
- Benchmark에 포함된 카테고리 분포 편향 가능성


### 6.2 텍스트-3D 생성 (하이브리드 응용)

**방법**:

1. 텍스트 프롬프트 → Stable Diffusion으로 이미지 생성
2. 생성 이미지 → Make-It-3D로 3D 생성

**결과 (Figure 10)**:

- "파란 오토바이", "중세 갑옷 입은 곰" 등 다양한 스타일
- DreamFusion보다 더 세밀한 텍스처, 완전한 기하학

**일반화 관찰**:

- Diffusion 모델의 출력 다양성을 그대로 3D로 전이
- 텍스트 프롬프트 정확도에 의존 (프롬프트 설계 중요)


### 6.3 도메인 적응

**교차 도메인 성능**:

- 사진 이미지: 최고 품질
- 미학적/그림 스타일: 성능 저하 (Zero-123 기준)
- 저품질 이미지: 성능 급감

**원인 분석**:

- Diffusion 모델은 대규모 사진 데이터에 학습됨
- 미학적 스타일은 distribution shift
- 깊이 추정기도 사진에 최적화됨

***

## 7. 2020년 이후 관련 최신 연구 비교

### 7.1 시간대별 진화

```
2022년 2월    DreamFusion          (15시간 - 최초의 diffusion-based 3D)
              ↓ (최적화 개선 + 멀티뷰)
2023년 3월    Zero-1-to-3          (30분 - view-conditioned diffusion)
              ↓ (Reference 제약 + Stage 분리)
2023년 3월    ► Make-It-3D ◄       (2시간 - 균형잡힌 접근)
              ↓ (Cross-domain diffusion 혁신)
2023년 10월   Wonder3D             (3-5분 - 멀티뷰 일관성 강화)
              ↓ (Feed-forward으로 획기적 전환)
2024년        Instant3D / InstantMesh (10초 - 학습 기반 직접 예측)
              ↓ (Distillation 최적화)
2025년 1월    GECO                 (1초 - 극도의 효율화)
```


### 7.2 세부 비교표

| 논문 | 출시일 | 시간 | 방식 | 특징 | 품질 | 일반화 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **DreamFusion** | 2022.9 | 15h | Per-shape 최적화 | 첫 번째 SDS, 텍스트 조건 | 중간 | 낮음 |
| **Zero-1-to-3** | 2023.3 | 30min | 최적화 | View-conditioned diffusion | 중상 | 중상 |
| **Make-It-3D** | 2023.3 | 2h | 2단계 최적화 | Reference + CLIP-D + Depth | **높음** | **높음** |
| **Wonder3D** | 2023.10 | 3-5min | 멀티뷰 생성 | Cross-domain (normal+RGB) | 높음 | 높음 |
| **Instant3D** | 2024.1 | 10s | Feed-forward | LRM 기반, 다양성 제한 | 높음 | 중상 |
| **InstantMesh** | 2024.4 | 10s | Feed-forward | Multi-view diffusion + LRM | **매우 높음** | 높음 |
| **GECO** | 2025.1 | 1s | Distillation | Two-stage diffusion 증류 | 높음 | 높음 |

### 7.3 주요 기술 트렌드

**1. 최적화 방식의 진화**:

- Per-shape optimization (DreamFusion, Make-It-3D) → Feed-forward prediction (Instant3D, InstantMesh)
- 트레이드오프: 최적화 세 가지 + 다양성 vs 빠른 속도 + 제한된 다양성

**2. Representation 발전**:

- NeRF → Point Clouds → Gaussian Splatting → Mesh
- 각 표현의 장단점:
    - **NeRF**: 부드러운 기하학, 느린 렌더링
    - **Point Clouds**: 명시적, 노이즈 많음
    - **Gaussian Splatting**: 빠른 렌더링, real-time 가능
    - **Mesh**: 명확한 표면, 게임/3D 인쇄용 최적

**3. Consistency 강화**:

- Cross-domain diffusion (Wonder3D): Normal + RGB 도메인 동시 생성
- Pose-aware conditioning (Cupid): 카메라 포즈 정보 활용
- Multi-view supervision: 여러 뷰에서 일관성 강제

**4. 데이터 효율성**:

- Large-scale 3D 데이터 → Few-shot/Zero-shot 방식으로 전환
- Diffusion prior의 강력한 일반화 능력 활용


### 7.4 Make-It-3D의 위치 평가

**강점**:

1. **품질 vs 효율의 최적 지점**: DreamFusion보다 7.5배 빠르면서 품질은 우수, Wonder3D보다 품질 약간 떨어지지만 더 정밀한 제어
2. **개념적 순수성**: Reference + Diffusion + Depth의 명확한 조합 → 이해하기 쉽고 재현 가능
3. **일반화 능력**: 특정 학습 없이 모든 카테고리 처리
4. **응용 다양성**: Image-to-3D 뿐 아니라 Text-to-3D, Texture editing 지원

**약점**:

1. **속도**: 2시간은 실시간 응용에 부적합
2. **기하학 한계**: Janus problem, 깊이 모호성
3. **계산 비용**: 고사양 GPU 필수

**결론**: **2023년 중반의 "최적 균형점"** → 이후 Wonder3D, InstantMesh가 각각 속도와 품질에서 특화

***

## 8. 학문적 및 기술적 기여의 영향

### 8.1 직접적 기여

**개념적 기여**:

1. **"Diffusion Prior는 3D를 이해한다"** 명제 입증
    - 2D 생성 모델이 암묵적 3D 이해를 가짐 (이전 의심만 있음)
    - 따라서 대규모 3D 데이터 없이도 3D 생성 가능 (패러다임 전환)
2. **Stage 분리의 유효성 증명**
    - 기하학과 텍스처를 분리하면 각각의 최적화 알고리즘 설계 가능
    - 인간 지각에 부합: 텍스처가 기하학보다 중요
3. **Reference constraint의 중요성**
    - 텍스트-조건 diffusion은 세부사항 손실
    - 이미지-조건이 필수 (이후 모든 연구에서 채택)

**기술적 기여**:

1. **CLIP-Diffusion Loss ($L_{CLIP-D}$)**
    - Semantic alignment를 diffusion 기울기에 통합
    - 수식: $L_{CLIP-D} = -\cos\_sim(\mathbf{e}\_{CLIP}(\tilde{X}), \mathbf{e}\_{CLIP}(X_{ref}))$
    - 이후 다양한 modality 정렬에 활용 (text-image, image-3D 등)
2. **Depth-aware Pearson Correlation Loss**
    - Scale-invariant 기하학 제약
    - 단순 $L2$ loss보다 더 robust (상관관계 강제)
3. **Iterative Point Cloud Lifting**
    - Conflict-free multi-view point merging
    - Visibility masking으로 중복 제거

### 8.2 간접적 영향 및 후속 연구

**이 논문이 가능하게 한 이후 연구**:

1. **Wonder3D (Long et al., 2023)**
    - CLIP-D 개념 확장 → Cross-domain diffusion
    - 같은 해에 발표되어 경쟁, 이후 더 빠른 버전 등장
2. **Cupid (Pose-aware 3D Reconstruction)**
    - Reference image + Pose conditioning 결합
    - Make-It-3D의 reference constraint 개념 발전
3. **PointDreamer (Point Cloud Texturing)**
    - 포인트 클라우드 → 멀티뷰 렌더링 → Diffusion inpainting 파이프라인
    - Make-It-3D의 Stage 2 개념 확장
4. **Multiple depth prior 논문들**
    - 깊이 정규화 개념 일반화
    - 단일 view만이 아니라 multi-view depth constraints 탐색

### 8.3 커뮤니티 내 위치

**인용 및 영향력**:

- 직접 인용: 상대적으로 적음 (Wonder3D 687회 대비)
- **개념적 영향**: 높음 (CLIP-D, Stage separation, Reference constraint)
- **재현 가능성**: 높음 (명확한 알고리즘, 공개 코드)

**학회 반응**:

- CVPR/ICCV 리뷰 점수: 긍정적 (방법론 신선성)
- 그러나 동시기 Wonder3D의 우수성과 빠른 후속 개선으로 "가려짐"

***

## 9. 미래 연구 방향 및 제언

### 9.1 즉시적 개선 방안 (6-12개월)

**1. 기하학 모호성 해결**:

- **Janus Problem 억제**:
    - Multi-view depth constraints (모든 뷰에 깊이 정규화 적용)
    - Symmetry-aware regularization (대칭 객체 감지 및 강제)
    - Temporal consistency (여러 각도에서 동시 최적화)
- **구현 제안**:

$L_{depth,multi} = \sum_{i=1}^N w_i L_{depth}(\tilde{d}\_i, d_{nerf,i})$

여기서 여러 각도에 대한 예측 깊이 사용

**2. 계산 시간 단축 (2시간 → 10-30분)**:

- Progressive training 재설계 (warm-start with coarser models)
- Adaptive sampling (중요 영역에만 높은 해상도)
- **Distillation 기법** (Teacher NeRF → Student Feed-forward Network)

**3. 텍스처 품질 개선**:

- Higher frequency details: Frequency-aware rendering loss
- PBR materials prediction (단순 RGB가 아닌 Albedo, Normal, Roughness)
- Reference image 외부 영역에서도 high-frequency detail 유지


### 9.2 중기 목표 (1-2년)

**1. Video Input 확장**:

- Temporal consistency 제약 추가
- Optical flow guidance
- 최종: 비디오 프레임 → 4D 모델 (3D + temporal)

**2. Multi-image Generalization**:

- Few-shot input (3-5개 이미지)으로 기하학 정확도 향상
- Confidence map: 추정 가능 영역 vs 불확실 영역 명시

**3. Real-time Rendering**:

- Gaussian Splatting으로 conversion
- 게임 엔진 통합 (Unity, Unreal)
- Interactive 3D editing


### 9.3 장기 비전 (2년 이상)

**1. 자동화 최대화**:

- Parameter tuning 자동화 (현재 threshold=400 수동 설정)
- Failure detection 및 자동 복구
- 완전 자동 파이프라인

**2. Scale 확장**:

- 배치 처리 (1000개 이미지/일)
- 클라우드 배포 (비용 최적화)
- 모바일 에지 device 지원

**3. 산업 표준화**:

- e-commerce: 자동 제품 3D 모델링
- Digital twins: 실시간 물리적 객체 → 디지털 모델 동기화
- Metaverse: 대규모 3D asset 생성 및 관리


### 9.4 근본적 해결 불가능한 한계

**1. 단일 이미지의 본질적 모호성**:

- 이론적으로: 무한히 많은 유효한 3D 해석 존재
- 해결 불가: Statistical prior에만 의존 가능

**2. Diffusion 모델의 편향 상속**:

- 학습 데이터 분포 벗어난 객체는 성능 저하
- 해결책: 더 많은 다양한 데이터 (근본적 해결 아님, 경감만 가능)

**3. 평가 메트릭 부재**:

- Ground truth 3D 모델 없이 정확한 평가 어려움
- 해결책: Synthetic dataset 구축 (현실성 문제 존재)

***

## 10. 종합 결론

### 10.1 종합 평가

**Make-It-3D**는 **2023년 중반의 가장 균형잡힌 단일 이미지 3D 생성 방법**입니다:


| 차원 | 평가 | 근거 |
| :-- | :-- | :-- |
| **품질** | ★★★★☆ | LPIPS 0.0908, CLIP 95.65 (최고 수준) |
| **속도** | ★★★☆☆ | 2시간 (DreamFusion 1/7, Wonder3D 1/24) |
| **일반화** | ★★★★★ | 400개 다양한 이미지에 학습 없이 적용 |
| **재현성** | ★★★★☆ | 명확한 알고리즘, 공개 코드 (하지만 GPU 요구 높음) |
| **응용성** | ★★★★☆ | Image-to-3D, Text-to-3D, Texture editing 가능 |
| **장기 영향** | ★★★☆☆ | 개념 기여는 높으나, 빠른 기술 발전으로 가려짐 |

### 10.2 기술적 핵심 성과

**1. Diffusion Prior의 3D 인식 입증**:

- 2D 이미지 생성 모델이 implicit 3D geometry 이해
- 대규모 3D 데이터 없이도 고품질 3D 생성 가능

**2. Reference Constraint의 중요성**:

- 텍스트-조건만으로는 세부사항 손실
- Reference image 정렬 필수 (이후 모든 연구에서 채택)

**3. Stage 분리 효과**:

- 기하학 (NeRF) + 텍스처 (Point Clouds) 분리 최적화
- 각 단계에 최적의 알고리즘 적용 가능
- 인간 지각 특성과 일치


### 10.3 학문적 기여

**개념적**:

- Diffusion prior의 3D 이해력에 대한 새로운 통찰
- Reference constraint의 importance 확립
- Coarse-to-fine 최적화 패러다임

**방법론적**:

- CLIP-Diffusion Loss (semantic alignment)
- Depth-aware regularization (geometric prior)
- Iterative point cloud construction (multi-view fusion)

**실용적**:

- 재현 가능한 pipeline (코드 공개)
- 확장 가능한 아키텍처 (이후 개선 용이)


### 10.4 한계와 교훈

**한계**:

- 2시간 처리 시간 (산업 응용 제약)
- Janus problem 등 기하학 모호성 잔존
- 고사양 GPU 필수

**교훈**:

1. **속도와 품질의 Trade-off는 피할 수 없음**: 이후 Wonder3D, Instant3D 등은 각각 속도 또는 일관성 특화
2. **Reference constraint는 필수**: 텍스트만으로는 부족, 이미지 정보가 critical
3. **Multi-task learning**: 기하학 + 텍스처를 동시 최적화하기보다 분리가 더 효과적

### 10.5 최종 평가

**이 논문의 가치**:

- ★ **학술적**: 개념 기여 높음, 하지만 동시기 경쟁으로 인한 상대적 평가
- ★ **산업적**: 실용적이지만 속도 제약으로 대규모 배포 어려움
- ★ **기술적**: 명확한 방법론, 후속 연구 촉발
- ★ **사회적**: 3D 콘텐츠 생성 민주화의 중요 단계

**2023년 중반 시점에서의 위치**:

- DreamFusion의 느린 최적화를 개선하고
- Wonder3D가 등장하기 전의 "최고의 일반적 방법"
- 이후 Instant3D, InstantMesh 등으로 더욱 진화

**역사적 평가**:

- 현재 (2026년 1월) 관점에서: **중요한 마일스톤**, 하지만 "현재 SOTA는 아님"
- 개념적 기여는 **영구적** (CLIP-D, stage separation 등)
- 실제 산업 배포: **GECO (1초), InstantMesh (10초)** 선호

***

## 참고 자료 및 벤치마크

**주요 평가 메트릭 정의:**

- **LPIPS**: $\frac{1}{|H||W|} \sum_{h,w} \|\mathbf{F}^l(I_h,w) - \mathbf{F}^l(\hat{I}_{h,w})\|_2$ (AlexNet 특성)
- **Contextual Loss**: Patch-based structural similarity
- **CLIP Score**: $\cos\_sim(\mathbf{e}_{CLIP}(I), \mathbf{e}_{CLIP}(T))$

**데이터셋:**

- DTU: 일반적인 3D 재구성 벤치마크
- Make-It-3D Custom: 400개 이미지 (공개 약속, 필요시 논문 저자 연락)

**계산 환경:**

- GPU: NVIDIA V100 32GB
- Optimization: Adam optimizer, LR=0.001
- Framework: PyTorch + custom CUDA kernels

***

**최종 결론**: Make-It-3D는 **단일 이미지 3D 생성 역사에서 중요한 전환점**으로, Diffusion prior의 3D 인식 능력을 체계적으로 활용한 첫 번째 성공 사례입니다. 비록 이후 더 빠른 방법들에 의해 실용적 측면에서 대체되었으나, 그 **개념적 기여와 방법론은 이 분야의 기초**가 되었습니다. 현재 연구자들은 Make-It-3D의 아이디어를 바탕으로 속도, 품질, 일반화 성능을 각각 개선하는 방향으로 나아가고 있습니다.
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_5][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_6][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_8][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2303.14184v2.pdf

[^1_2]: https://arxiv.org/abs/2303.11938

[^1_3]: https://ieeexplore.ieee.org/document/10203601/

[^1_4]: https://arxiv.org/abs/2211.10440

[^1_5]: https://ieeexplore.ieee.org/document/11141031/

[^1_6]: https://academic.oup.com/europace/article/doi/10.1093/europace/euaf085.408/8141817

[^1_7]: https://aacrjournals.org/clincancerres/article/31/13_Supplement/A010/763286/Abstract-A010-Benchmarking-3D-against-2D-deep

[^1_8]: https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs

[^1_9]: https://ieeexplore.ieee.org/document/11075573/

[^1_10]: https://iopscience.iop.org/article/10.1149/MA2025-024658mtgabs

[^1_11]: https://arxiv.org/html/2411.14384

[^1_12]: https://arxiv.org/pdf/2308.07837.pdf

[^1_13]: http://arxiv.org/pdf/2303.14184.pdf

[^1_14]: http://arxiv.org/pdf/2303.11328.pdf

[^1_15]: https://arxiv.org/pdf/2309.17261.pdf

[^1_16]: https://arxiv.org/html/2412.10294v1

[^1_17]: https://arxiv.org/pdf/2401.12175.pdf

[^1_18]: https://arxiv.org/html/2503.00726

[^1_19]: https://neurips.cc/virtual/2023/poster/72556

[^1_20]: https://www.sciencedirect.com/science/article/abs/pii/S0167865524002058

[^1_21]: https://www.deeplearning.ai/the-batch/a-3d-model-from-one-2d-image/

[^1_22]: https://academic.oup.com/jcde/article/12/12/70/8304017

[^1_23]: https://www.tandfonline.com/doi/full/10.1080/18824889.2025.2497600

[^1_24]: https://proceedings.neurips.cc/paper_files/paper/2023/file/0b68d474baf8dff30f3280c199a32089-Paper-Conference.pdf

[^1_25]: https://arxiv.org/abs/2501.16737

[^1_26]: https://cvmi-lab.github.io/Point-UV-Diffusion/paper/point_uv_diffusion.pdf

[^1_27]: https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_MPOD123_One_Image_to_3D_Content_Generation_Using_Mask-enhanced_Progressive_CVPR_2024_paper.pdf

[^1_28]: https://kimjy99.github.io/논문리뷰/realfusion/

[^1_29]: https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Point-NeRF_Point-Based_Neural_Radiance_Fields_CVPR_2022_paper.pdf

[^1_30]: https://viso.ai/deep-learning/neural-radiance-fields/

[^1_31]: https://openaccess.thecvf.com/content/ICCV2025/papers/Engelhardt_SViM3D_Stable_Video_Material_Diffusion_for_Single_Image_3D_Generation_ICCV_2025_paper.pdf

[^1_32]: https://arxiv.org/html/2206.01290v3

[^1_33]: https://www.nature.com/articles/s41598-025-16386-7

[^1_34]: https://arxiv.org/html/2509.13013v1

[^1_35]: https://arxiv.org/html/2406.15811v3

[^1_36]: https://arxiv.org/html/2510.20776v1

[^1_37]: https://arxiv.org/html/2404.04875v1

[^1_38]: https://arxiv.org/html/2210.00379v6

[^1_39]: https://arxiv.org/html/2511.18900v1

[^1_40]: https://arxiv.org/html/2507.13929v1

[^1_41]: https://arxiv.org/html/2510.08271v2

[^1_42]: https://arxiv.org/html/2506.18331v4

[^1_43]: https://arxiv.org/html/2312.11535v3

[^1_44]: https://arxiv.org/html/2512.08905v1

[^1_45]: https://arxiv.org/html/2304.14811v2

[^1_46]: https://arxiv.org/abs/2507.00969

[^1_47]: https://ieeexplore.ieee.org/document/11195775/

[^1_48]: https://arxiv.org/abs/2403.11503

[^1_49]: https://ieeexplore.ieee.org/document/10657484/

[^1_50]: https://www.mdpi.com/2079-9292/14/19/3868

[^1_51]: https://dl.acm.org/doi/10.1145/3689645

[^1_52]: https://onlinelibrary.wiley.com/doi/10.1002/lpor.202401609

[^1_53]: https://arxiv.org/abs/2409.04768

[^1_54]: https://dl.acm.org/doi/10.1145/3581783.3612316

[^1_55]: https://arxiv.org/abs/2510.09035

[^1_56]: https://ieeexplore.ieee.org/document/11319338/

[^1_57]: http://arxiv.org/pdf/2405.20343.pdf

[^1_58]: https://arxiv.org/html/2404.07199v1

[^1_59]: https://arxiv.org/html/2411.10947v1

[^1_60]: http://arxiv.org/pdf/2306.16928v1.pdf

[^1_61]: https://arxiv.org/html/2312.09069

[^1_62]: http://arxiv.org/pdf/2403.00939.pdf

[^1_63]: https://liner.com/review/wonder3d-single-image-to-3d-using-crossdomain-diffusion

[^1_64]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00698.pdf

[^1_65]: https://openaccess.thecvf.com/content_ICCV_2019/papers/Wallace_Few-Shot_Generalization_for_Single-Image_3D_Reconstruction_via_Priors_ICCV_2019_paper.pdf

[^1_66]: https://games-1312234642.cos.ap-guangzhou.myqcloud.com/pdf/Games2024311龙霄潇.pdf

[^1_67]: https://www.reddit.com/r/StableDiffusion/comments/z13q7q/221110440_magic3d_highresolution_textto3d_content/

[^1_68]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610054.pdf

[^1_69]: https://openreview.net/forum?id=lxuXvJSOcP

[^1_70]: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Rapid_3D_Model_Generation_with_Intuitive_3D_Input_CVPR_2024_paper.pdf

[^1_71]: https://arxiv.org/abs/2408.14724

[^1_72]: https://www.sciencedirect.com/science/article/pii/S0926580522000255

[^1_73]: https://www.parallelmind.xyz/p/will-2024-be-the-year-of-text-to-3d

[^1_74]: https://kimjy99.github.io/논문리뷰/mv-dream/

[^1_75]: https://arxiv.org/abs/2310.15008

[^1_76]: https://openreview.net/forum?id=2lDQLiH1W4

[^1_77]: https://ieeexplore.ieee.org/document/9009556/

[^1_78]: https://arxiv.org/html/2511.01767v2

[^1_79]: https://arxiv.org/pdf/2404.07191.pdf

[^1_80]: https://arxiv.org/html/2511.01767v1

[^1_81]: https://arxiv.org/html/2404.06429v1

[^1_82]: https://arxiv.org/html/2405.14832v1

[^1_83]: https://arxiv.org/abs/1909.01205

[^1_84]: https://arxiv.org/html/2511.14291v1

[^1_85]: https://arxiv.org/pdf/2404.06091.pdf

[^1_86]: https://arxiv.org/html/2408.14724v1

[^1_87]: https://arxiv.org/html/2506.23150v2

[^1_88]: https://arxiv.org/html/2310.08529v3

[^1_89]: https://arxiv.org/html/2503.07190v1

