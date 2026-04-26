# EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

EdgeRunner는 기존 auto-regressive 메시 생성 방법들이 가진 세 가지 핵심 문제—**(1) 적은 face 수 생성 한계, (2) 낮은 해상도, (3) 일반화 성능 부족**—를 동시에 해결하는 새로운 프레임워크를 제안한다.

### 주요 기여 (4가지)

| 기여 | 내용 |
|------|------|
| **Compact Mesh Tokenization** | EdgeBreaker 기반 수정 토크나이저로 50% 압축률 달성 |
| **ArAE (Auto-regressive Auto-encoder)** | 가변 길이 메시 → 고정 길이 잠재 코드 압축 |
| **Latent Diffusion 연계** | ArAE 잠재 공간을 활용한 이미지 조건부 생성 |
| **높은 해상도 및 복잡도** | 최대 4,000 faces, $512^3$ 해상도 지원 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 auto-regressive 메시 생성 방법들(MeshGPT, MeshXL, MeshAnything 등)은 다음 세 가지 한계를 가진다:

1. **비효율적 토크나이제이션**: face당 9개 토큰 사용 → 최대 1,600 faces 한계
2. **낮은 공간 해상도**: $128^3$ 격자 양자화 → 표면이 불규칙하고 정밀도 낮음
3. **모델 일반화 부족**: 단일 뷰 이미지 같은 복잡한 조건에서 학습 외 도메인으로 일반화 실패

---

### 2.2 제안하는 방법

#### 2.2.1 Compact Mesh Tokenization (EdgeBreaker 기반)

반-엣지(half-edge) 자료구조를 사용하여 삼각 메시를 순회하며 1D 토큰 시퀀스로 변환한다.

**핵심 원리: 인접 삼각형 간 엣지 공유(edge sharing)**

- 첫 삼각형: `B v1 v2 v3` (B + 3 vertices = 4 tokens)
- 이후 삼각형: `N v4` 또는 `P v7` (방향 토큰 + 1 vertex = 2 tokens)

이를 통해 face당 평균 **4~5 토큰**을 사용(기존 9 토큰 대비 약 50% 절감).

**어휘(vocabulary) 구성:**

$$\mathcal{V} = \underbrace{512}_{\text{좌표 토큰}} + \underbrace{3}_{\text{face type: N, P, B}} + \underbrace{3}_{\text{special: BOS, EOS, PAD}} = 518$$

**기존 EdgeBreaker와의 차이점:**

- 절대 좌표 사용 (상대 좌표 아님): 양자화 정밀도 유지
- S 토큰 제거 → 장거리 의존성(long-range dependency) 제거
- C 토큰 → L로 통합, E 토큰 → B에 병합
- 결과: L, R(=N, P), B 세 가지 face type만 사용

**압축률 비교 (Table 1):**

| | Compression Ratio ↓ | Sub-sequence Count ↓ | Tokenization Speed ↑ |
|---|---|---|---|
| AMT (MeshAnythingV2) | **46.2%** | 199.5 | 8.6 |
| Ours | 47.4% | **54.7** | **25.2** |

#### 2.2.2 Auto-regressive Auto-encoder (ArAE)

**인코더**: 메시 표면에서 $N$개의 점을 샘플링하여 고정 길이 잠재 코드 추출

$$\mathbf{Z} = \text{CrossAtt}\!\left(\mathbf{Q},\, \text{PosEmbed}(\mathbf{X})\right) $$

여기서:
- $\mathbf{X} \in \mathbb{R}^{N \times 3}$: 메시 표면에서 샘플링한 $N$개 점 ($N = 8192$)
- $\mathbf{Q} \in \mathbb{R}^{M \times C}$: 학습 가능한 쿼리 임베딩
- $\text{PosEmbed}(\cdot)$: 3D 점에 대한 주파수 임베딩 함수
- $\mathbf{Z} \in \mathbb{R}^{M \times L}$: 고정 길이 잠재 코드 ($M=2048, L=64$)
- $M < N$, $L < C$

**디코더**: OPT 아키텍처 기반 auto-regressive transformer (24 self-attention layers, hidden dim 1536, 16 heads)

**손실 함수:**

$$\mathcal{L}_{\text{ce}} = \text{CrossEntropy}\!\left(\hat{\mathbf{S}}[:-1],\, \mathbf{S}[1:]\right) $$

$$\mathcal{L}_{\text{reg}} = \|\mathbf{Z}\|_2^2 $$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}} + \lambda \mathcal{L}_{\text{reg}}$$

- $\mathbf{S}$: 정답 토큰 시퀀스 (one-hot)
- $\hat{\mathbf{S}}$: 예측 분류 로짓 시퀀스
- $\mathcal{L}_{\text{reg}}$: 잠재 공간의 범위를 제한하여 이후 확산 모델 학습을 용이하게 함

**Face Count Conditioning:** 4개의 범위 토큰 (≤1000, 1000~2000, 2000~4000, >4000)과 1개의 unconditional 토큰을 추가하여 생성 face 수를 coarse하게 제어.

#### 2.2.3 Image-conditioned Latent Diffusion (DiT)

고정 길이 잠재 공간 $\mathbf{Z}$를 활용해 DiT(Diffusion Transformer) 기반 확산 모델을 학습.

각 DiT 레이어:

$$\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2 = \text{Chunk}(\mathbf{T} + \mathbf{t}_e) $$

$$\mathbf{x} = \mathbf{x} + \alpha_1 \times \text{SelfAttention}\!\left(\text{LayerNorm}_1(\mathbf{x}) \times (1 + \beta_1) + \gamma_1\right) $$

$$\mathbf{x} = \mathbf{x} + \text{CrossAttention}(\mathbf{x}, \mathbf{c}) $$

$$\mathbf{x} = \mathbf{x} + \alpha_2 \times \text{FeedForward}\!\left(\text{LayerNorm}_2(\mathbf{x}) \times (1 + \beta_2) + \gamma_2\right) $$

- $\mathbf{T}$: 레이어별 학습 가능 스케일-시프트 테이블
- $\mathbf{t}_e$: 타임스텝 특성
- $\mathbf{c}$: CLIP-H(OpenCLIP)에서 추출한 이미지 조건 특성

**학습**: DDPM 프레임워크, MSE 손실, min-SNR 전략(weight=5.0), CFG scale=7.5

---

### 2.3 모델 구조 요약

```
[Point Cloud Input]
       ↓
[Encoder: CrossAtt with learnable Q]
       ↓
[Fixed-length Latent Code Z ∈ R^{2048×64}]
       ↓                        ↓
[Auto-regressive Decoder]   [DiT (Image-conditioned)]
       ↓                        ↓
[Variable-length Mesh Tokens]  [Generated Z from Image]
       ↓
[De-tokenizer → Reconstructed Mesh]
```

**전체 파라미터**: ArAE ≈ 0.7B, DiT ≈ 0.5B

---

### 2.4 성능 향상

**정량적 결과 (User Study, Table 2):**

| | Input Consistency | Triangle Aesthetic | Overall Quality |
|---|---|---|---|
| MeshAnything | 2.93 | 2.43 | 2.43 |
| MeshAnythingV2 | 2.64 | 2.36 | 2.14 |
| **Ours** | **4.83** | **4.54** | **4.58** |

**주요 성능 지표:**
- 최대 **4,000 faces** 생성 (기존 1,600의 2.5배)
- **$512^3$** 공간 해상도 (기존 $128^3$의 64배 세밀도)
- Sub-sequence 수 54.7 (AMT 199.5 대비 약 73% 감소)
- 토크나이제이션 속도 25.2 mesh/sec (AMT 8.6의 약 3배)

---

### 2.5 한계

1. **압축률 부족**: 47.4%로 여전히 복잡한 게임 캐릭터 등 초고폴리 메시 불가
2. **성공률 저하**: 4,000 faces 이상에서 생성 실패율 증가 및 시간 급증 (~3분)
3. **제어의 어려움**: 동일 입력에서 다양한 결과 → 정밀한 위상(topology) 제어 불가
4. **추론 속도**: A100 GPU에서 약 100 tokens/sec → 1,000 faces에 45초 소요
5. **훈련 비용**: 64×A100 80GB GPU로 약 1주 소요

---

## 3. 모델의 일반화 성능 향상 가능성 (심층 분석)

### 3.1 일반화 문제의 근본 원인과 ArAE의 해결책

기존 auto-regressive 방법들의 일반화 실패 원인은 **복잡한 조건(이미지 등)을 가변 길이 시퀀스에 직접 연결**하는 구조에 있다. EdgeRunner는 이를 **2단계 분리 학습**으로 해결한다:

```
[복잡한 조건 (Image)]
         ↓
[DiT: 조건 → 잠재 코드] ← 일반화 담당
         ↓
[고정 길이 잠재 코드 Z]
         ↓
[ArAE Decoder: 잠재 코드 → 메시] ← 메시 구조 학습 담당
```

**핵심 설계 철학**: 어려운 조건 매핑은 확산 모델이 담당하고, 메시 생성은 기하학적으로 직관적인 포인트 클라우드 조건으로 학습한다.

### 3.2 일반화 향상의 구체적 메커니즘

#### (a) Fixed-length Latent Space의 역할

- 가변 길이 메시를 $\mathbf{Z} \in \mathbb{R}^{2048 \times 64}$로 압축 → **표준 확산 모델 학습 가능**
- $\mathcal{L}_{\text{reg}} = \|\mathbf{Z}\|_2^2$를 통해 잠재 공간의 분포를 정규화 → DiT가 예측하기 용이한 공간 형성
- 이는 2D LDM(Rombach et al., 2022)의 VAE 역할과 동일한 원리

#### (b) 직접 이미지 조건부 방식 vs ArAE 방식 비교 (Ablation Study)

논문 Figure 8 우측의 ablation은 **직접 이미지 조건부 방식의 실패**를 보여준다:

| 방식 | 수렴성 | 미학습 케이스 일반화 |
|------|--------|---------------------|
| Direct Image Conditioning | 수렴 어려움 | 불량 |
| **DiT + ArAE (Ours)** | 안정적 | **우수** |

#### (c) CLIP 특성 활용

CLIP-H 모델의 풍부한 시각-의미 표현을 이미지 조건으로 활용:
- 훈련 시: 단순 쉐이딩으로 렌더링된 이미지만 사용
- 추론 시: 2D 스타일 이미지, 사실적 조명 이미지에서도 우수한 성능
- **Domain gap를 CLIP의 대규모 사전학습이 보완**

#### (d) 데이터 증강을 통한 일반화

```
훈련 중 데이터 증강:
1. 스케일 랜덤화: [0.75, 0.95] 범위
2. 수직축 회전: ±30도
3. Quadric edge collapse 데시메이션: 50% 확률로 면 수 1/4로 감소
```

이를 통해 하나의 포인트 클라우드 입력에서 다양한 메시 출력 가능성을 학습.

#### (e) 조건 모달리티 확장 가능성

고정 길이 잠재 공간은 향후 **텍스트, 멀티뷰 이미지, 스케치** 등 다양한 조건을 DiT만 교체하여 적용 가능한 플러그앤플레이 구조를 제공한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 연구 흐름 분류

```
3D 메시 생성 연구 계보 (2020~2024)
├── 최적화 기반 (SDS)
│   ├── DreamFusion (2022) - 2D diffusion → 3D
│   ├── Magic3D (2023) - 고해상도
│   └── DreamGaussian (2023) - 가우시안 스플래팅
├── 피드포워드 (LRM 계열)
│   ├── LRM (2023) - triplane-NeRF
│   ├── InstantMesh (2024) - sparse-view
│   └── MeshLRM (2024) - 직접 메시 출력
├── 3D 네이티브 확산
│   ├── Michelangelo (2023) - shape-image-text 정렬
│   ├── CLAY (2024) - 대규모 잠재 확산
│   └── Direct3D (2024) - 이미지→3D 잠재 확산
└── Auto-regressive 메시 생성 ← EdgeRunner의 영역
    ├── MeshGPT (2024) - VQ-VAE + GPT
    ├── MeshXL (2024) - 좌표 직접 예측
    ├── MeshAnything (2024) - 포인트클라우드 조건
    ├── MeshAnythingV2 (2024) - AMT 토크나이저
    └── EdgeRunner (2024) - ArAE + EdgeBreaker
```

### 4.2 상세 비교 표

| 방법 | 표현 | 최대 Faces | 해상도 | 일반화 조건 | 토폴로지 보존 |
|------|------|-----------|--------|------------|--------------|
| DreamFusion (2022) | NeRF | N/A | 연속 | 텍스트 | ✗ |
| LRM (2023) | triplane-NeRF | N/A | 연속 | 단일 이미지 | ✗ |
| CLAY (2024) | SDF+잠재 | N/A | 연속 | 텍스트/이미지 | ✗ |
| MeshGPT (2024) | 삼각 메시 | ~800 | $128^3$ | 비조건 | ✓ |
| MeshXL (2024) | 삼각 메시 | ~1000 | $128^3$ | 포인트클라우드 | ✓ |
| MeshAnything (2024) | 삼각 메시 | ~800 | $128^3$ | 포인트클라우드 | ✓ |
| MeshAnythingV2 (2024) | 삼각 메시 | 1,600 | $128^3$ | 포인트클라우드 | ✓ |
| **EdgeRunner (2024)** | **삼각 메시** | **4,000** | **$512^3$** | **포인트클라우드 + 이미지** | **✓** |

### 4.3 핵심 차별점 분석

**MeshGPT (Siddiqui et al., CVPR 2024) 대비:**
- 손실 압축(VQ-VAE) → 무손실 압축(EdgeBreaker)으로 메시 품질 향상
- 고정 길이 잠재 공간 추가로 조건부 생성 가능

**MeshAnythingV2 (Chen et al., 2024) 대비:**
- Sub-sequence 수: 199.5 → 54.7 (73% 감소)로 학습 용이성 향상
- 양방향(N/P) 이동 vs AMT의 단방향 이동
- 이미지 조건부 생성 가능 (MeshAnythingV2 불가)

**CLAY (Zhang et al., 2024) 대비:**
- CLAY는 SDF 기반 연속 표현 → 후처리 필요, 아티스트 메시 불가
- EdgeRunner는 직접 아티스트 스타일 삼각 메시 생성

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 향후 연구에 미치는 영향

#### (a) 2단계 생성 패러다임의 확립

EdgeRunner가 제시한 **"ArAE로 잠재 공간 구성 → DiT로 조건부 생성"** 패러다임은 향후 3D 메시 생성 연구의 표준 프레임워크가 될 가능성이 높다. 이는 2D 이미지 생성에서 VAE + LDM이 표준이 된 것과 유사한 흐름이다.

#### (b) 메시 토크나이저 연구 촉진

EdgeBreaker 기반 토크나이저의 성공은 더 효율적인 메시 압축 알고리즘 연구를 자극할 것이다. 특히:
- 학습 친화적 압축 vs 이론적 최적 압축의 트레이드오프 연구
- 메시 위상(topology) 정보를 더 잘 보존하는 토크나이저 설계

#### (c) 3D 대규모 언어/생성 모델의 기반

고정 길이 잠재 공간은 **텍스트-3D, 오디오-3D, 비디오-3D** 등 다양한 멀티모달 연결을 가능하게 하는 기반이 된다.

#### (d) 인터랙티브 3D 편집 연구

Face count condition 메커니즘은 향후 사용자 인터랙티브 메시 편집 연구의 방향을 제시한다. LOD(Level of Detail) 자동화, 게임 엔진 통합 등에 응용 가능.

### 5.2 향후 연구 시 고려해야 할 점

#### (a) 토크나이저 압축률 개선
현재 47.4% 압축률은 여전히 복잡한 게임 캐릭터 메시(수만 개 face)에는 부족하다. 다음을 고려해야 한다:
- 계층적 메시 표현(coarse-to-fine) 도입
- 메시 간소화(simplification)와 생성을 연계하는 방법
- 학습 기반의 적응형 토크나이저 설계

#### (b) 추론 속도 병목 해결
현재 A100에서 4,000 faces 생성에 약 3분 소요는 실용성 저해 요인이다:
- Speculative decoding 적용 검토
- Non-auto-regressive 디코딩과의 하이브리드 방식
- Diffusion 기반 직접 메시 생성으로 전환 가능성

#### (c) 잠재 공간의 연속성과 편집 가능성
$\mathcal{L}_{\text{reg}} = \|\mathbf{Z}\|_2^2$는 단순한 정규화이며, VAE의 KL divergence와 달리 잠재 공간의 구조적 의미를 보장하지 않는다. 향후:
- VQ-VAE나 $\beta$-VAE 등을 통한 더 구조화된 잠재 공간 탐구
- 잠재 코드의 보간(interpolation)을 통한 메시 편집 가능성

#### (d) 텍스처 및 재질(PBR) 통합
현재 EdgeRunner는 기하학적 메시만 생성하며 텍스처가 없다. 최근 Meta 3D AssetGen(Siddiqui et al., 2024b)처럼 UV 매핑과 PBR 재질까지 포함하는 통합 생성으로 확장이 필요하다.

#### (e) 대규모 데이터셋과 스케일링
약 112K 메시로 학습했으나, 현재 Objaverse-XL에는 1000만 개 이상의 3D 객체가 존재한다. 대규모 학습 시의 스케일링 법칙(scaling law)과 품질 필터링 전략이 중요한 연구 과제이다.

#### (f) 평가 지표 표준화
현재 논문은 User Study에 크게 의존하며 자동화된 정량적 평가 지표가 부족하다. 메시 품질 평가를 위한 표준 벤치마크 구축이 필요하다:
- Chamfer Distance (기하학적 정확도)
- 위상 정확도 (non-manifold 비율 등)
- 아티스트 메시 유사도를 측정하는 새로운 지표

---

## 참고자료 (출처)

1. **Tang et al. (2024). "EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation"** — arXiv:2409.18114v1 [cs.CV] *(본 논문, 제공된 PDF)*

2. **Siddiqui et al. (2024). "MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers"** — CVPR 2024, pp. 19615–19625 *(논문 내 인용)*

3. **Chen et al. (2024). "MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers"** — arXiv:2406.10163 *(논문 내 인용)*

4. **Chen et al. (2024). "MeshAnythingV2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization"** — arXiv:2408.02555 *(논문 내 인용)*

5. **Chen et al. (2024). "MeshXL: Neural Coordinate Field for Generative 3D Foundation Models"** — arXiv:2405.20853 *(논문 내 인용)*

6. **Rossignac, J. (1999). "EdgeBreaker: Connectivity Compression for Triangle Meshes"** — IEEE Transactions on Visualization and Computer Graphics, 5(1):47–61 *(논문 내 인용)*

7. **Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models"** — CVPR 2022, pp. 10684–10695 *(논문 내 인용)*

8. **Zhang et al. (2024). "CLAY: A Controllable Large-Scale Generative Model for Creating High-Quality 3D Assets"** — arXiv:2406.13897 *(논문 내 인용)*

9. **Zhang et al. (2022). "OPT: Open Pre-trained Transformer Language Models"** — arXiv:2205.01068 *(논문 내 인용)*

10. **Deitke et al. (2023). "Objaverse: A Universe of Annotated 3D Objects"** — CVPR 2023; **"Objaverse-XL"** — arXiv:2307.05663 *(논문 내 인용)*

11. **Ho et al. (2020). "Denoising Diffusion Probabilistic Models"** — NeurIPS 2020 *(논문 내 인용)*

12. **Ilharco et al. (2021). "OpenCLIP"** — Zenodo *(논문 내 인용)*

13. **Weng et al. (2024). "PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance"** — arXiv:2405.16890 *(논문 내 인용)*

14. **Wu et al. (2024). "Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image"** — arXiv *(논문 내 인용)*

15. **Hang et al. (2023). "Efficient Diffusion Training via Min-SNR Weighting Strategy"** — ICCV 2023 *(논문 내 인용)*
