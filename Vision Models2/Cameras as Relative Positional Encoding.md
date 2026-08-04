# Cameras as Relative Positional Encoding

---

## 1. Executive Summary (10문장 이내)

**"Cameras as Relative Positional Encoding"** (Li et al., 2025, arXiv:2507.10496v2)은 멀티뷰 트랜스포머에서 카메라 기하학을 조건화하는 방법론을 체계적으로 비교·분석하고 새로운 방법인 **PRoPE(Projective Positional Encoding)**를 제안한다.  
기존 주류 방식인 raymap 인코딩(절대 인코딩)은 임의적인 세계 좌표계에 의존하여 일반화 성능을 저해한다는 근본적 한계를 갖는다.  
PRoPE는 카메라 내부 파라미터(intrinsics)와 외부 파라미터(extrinsics) 모두를 상대적 투영 관계 $\tilde{P}\_{i_1}\tilde{P}_{i_2}^{-1}$로 인코딩하는 어텐션 수준의 상대적 위치 인코딩이다.  
실험은 Novel View Synthesis(NVS), 스테레오 깊이 추정, 공간 인지 판별 등 세 가지 태스크, 여섯 개 데이터셋에 걸쳐 수행되었다.  
PRoPE는 공유 intrinsics와 가변 intrinsics 두 설정 모두에서 기존 절대/상대 인코딩 대비 우월한 성능을 보였다.  
특히 학습 시와 다른 시퀀스 길이 및 focal length에 대한 OOD(Out-of-Distribution) 일반화에서 두드러진 강건성을 보였다.  
토큰 수준(CamRay)과 어텐션 수준(PRoPE)의 하이브리드 인코딩이 상호 보완적임을 확인하였다.  
PRoPE는 UniMatch(깊이 추정)와 CAT3D(대형 멀티뷰 확산 모델)에 통합 시에도 일관된 성능 향상을 보여 태스크 일반성을 입증했다.  
추가 파라미터 없이 기존 FlashAttention 커널과 호환 가능하여 실용적이다.  
본 논문은 카메라 기하학을 상대적 위치 인코딩으로 표현하는 패러다임 전환을 촉구하는 중요한 연구이다.

---

### 1-1. 연구의 목적과 필요성

멀티뷰 컴퓨터 비전(3D 재구성, NVS, 깊이 추정, 로보틱스 등)에서 트랜스포머 모델은 각 이미지 패치 토큰에 카메라 기하학 정보를 결합해야 한다.  
기존 지배적 방법인 **raymap 인코딩**은 픽셀마다 절대적 세계 좌표계 기반의 ray 원점/방향을 부여하는 절대 위치 인코딩(APE) 방식이다. 이는 두 가지 근본적 문제를 야기한다:

1. **좌표계 임의성(Reference Frame Arbitrariness)**: 세계 좌표계의 선택이 임의적이어서 모델의 일반화를 저해한다 (p.3).
2. **Intrinsics 무시**: 기존 SE(3) 기반 상대 인코딩(CaPE, GTA)은 카메라 포즈만 고려하고 내부 파라미터(focal length, FOV)를 무시한다 (p.5).

언어 모델에서의 진화(APE → RPE/RoPE)와 마찬가지로, 멀티뷰 비전에서도 절대 → 상대 인코딩으로의 전환이 필요하다는 것이 본 연구의 핵심 동기이다 (p.2, Figure 1).

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거(실험) | 위치 |
|---|-----------|-----------|------|
| 1 | 상대 인코딩이 절대 인코딩보다 우수 | RealEstate10K: PRoPE PSNR 22.80 vs Plücker 20.48 | Table 1, p.6 |
| 2 | PRoPE는 가변 intrinsics에서 기존 상대 인코딩(GTA, CaPE)을 압도 | 가변 intrinsics 설정에서 GTA PSNR 15.77 → PRoPE 21.42 | Table 2, p.6 |
| 3 | PRoPE는 OOD 시퀀스 길이에 강건 | 16-view 테스트 시 PRoPE가 일관되게 최고 성능 | Figure 4(a), p.8 |
| 4 | PRoPE는 OOD focal length에 강건 | 5× zoom 테스트에서 PRoPE+CamRay SSIM 0.776 (최고) | Table A.3, p.16 |
| 5 | 토큰/어텐션 수준 인코딩은 상호 보완적 | PRoPE+CamRay > PRoPE > GTA+CamRay (varying intrinsics) | Table 3, p.7 |
| 6 | PRoPE는 깊이 추정 태스크로 일반화 | UniMatch+PRoPE Abs Rel 0.105 vs 0.123 (RGBD) | Table 4, p.9 |
| 7 | PRoPE는 공간 인지 판별 태스크로 일반화 | PRoPE+CamRay 94.3% (17-view) vs Plücker 74.6% | Table 5, p.9 |
| 8 | 대형 모델에서도 PRoPE 이점 지속 | 100× compute: PRoPE PSNR 26.56 vs Plücker 25.64 | Table 6, p.10 |
| 9 | PRoPE는 추가 파라미터/오버헤드 없음 | CAT3D에 적용 시 zero additional parameters | p.10, Table A.2 |

---

### 2-1. 해결 문제·제안 방법·모델 구조·성능 향상·한계 상세 설명

#### 해결하고자 하는 문제

- **문제 1**: 기존 raymap(절대 인코딩)은 임의적 세계 좌표계에 민감하여 일반화 저하
- **문제 2**: 기존 상대 인코딩(CaPE, GTA)은 SE(3) 포즈만 고려하고 카메라 intrinsics 무시
- **문제 3**: 학습 시와 다른 입력 뷰 수 또는 focal length에 대한 취약한 OOD 강건성

---

#### 제안 방법: PRoPE (Projective Positional Encoding)

**핵심 아이디어**: 두 카메라 $i_1$, $i_2$ 간의 관계를 SE(3) 포즈 변환 대신 완전한 투영 행렬(projection matrix) 간의 관계로 정의한다.

**투영 행렬 정의** (p.3, Eq. 3, 4):

```math
\boldsymbol{P}_i = \begin{bmatrix} \boldsymbol{K}_i & \boldsymbol{0}_{3\times1} \end{bmatrix} \boldsymbol{T}_i^{cw} \in \mathbb{R}^{3\times4}
```

```math
\tilde{\boldsymbol{P}}_i = \begin{bmatrix} \boldsymbol{P}_i \\ \boldsymbol{e}_4^\top \end{bmatrix} \in \mathbb{R}^{4\times4}
```

**PRoPE의 핵심 상대 관계** (p.5, Eq. 16):

$$\tilde{\boldsymbol{P}}_{i_1} \tilde{\boldsymbol{P}}_{i_2}^{-1}$$

이를 전개하면 (p.5, Eq. 20):

```math
\tilde{\boldsymbol{P}}_{i_1} \tilde{\boldsymbol{P}}_{i_2}^{-1} = \begin{bmatrix} \boldsymbol{K}_i & \boldsymbol{0} \\ \boldsymbol{0} & 1 \end{bmatrix} \boldsymbol{T}_i^{cw}(\boldsymbol{T}_j^{cw})^{-1} \begin{bmatrix} \boldsymbol{K}_j^{-1} & \boldsymbol{0} \\ \boldsymbol{0} & 1 \end{bmatrix}
```

→ 세계 좌표계 재정의 시 우변의 SE(3) 항이 대수적으로 상쇄되어 **전역 좌표계 불변성** 보장

**PRoPE 행렬 정의** (p.5, Eq. 17–19):

```math
\mathbf{D}_t^{\text{PRoPE}} = \begin{bmatrix} \mathbf{D}_t^{\text{Proj}} & \mathbf{0} \\ \mathbf{0} & \mathbf{D}_t^{\text{RoPE}} \end{bmatrix}
```

$$\mathbf{D}_t^{\text{Proj}} = \mathbf{I}_{d/8} \otimes \tilde{\boldsymbol{P}}_{i(t)} \in \mathbb{R}^{\frac{d}{2} \times \frac{d}{2}}$$

```math
\mathbf{D}_t^{\text{RoPE}} = \begin{bmatrix} \text{RoPE}_{d/4}(x_t) & \mathbf{0} \\ \mathbf{0} & \text{RoPE}_{d/4}(y_t) \end{bmatrix} \in \mathbb{R}^{\frac{d}{2} \times \frac{d}{2}}
```

- $\mathbf{D}_t^{\text{Proj}}$: 카메라 frustum 간 투영 관계 인코딩 (intrinsics + extrinsics)
- $\mathbf{D}_t^{\text{RoPE}}$: 2D 패치 좌표 $(x_t, y_t)$ 기반 RoPE 인코딩

**PRoPE 어텐션 메커니즘** (GTA-style, p.4, Eq. 14 기반):

$$\text{Attn}^{\text{GTA}}(Q,K,V) = \mathbf{D}^{\text{GTA}} \circledast \text{Attn}\left((\mathbf{D}^{\text{GTA}})^\top \circledast Q,\ (\mathbf{D}^{\text{GTA}})^{-1} \circledast K,\ (\mathbf{D}^{\text{GTA}})^{-1} \circledast V\right)$$

PRoPE에서는 $\mathbf{D}^{\text{GTA}} \leftarrow \mathbf{D}^{\text{PRoPE}}$로 대체.

---

**비교 메서드별 수식 요약**:

| 방법 | 인코딩 위치 | 핵심 수식 | Intrinsics 포함 여부 |
|------|------------|-----------|---------------------|
| Naïve Raymap | 토큰 수준 | $\mathbf{M}_{i,\text{Naive}}^{u,v} = [\mathbf{o}_i;\ \mathbf{d}_i^{u,v}] \in \mathbb{R}^6$ (Eq.6) | ✅ (절대) |
| Plücker Raymap | 토큰 수준 | $\mathbf{M}_{i,\text{Plücker}}^{u,v} = [\mathbf{o}_i \times \mathbf{d}_i^{u,v};\ \mathbf{d}_i^{u,v}] \in \mathbb{R}^6$ (Eq.8) | ✅ (절대) |
| CaPE | 어텐션 수준 | $\mathbf{D}\_t^{\text{CaPE}} = \mathbf{I}\_{d/4} \otimes \boldsymbol{T}_{i(t)}^{cw}$ (Eq.11) | ❌ |
| GTA | 어텐션 수준 | SE(3) + RoPE (Eq.14) | ❌ |
| **PRoPE** | **어텐션 수준** | $\tilde{\boldsymbol{P}}\_{i_1}\tilde{\boldsymbol{P}}_{i_2}^{-1}$ + RoPE (Eq.17–19) | **✅ (상대)** |
| CamRay | 토큰 수준 | $\mathbf{M}_{i,\text{CamRay}}^{u,v} = \mathbf{R}_i^{cw}\mathbf{d}_i^{u,v} \propto \boldsymbol{K}_i^{-1}[u\ v\ 1]^\top$ (Eq.21) | ✅ (지역 프레임) |

---

#### 모델 구조

- **기반 모델**: LVSM (Large View Synthesis Model) [8] 프레임워크를 기반으로 재구현
- **주요 실험 모델 크기**: ~25M 파라미터, 6개 트랜스포머 블록, 256×256 해상도
- **스케일업 모델**: 12개 블록, MLP dim 3072, 8× GPU, batch 64 (Section 4.7)
- **깊이 추정**: UniMatch [14]의 cross-view attention에 PRoPE 통합 (~50줄 코드 수정)
- **공간 인지**: LVSM 기반, 마지막 선형 레이어를 분류 헤드로 교체

---

#### 성능 향상

| 설정 | 지표 | Plücker | GTA | PRoPE | 향상 (vs Plücker) |
|------|------|---------|-----|-------|------------------|
| NVS, 고정 intrinsics (RE10K) | PSNR↑ | 20.48 | 22.51 | **22.80** | +2.32 dB |
| NVS, 가변 intrinsics (RE10K) | PSNR↑ | 19.89 | 15.77 | **21.42** | +1.53 dB |
| 스테레오 깊이 (Scenes11) | Abs Rel↓ | - | - | **0.049** vs 0.065 | -24.6% |
| 공간 인지 (17-view) | Acc↑ | 74.6% | - | **94.3%** | +19.7%p |
| 대형 모델 (100× compute) | PSNR↑ | 25.64 | - | **26.56** | +0.92 dB |

---

#### 한계

1. **수치 불안정성**: 망원 렌즈(telephoto) 등 ill-conditioned 투영 행렬이 Q/K/V 벡터에 직접 곱해질 때 수치 안정성 문제 가능 (p.10)
2. **왜곡 카메라 미지원**: 어안 렌즈 등 왜곡(distorted) 카메라 모델에는 적용 불가 (p.10)
3. **다중 주파수 인코딩 미구현**: 투영 변환의 비가환성(non-commutativity)으로 인해 multi-frequency 인코딩 적용이 비자명 (p.10)
4. **소규모 학습 자원**: 주요 실험이 batch size 4, 2×GPU의 소규모 환경에서 수행됨 (Appendix A.1.1)

---

## 3. 각 주장별 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|------|----------|
| 절대 vs 상대 인코딩 비교 철학 | p.2, Figure 1 |
| Naïve raymap 수식 | p.3, Eq. 6-7 |
| Plücker raymap 수식 | p.4, Eq. 8 |
| CaPE 수식 | p.4, Eq. 11-13 |
| GTA 수식 | p.4, Eq. 14-15 |
| PRoPE 수식 | p.5, Eq. 16-20 |
| 고정 intrinsics NVS 결과 | p.6, Table 1 |
| 가변 intrinsics NVS 결과 | p.6, Table 2 |
| 하이브리드 인코딩 결과 | p.7, Table 3, Figure 2 |
| OOD 강건성 결과 | p.8, Figure 3, 4, 5, 6 |
| 깊이 추정 결과 | p.9, Table 4, Figure 7 |
| 공간 인지 결과 | p.9, Table 5 |
| 스케일링 결과 | p.10, Table 6 |
| CAT3D 통합 결과 | p.10, Table A.2 (p.16) |
| PRoPE 절제 연구 | Table A.1 (p.16) |
| OOD intrinsics 추가 결과 | Table A.3 (p.16) |

---

## 4. 저자 직접 보고 결과 vs. 독자 해석 분리

### 4-1. 저자가 직접 보고한 결과

**연구 주제**: 멀티뷰 트랜스포머에 카메라 기하학을 조건화하는 방법 비교 및 PRoPE 제안

**저자 직접 보고 수치**:
- Table 1: PRoPE PSNR 22.80 (RE10K), 23.70 (Objaverse) — GTA와 동점 (Objaverse)
- Table 2: 가변 intrinsics에서 CAPE PSNR 15.94, GTA 15.77로 raymap보다 오히려 열등; PRoPE 21.42로 최고
- Table 6: 100× compute PRoPE PSNR 26.56 vs Plücker 25.64
- Table A.1: $\mathbf{D}_t^{\text{Proj}}$ 제거 시 PSNR 16.04 (vs PRoPE 21.78), $\mathbf{D}_t^{\text{RoPE}}$ 제거 시 21.39

**저자 직접 기술한 한계**:
> "improving numerical stability when directly multiplying projective matrices with Q/K/V vectors; it may be possible, for example, for ill-conditioned matrices to emerge from telephoto focal lengths." (p.10)

---

### 4-2. 독자(검토자) 해석

- **고정 intrinsics에서의 PRoPE 우위**: Objaverse(고정 intrinsics)에서 GTA와 동점인 것은 저자 주장(PRoPE가 intrinsics 불필요 시 GTA로 환원)과 일치하나, RE10K에서의 소폭 우위(22.80 vs 22.51)는 모델 크기가 25M 수준의 소규모임을 감안할 때 통계적 유의성이 불명확하다. ⚠️ **통계적 유의성 검정 미제시**

- **CaPE/GTA의 가변 intrinsics 급락**: Table 2에서 CaPE(15.94)와 GTA(15.77)가 raymap(19.89, 20.56)보다 현저히 낮은 것은, SE(3)만 고려하는 상대 인코딩이 intrinsics 변화에 오히려 혼란을 야기함을 시사한다. 이는 저자 주장을 강하게 지지하는 흥미로운 결과이다.

- **OOD 강건성의 메커니즘**: PRoPE가 OOD focal length에 강건한 것은 $\tilde{P}\_{i_1}\tilde{P}_{i_2}^{-1}$ 내에 $K_i K_j^{-1}$ 항이 명시적으로 포함되어 있기 때문으로 해석되나, 이 메커니즘이 실제로 작동하는지에 대한 어텐션 맵 수준의 분석은 제공되지 않았다.

- **UniMatch 통합의 공정성 문제**: Table A.4의 추가 실험이 1/8 학습 자원으로 수행되었다는 점은 Table 4의 주요 결과와 비교 시 일관성 문제를 야기할 수 있다. ⚠️

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 문제점 | 위치 |
|------|--------|------|
| ⚠️ 표준편차/신뢰구간 미제시 | 모든 정량 테이블에서 단일 수치만 보고, 반복 실험 결과 없음 | Table 1–6 전체 |
| ⚠️ 소규모 모델 편향 | 주 실험이 ~25M, batch 4, 2GPU로 수행 — 산업 수준 모델(수십억 파라미터)과 비교 불가 | Appendix A.1.1 |
| ⚠️ Table 4 RGBD의 Sq Rel 신뢰도 저하 | "†The 'Sq Rel' metric is less reliable on the RGBD dataset due to imperfect depth and camera pose" 저자 직접 인정 | Table 4, p.9 |
| ⚠️ Table A.4의 학습 자원 불일치 | UniMatch 추가 실험이 1/8 자원(2 GPU × 50k steps)으로 수행 | Table A.4, p.17 |
| ⚠️ CAT3D 실험의 재현 불투명성 | "With assistance from the original authors"로 수행 — 독립 재현 가능성 불명확 | p.10 |
| ⚠️ OOD intrinsics 테스트 분포 편향 | 1–3× zoom으로 학습 후 1–7× zoom 테스트 — 분포 차이의 범위가 제한적 | Table A.3 |
| ⚠️ Objaverse 가변 intrinsics 범위 협소 | FOV 35–50도 범위만 테스트 — 극단적 광각/망원 미포함 | p.6 |

---

## 6. 논문이 답하지 않는 질문

1. **학습 데이터 다양성의 영향**: 더 다양한 intrinsics를 가진 대규모 데이터셋(예: 야외 자율주행 데이터)에서의 PRoPE 성능은 어떠한가?

2. **수치 안정성의 정량적 한계**: 어느 수준의 focal length에서 ill-conditioned 행렬 문제가 실제로 발생하는가? 이를 완화하는 정규화 방법은 무엇인가?

3. **왜곡 카메라 모델 확장 방법**: 어안 렌즈 등의 비선형 카메라 모델에 PRoPE를 어떻게 확장할 수 있는가? 패치별 지역 근사 외에 다른 접근법이 있는가?

4. **어텐션 맵 분석**: PRoPE가 실제로 카메라 기하학에 어텐션을 집중시키는지에 대한 해석 가능성(interpretability) 분석이 없다.

5. **포즈 추정 오차에 대한 강건성**: 카메라 포즈가 완벽하게 알려지지 않은 실세계 설정에서의 성능 저하는 어느 수준인가?

6. **비디오/동적 장면 적용**: 시간적으로 변화하는 장면에서 PRoPE가 어떻게 확장될 수 있는가?

7. **단일 이미지 입력**: 참조 이미지가 1장인 극단적 sparse 설정에서의 성능은 어떠한가?

8. **Cross-attention 적용 가능성**: 본 논문은 self-attention에 집중하였으나, encoder-decoder 구조의 cross-attention에도 PRoPE를 어떻게 적용할 수 있는가?

9. **학습 안정성**: 투영 행렬을 Q/K/V에 직접 곱하는 방식이 학습 초기 안정성에 미치는 영향은?

10. **Multi-frequency 인코딩**: 비가환성 문제를 극복하여 PRoPE에 다중 주파수 인코딩을 적용하는 방법은 존재하는가?

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2): Cameras as Relative Positional Encoding

**내용**: 언어 모델과 멀티뷰 트랜스포머의 위치 인코딩 패러다임 비교 다이어그램.

**해석**: 언어 모델이 절대 위치(1,2,3) → 상대 거리(distance)로 진화한 것처럼, 멀티뷰 트랜스포머도 절대 카메라 파라미터 $(K_1, T_1^{cw}), (K_2, T_2^{cw}), (K_3, T_3^{cw})$ → 상대적 변환(Transformation)으로 진화해야 함을 직관적으로 제시한다. 이 그림은 본 논문의 핵심 철학을 한 장으로 압축한 것으로, NLP 분야의 성공적 전환(APE→RoPE)을 컴퓨터 비전의 카메라 인코딩에 analogize하는 논문의 핵심 내러티브를 확립한다.

---

### Figure 3 (p.8): Out-of-Distribution Tasks

**내용**: OOD 설정 두 가지(긴 시퀀스, 미지 intrinsics)와 각 방법의 성능 위치를 2D 플롯으로 시각화.

**해석**: 우측 산점도에서 PRoPE가 두 OOD 축(긴 시퀀스 길이 + 미지 intrinsics) 모두에서 최상단에 위치하고, Raymap은 하단(긴 시퀀스에 강하나 intrinsics OOD에 취약), CAPE/GTA는 좌측(시퀀스 확장성은 있으나 intrinsics OOD 취약)에 위치한다. 이 그림은 PRoPE가 두 가지 일반화 방향 모두를 동시에 개선하는 유일한 방법임을 명확히 보여준다. Raymap이 intrinsics 정보를 인코딩함에도 OOD intrinsics에 취약한 이유는 절대 좌표계 의존성 때문으로 해석된다.

---

### Figure 4 (p.8): Evaluation on RealEstate10K

**내용**: (a) 2–16 input views 테스트 시 PSNR/SSIM/LPIPS 곡선, (b) 1–3× zoom-in 테스트 시 성능 곡선.

**해석**: 
- **(a) 시퀀스 길이 OOD**: 모든 방법이 뷰 수 증가 시 성능 향상되나, PRoPE/PRoPE+CamRay가 가장 가파른 상승 곡선을 보인다. CAPE는 16-view에서도 Plücker보다 낮아, SE(3) 상대 인코딩만으로는 시퀀스 확장 이점이 제한적임을 보여준다.
- **(b) Focal length OOD**: 3×~5× zoom에서 Plücker와 GTA는 성능이 급락하나, PRoPE 계열은 완만한 하락을 보인다. 특히 PRoPE가 PRoPE+CamRay보다 극단적 zoom에서 더 안정적인 것은, 절대 intrinsics 정보(CamRay)가 오히려 극단적 OOD에서 방해가 됨을 시사한다.

---

### Figure 5 (p.8): Results of Longer Input Sequences at Test Time

**내용**: 2-view(학습 설정)와 16-view(OOD 테스트) 시 각 방법의 NVS 결과 이미지 비교.

**해석**: 2-view 테스트에서는 방법 간 시각적 차이가 상대적으로 작으나, 16-view 테스트에서 Plücker와 CAPE는 심각한 아티팩트(흐림, 잘못된 기하학)가 발생하는 반면, PRoPE와 PRoPE+CamRay는 명확한 기하학적 구조(예: 가구 모서리, 질감)를 유지한다. 이는 PRoPE의 상대적 frustum 관계 인코딩이 학습 시와 다른 뷰 수에도 일관된 기하학적 이해를 유지하게 함을 직관적으로 보여주며, 실제 배포 시나리오에서의 실용성을 강조한다.

---

### Figure 7 (p.9): Qualitative Results on Stereo Depth Estimation

**내용**: UniMatch vs UniMatch+PRoPE의 스테레오 깊이 추정 결과를 RGBD, Scenes11, SUN3D 세 데이터셋에서 비교.

**해석**: UniMatch+PRoPE는 경계(edge) 보존 및 원거리 객체 깊이 추정에서 뚜렷한 개선을 보인다. 특히 가구 경계와 벽-바닥 전환 영역에서 PRoPE 적용 시 날카로운 깊이 불연속이 복원된다. 이 결과는 PRoPE가 NVS에 특화된 것이 아니라 일반적인 멀티뷰 기하학 이해를 향상시킴을 입증하며, 약 50줄의 코드 수정만으로 기존 모델에 통합 가능한 실용성도 확인된다. 단, NVS 실험과 달리 정성적 결과만 제시되어 있어 개선 정도의 정량화가 Table 4로 분리되어 있다.

---

## 8. 결론, 시사점, 후속 연구

### 8-1. 저자 제시 시사점 및 후속 연구 계획

**저자 직접 제시 결론** (p.10):
- 카메라를 상대적 위치 인코딩으로 표현하는 것—특히 intrinsics와 extrinsics를 모두 포착하는 방식—이 다양한 태스크와 설정에서 멀티뷰 트랜스포머를 일관되게 개선한다.

**저자 제시 후속 연구 방향** (p.10):
1. **수치 안정성 개선**: 투영 행렬을 Q/K/V에 직접 곱할 때의 ill-conditioned 행렬 문제 해결
2. **왜곡 카메라 모델 확장**: PRoPE를 패치별 지역 투영 근사(per-patch projective approximations)로 확장하여 왜곡 카메라 지원
3. **Multi-frequency 인코딩**: 투영 변환의 비가환성 문제를 극복한 다중 주파수 카메라 파라미터 인코딩

---

### 8-1. 모델의 일반화 성능 향상 가능성 (심화 분석)

PRoPE의 일반화 성능 향상은 세 가지 메커니즘에서 비롯된다:

**메커니즘 1: 전역 좌표계 불변성**

$$\tilde{P}_{i_1}\tilde{P}_{i_2}^{-1} = \begin{bmatrix}K_i & 0\\0 & 1\end{bmatrix}T_i^{cw}(T_j^{cw})^{-1}\begin{bmatrix}K_j^{-1} & 0\\0 & 1\end{bmatrix}$$

세계 좌표계를 $T_{\text{new}}^w$로 재정의하면 양측 SE(3) 항이 소거되어, 학습 데이터의 특정 좌표계 관습(convention)에 과적합되지 않는다.

**메커니즘 2: Intrinsics 명시적 상대화**

기존 GTA/CaPE가 $K$를 무시하는 것과 달리, PRoPE는 $K_i K_j^{-1}$ 항을 통해 두 카메라 간의 focal length 비율을 직접 어텐션에 주입한다. 이는 모델이 특정 focal length 값 자체가 아닌 **비율적 관계**를 학습하게 하여, 미지 focal length에 대한 강건성을 부여한다.

**메커니즘 3: RoPE 기반 시퀀스 외삽**

$\mathbf{D}_t^{\text{RoPE}}$ 항이 GTA의 SO(2) 인코딩을 그대로 계승하여, 언어 모델에서 RoPE가 긴 컨텍스트로 외삽(extrapolation)되는 것과 동일한 이점을 시퀀스 길이 일반화에 제공한다.

**일반화 성능 향상의 실증적 증거**:

| 일반화 유형 | PRoPE 성능 (RE10K) | 최선 기존 방법 | 개선 |
|------------|-------------------|-------------|------|
| 16-view (학습: 2-view) | ~22 PSNR | GTA ~19 | +3 dB |
| 5× zoom (학습: 1×) | SSIM 0.775 | GTA+CamRay 0.764 | +0.011 |
| 7× zoom (학습: 1×) | SSIM 0.794 | GTA+CamRay 0.780 | +0.014 |

**향후 일반화 개선을 위한 연구 방향**:

1. **Temperature Scaling for PRoPE**: 투영 행렬의 스케일 차이($K$의 크기가 focal length에 비례)가 어텐션 소프트맥스에 미치는 영향을 완화하기 위한 학습 가능한 스케일링 인자 도입
2. **Conditional Normalization**: $K$ 행렬을 정규화하여 ill-conditioning 방지 (예: $\hat{K} = K / f_{\max}$ 형태)
3. **Meta-learning 기반 적응**: 소수의 새로운 카메라 설정으로 빠르게 적응하는 few-shot 방식
4. **물리적 제약 통합**: 에피폴라 기하학(epipolar geometry)을 어텐션 마스크로 활용하여 더 강력한 귀납적 편향 부여

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 제공된 논문 원문의 참고문헌과 공개 정보에 기반합니다. 본 논문(2507.10496v2, 2025년 7월)이 최신 arXiv 논문으로, 일부 후속 연구와의 직접 비교는 검증이 완전하지 않을 수 있습니다.

#### 주요 관련 연구 계보

| 연구 | 연도 | 방법 | PRoPE와의 관계 |
|------|------|------|--------------|
| NeRF [44] | 2021 | 암묵적 신경 표현, 절대 좌표 | PRoPE가 대체하려는 패러다임의 배경 |
| RoPE [1] (Su et al.) | 2024 | 1D 시퀀스 상대 위치 인코딩 | PRoPE의 $\mathbf{D}_t^{\text{RoPE}}$ 항이 직접 채용 |
| GTA [3] (Miyato et al., ICLR 2024) | 2024 | SE(3) 기반 상대 위치 인코딩 | PRoPE의 직접 전신; intrinsics 미포함이 핵심 차별점 |
| CaPE [4] (Kong et al., CVPR 2024) | 2024 | SE(3) 상대 인코딩 (값 변환 없음) | PRoPE가 GTA-style 값 변환을 추가하여 개선 |
| LVSM [8] (Jin et al.) | 2024 | Plücker raymap 기반 대형 NVS 모델 | PRoPE 통합의 기반 모델 |
| CAT3D [7] (Gao et al.) | 2024 | Naïve raymap 기반 멀티뷰 확산 모델 | PRoPE 통합으로 일관된 개선 확인 |
| Stable Virtual Camera [9] | 2025 | 확산 모델 기반 뷰 합성 | raymap 사용; PRoPE 통합 가능성 있음 |
| UniMatch [14] (Xu et al.) | 2023 | 통합 flow/stereo/depth 추정 | PRoPE 통합으로 성능 향상 입증 |

#### PRoPE가 앞으로의 연구에 미치는 영향

**단기적 영향 (1–2년)**:

1. **대형 멀티뷰 모델의 기본 인코딩 전환**: LVSM, CAT3D 계열 후속 모델들이 Plücker raymap 대신 PRoPE를 채택할 가능성이 높다. 추가 파라미터 없이 성능 향상이 가능하므로 실용적 채택 장벽이 낮다.

2. **자율주행 인식 시스템**: 다양한 zoom 레벨의 카메라를 동시에 사용하는 자율주행 멀티카메라 리그에서 PRoPE의 intrinsics 일반화 능력이 직접적으로 유용하다.

3. **의료 영상**: 다양한 스캐너 설정(FOV, 해상도)을 가진 멀티뷰 CT/MRI 데이터에 적용 가능성이 있다.

**중장기적 영향 (3–5년)**:

1. **비디오 생성 모델**: 시간 축을 추가적인 차원으로 갖는 비디오 생성 모델에서 시공간 카메라 관계를 PRoPE로 인코딩하는 방향

2. **Embodied AI**: 로봇이 다양한 시점에서 수집한 이미지를 처리할 때, 카메라 내/외부 파라미터의 실시간 변화에 강건한 지각 모델 구축

3. **Foundation Models for 3D**: GPT-4V와 같은 대형 비전-언어 모델에 3D 공간 이해를 부여하는 과정에서 PRoPE가 카메라 조건화 모듈로 통합될 가능성

#### 앞으로 연구 시 고려할 점

**기술적 고려사항**:

1. **정규화 전략**: $\tilde{P}_i$ 행렬의 수치적 범위는 focal length에 따라 크게 달라지므로, 학습 안정성을 위한 표준화 방법 필요. 예를 들어:

$$\hat{P}_i = \tilde{P}_i / \|\tilde{P}_i\|_F$$

또는 focal length에 따른 적응적 스케일링.

2. **비가환성 문제의 해결**: $\tilde{P}\_{i_1}\tilde{P}\_{i_2}^{-1} \neq \tilde{P}\_{i_2}\tilde{P}_{i_1}^{-1}$이므로, multi-frequency 인코딩을 위한 Lie algebra 기반 접근법 탐색 필요

3. **FlashAttention과의 호환성 검증**: 논문에서는 호환 가능하다고 주장하나, 구체적인 구현 세부사항(행렬 크기, 희소성 등)에 대한 추가 검증 필요

4. **포즈 불확실성 통합**: SfM(Structure from Motion)으로 추정된 포즈는 오차를 포함하므로, PRoPE를 확률적 프레임워크(Bayesian 또는 dropout 기반)로 확장하는 방안

**평가 설계 고려사항**:

1. **분리 평가의 필요성**: NVS 성능이 높다고 해서 기하학적 이해가 정확한 것은 아니다. 에피폴라 기하학 준수도, 포즈 추정 오차와의 상관관계 등 더 세분화된 평가 지표 개발

2. **실세계 다양성 증가**: 현재 실험은 주로 실내(RE10K) 및 합성(Objaverse) 데이터에 집중. 야외, 항공, 의료 등 다양한 도메인에서의 검증 필요

3. **계산 비용의 정량화**: "negligible overhead"라는 서술적 주장을 추론 시간, FLOPs 등으로 정량화 필요

---

## 참고자료 (본 답변에서 직접 인용한 논문 및 자료)

본 답변은 제공된 PDF 원문만을 기반으로 작성되었습니다:

- **주 논문**: Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, Angjoo Kanazawa. "Cameras as Relative Positional Encoding." arXiv:2507.10496v2 [cs.CV], 13 Nov 2025.

**논문 내 핵심 참고문헌** (원문에서 직접 인용된 것만):
- [1] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," *Neurocomputing*, 2024.
- [3] Miyato et al., "GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers," *ICLR*, 2024.
- [4] Kong et al., "EscherNet: A Generative Model for Scalable View Synthesis (CaPE)," *CVPR*, 2024.
- [5] Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.
- [7] Gao et al., "CAT3D: Create Anything in 3D with Multi-View Diffusion Models," arXiv:2405.10314, 2024.
- [8] Jin et al., "LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias," arXiv:2410.17242, 2024.
- [11] Dao, "FlashAttention-2," arXiv:2307.08691, 2023.
- [12] Zhou et al., "Stereo Magnification: Learning View Synthesis Using Multiplane Images (RealEstate10K)," *ACM TOG*, 2018.
- [13] Deitke et al., "Objaverse," *CVPR*, 2023.
- [14] Xu et al., "UniMatch: Unifying Flow, Stereo and Depth Estimation," *IEEE TPAMI*, 2023.
- [15] Ling et al., "DL3DV-10K," *CVPR*, 2024.
- [28] Heo et al., "Rotary Position Embedding for Vision Transformer," *ECCV*, 2024.

> 💡 **정확도 고지**: 본 답변은 제공된 PDF 원문에 명시된 내용만을 기반으로 작성하였습니다. 8-2절의 "관련 최신 연구 비교 분석" 중 PRoPE 이후 발표된 후속 연구(2025년 하반기 이후)에 대한 내용은 원문에 포함되지 않아 서술하지 않았습니다. 검증되지 않은 추측성 정보는 제외하였습니다.
