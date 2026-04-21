# 3D Gaussian Splatting with Deferred Reflection

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
본 논문은 **Deferred Shading(지연 셰이딩)** 파이프라인을 3D Gaussian Splatting(3DGS)에 접목하여, 기존 방법 대비 고품질의 정반사(specular reflection)를 **실시간에 가까운 속도(~80 FPS)**로 렌더링할 수 있음을 주장합니다.

### 주요 기여 (4가지)

| 기여 | 설명 |
|------|------|
| **Deferred Shading 통합** | 픽셀 단위 반사 계산으로 Gaussian 경계 아티팩트 제거 |
| **Normal Propagation** | 올바른 법선 벡터를 인접 Gaussian에 점진적으로 전파 |
| **Color Sabotage** | 미반사 Gaussian의 색상 과적합 방지로 반사 영역 탐색 촉진 |
| **효율성 유지** | 훈련 47분, 렌더링 80 FPS (vanilla 3DGS 대비 거의 동일) |

---

## 2. 상세 기술 분석

### 2.1 해결하고자 하는 문제

**기존 3DGS의 한계:**
- 3DGS는 Spherical Harmonics(SH) 함수로 뷰 의존적 색상을 표현하지만, SH의 주파수 한계로 **고주파 정반사 표현 불가**
- 반사를 표현하려면 Gaussian을 분열(split)시켜 기하학적 품질 저하 발생
- 환경 맵 반사 모델은 정확한 법선 벡터를 필요로 하나, 불연속적인 그래디언트로 법선 추정이 어려움

$$\frac{\partial \mathcal{L}}{\partial N} = \frac{\partial \mathcal{L}}{\partial E} \frac{\partial E}{\partial N}$$

여기서 $E$가 텍스처 쿼리이므로, $\frac{\partial E}{\partial N}$의 비영(non-zero) 성분은 bilinear 필터 4개 텍셀에만 한정 → **이미 정확한 법선 근처에서만 의미 있는 그래디언트 발생**

---

### 2.2 제안하는 방법 및 수식

#### 렌더링 파이프라인 (2-Pass)

**Pass 1: Gaussian Splatting Pass**

기존 3DGS의 색상 블렌딩:

$$C(\mathbf{v}) = \sum_i c_i(\mathbf{v}) G(\Theta_i, \mathbf{v}) \tag{1}$$

추가로 법선 맵 $N(\mathbf{v})$과 반사 강도 맵 $R(\mathbf{v})$를 동일 가중치 $G$로 블렌딩:

$$N(\mathbf{v}) = \sum_i n_i G(\Theta_i, \mathbf{v}), \quad R(\mathbf{v}) = \sum_i r_i G(\Theta_i, \mathbf{v}) \tag{2}$$

- $n_i$: 각 Gaussian 타원체의 **최단 축** (법선 벡터로 해석, 카메라 방향으로 flip)
- $r_i$: 각 Gaussian의 **반사 강도** (학습 가능한 스칼라 파라미터)

**Pass 2: Deferred Reflection Pass**

최종 픽셀 색상 합성:

$$C'(\mathbf{v}) = (1 - R(\mathbf{v}))C(\mathbf{v}) + R(\mathbf{v}) E\!\left(\frac{2\mathbf{v} \cdot N(\mathbf{v})N(\mathbf{v})}{\|N(\mathbf{v})\|} - \mathbf{v}\right) \tag{3}$$

- $E(\cdot)$: bilinear 필터로 쿼리되는 **학습 가능한 환경 맵**
- 반사 방향: $\mathbf{d}_{refl} = \frac{2\mathbf{v} \cdot N(\mathbf{v}) N(\mathbf{v})}{\|N(\mathbf{v})\|} - \mathbf{v}$

#### 손실 함수

$$\mathcal{L} = (1 - \lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{D\text{-}SSIM}, \quad \lambda = 0.2 \tag{4}$$

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────────────┐
│                      3D Gaussians                           │
│  파라미터: x, Σ, SH, α, r (반사강도)                         │
└─────────────────┬───────────────────────────────────────────┘
                  │ Splatting Pass
                  ▼
┌──────────────────────────────────────────────────┐
│  Screen-Space Maps                               │
│  • Normal Map N(v)      (최단 축 블렌딩)          │
│  • Refl. Strength R(v)  (반사 강도 블렌딩)        │
│  • Base Color C(v)      (SH 색상 블렌딩)          │
└──────────────────┬───────────────────────────────┘
                   │ Deferred Shading Pass
                   ▼
┌──────────────────────────────────────────────────┐
│  Environment Map Query                           │
│  반사 방향 계산 → E(refl_dir) → Refl. Color      │
└──────────────────┬───────────────────────────────┘
                   │ Blending: C'(v) = (1-R)C + R·E
                   ▼
              최종 렌더링 결과 → Image Loss
```

#### 훈련 전략 (3단계)

1. **View-Independent Bootstrap**: $r_i = 0$, SH 0차(상수 색상)만 사용. 기본 기하 구조 학습
2. **Reflection Training + Normal Propagation**:
   - $r_i > 0.1$인 Gaussian의 법선을 인근 Gaussian으로 전파
   - 반사 Gaussian의 두 장축을 **1.5배 확대**, 불투명도를 ≥0.9로 설정
   - **Color Sabotage**: $r_i \leq 0.1$인 Gaussian에 ±10% 색상 노이즈 추가
3. **High-Order SH 학습**: 반사 표면 수렴 후 고차 SH 계수 학습

---

### 2.4 성능 향상

#### 정량적 결과 (Shiny Blender Dataset - ball 장면 예시)

| 방법 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FPS | 훈련 시간 |
|------|--------|--------|---------|-----|---------|
| Ref-NeRF | 33.16 | 0.971 | 0.166 | 0.06 | 19h |
| 3DGS | 27.65 | 0.937 | 0.162 | 277 | 6min |
| GShader | 30.99 | 0.966 | 0.121 | 51 | 60min |
| **Ours** | **33.66** | **0.979** | **0.098** | **251** | **16min** |

#### 법선 재구성 품질 (MAE°, Shiny Blender Dataset 평균)

| 방법 | MAE° ↓ | LPIPS ↓ |
|------|--------|---------|
| GShader | 22.31 | 0.621 |
| NVDiffRec | 17.02 | 0.636 |
| ENVIDR | **4.618** | 0.615 |
| **Ours** | 4.871 | **0.511** |

#### Ablation Study (ball 장면 기준)

| 설정 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Full (Ours) | **33.66** | **0.979** | **0.098** |
| w/o propagation | 27.85 | 0.938 | 0.159 |
| w/o sabotage | 30.00 | 0.959 | 0.128 |

---

### 2.5 한계

1. **단일 레이어 반사 제한**: 전통적 Deferred Shading 특성상 픽셀당 최대 1개 반사 레이어만 처리 → 투명 창문(자동차 유리) 처리 불일관성
2. **오목 장면에서 느린 수렴**: 오목한 물체(bell 장면)에서 법선 전파가 비효율적
3. **완전한 역방향 렌더링 미지원**: 기하-조명-재질 완전 분리 불가 (거친 반사, 이방성/다층 재질, 전역 조명 미지원)
4. **배경 간섭**: 제한된 시점에서만 촬영된 배경 객체가 반사 객체처럼 동작하여 환경 맵 학습 방해

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 관련 설계 요소

**긍정적 요소:**
- **구형 도메인 $M$ 제한**: 실제 장면에서 전경 객체만을 대상으로 반사 처리하여, 배경의 노이즈 간섭 최소화
- **Color Sabotage의 일반화 효과**: 색상 과적합 방지 메커니즘이 새로운 뷰에서도 반사 표현의 견고성을 높임
- **환경 맵 분리 학습**: Gaussian으로부터 분리된 환경 맵은 조명 조건 변화에 대한 적응 가능성 제공
- **비반사 장면에서의 회귀 안정성**: NeRF Synthetic 데이터셋에서 vanilla 3DGS와 거의 동일한 성능 유지 (특수화 없이도 일반 장면 적용 가능)

**부정적 요소:**
- **장면별 환경 맵 학습**: 환경 맵이 각 장면마다 독립적으로 학습되어, 미지의 조명 환경에 대한 **제로샷 일반화 불가**
- **법선 전파의 장면 의존성**: Normal propagation 전략이 반사 강도 $r_i > 0.1$이라는 임계값에 의존하므로, 약한 반사 또는 복잡한 BRDF를 가진 장면에서 일반화 어려움
- **단일 장면 최적화**: 다른 3DGS 기반 방법들처럼 scene-specific 최적화이므로, feed-forward 일반화 모델 대비 새 장면 적용 시 처음부터 재학습 필요

### 3.2 일반화 향상을 위한 가능성

**단기적 가능성:**
- **사전 학습된 법선 추정 모델 통합**: OmniData [Eftekhar et al., 2021], DSINE [Bae et al., 2024] 등과 결합하여 법선 초기화 품질 향상 → 법선 전파의 시작점 품질 개선
- **거칠기(Roughness) 파라미터 추가**: 현재는 완전 거울 반사만 처리하지만, Disney BRDF 모델 등을 통합하면 glossy 소재까지 일반화 가능

**중장기적 가능성:**
- **Generalizable 3DGS 프레임워크와의 결합**: pixelSplat [Charatan et al., 2024], MVSplat [Chen et al., 2024] 등 feed-forward 3DGS 방법에 deferred reflection 모듈 통합
- **데이터 기반 환경 맵 선행 학습**: HDRI 데이터베이스로 환경 맵 prior를 학습하면 소수 뷰에서도 안정적인 조명 추정 가능
- **스크린 스페이스 반사(SSR) 통합**: 논문 저자가 직접 언급한 방향으로, 환경 맵을 넘어선 다중 반사 표현 가능

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 비교 표

| 논문 | 방법 | 반사 처리 | FPS | 일반화 | 역방향 렌더링 |
|------|------|---------|-----|--------|------------|
| **NeRF** [Mildenhall et al., 2020] | MLP implicit | ❌ (SH 한계) | <1 | ❌ | ❌ |
| **Ref-NeRF** [Verbin et al., 2022] | 반사 방향 재파라미터화 | ✅ | ~0.06 | ❌ | △ |
| **NPC** [Kopanas et al., 2022] | 가상 포인트 catacaustics | ✅ | ~1 | ❌ | ❌ |
| **Instant-NGP** [Müller et al., 2022] | 해시 인코딩 | △ | ~30 | ❌ | ❌ |
| **3DGS** [Kerbl et al., 2023] | SH per-Gaussian | ❌ | ~84 | ❌ | ❌ |
| **GaussianShader** [Jiang et al., 2023] | Forward shading per-Gaussian | ✅ | ~31 | ❌ | ✅ |
| **ENVIDR** [Liang et al., 2023] | SDF + Neural rendering | ✅ | ~3 | ❌ | ✅ |
| **NeRO** [Liu et al., 2023] | SDF + BRDF | ✅ | <1 | ❌ | ✅ |
| **3DGS-DR (Ours)** [Ye et al., 2024] | Deferred shading | ✅ | ~80 | △ | △ |

### 4.2 핵심 비교 분석

**vs. Ref-NeRF [Verbin et al., 2022]:**
- Ref-NeRF는 반사 방향으로의 radiance 재파라미터화로 반사 표현 개선 시도
- 그러나 NeRF의 volumetric 특성상 훈련 시간 극도로 길고(~19h), 렌더링 속도 0.06 FPS로 실시간 불가
- 본 논문은 유사한 수준의 PSNR(ball 기준 33.66 vs 33.16)을 4000배 이상 빠른 속도로 달성

**vs. GaussianShader [Jiang et al., 2023]:**
- 가장 직접적인 경쟁 방법으로 동일한 3DGS 기반
- GShader는 **Gaussian별** 반사 계산(forward shading) → 픽셀 정밀도 부족, Gaussian 경계 아티팩트 발생
- 본 논문은 **픽셀별** 반사 계산(deferred shading) → 동일 비용으로 더 많은 반사 샘플 생성, 그래디언트 안정화

$$\text{반사 샘플 수 (GShader)} = N_{Gaussian}, \quad \text{반사 샘플 수 (Ours)} = H \times W$$

- GShader 대비 PSNR +2.67dB (ball), 훈련 시간 1/4, FPS 2.6배 향상

**vs. ENVIDR [Liang et al., 2023]:**
- SDF 기반 완전 역방향 렌더링 가능하지만, SDF의 내재적 평활성으로 **세부 기하 손실**
- 훈련 ~3.2h vs 본 논문 16min, FPS 3 vs 251
- 법선 MAE° 거의 동등(4.618 vs 4.871)하지만 환경 맵 품질은 본 논문이 우수 (LPIPS 0.615 vs 0.511)

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5.1 연구에 미치는 영향

**① Gaussian-based 렌더링 패러다임 확장**

본 논문은 3DGS의 **표현력-속도 트레이드오프**를 개선하는 방향을 제시합니다. Deferred shading 아이디어는 단순히 반사에 그치지 않고, Gaussian splatting에서 다양한 고주파 효과(그림자, Ambient Occlusion, 간접 조명 등)를 처리하는 일반적 프레임워크로 확장될 수 있습니다.

**② 법선 추정 방법론의 혁신**

Normal Propagation은 순수 색상 손실만으로 법선 벡터를 안정적으로 학습하는 독창적 메커니즘입니다. 이 아이디어는:
- Gaussian 기반 동적 장면 재구성에서 법선 추정 개선
- Few-shot 3D 재구성에서 기하 품질 향상
에 활용될 수 있습니다.

**③ 실시간 고품질 렌더링의 실용화**

AR/VR, 자율주행 시뮬레이터, 영화 제작 파이프라인에서 실시간 반사 렌더링의 실용적 적용 가능성을 제시합니다.

**④ 역방향 렌더링과의 연결 다리**

완전한 역방향 렌더링의 과제(under-constrained optimization)를 우회하면서도 고품질 결과를 달성하는 **"실용적 절충안"**을 제시하여, 역방향 렌더링 커뮤니티에 새로운 관점을 제공합니다.

---

### 5.2 앞으로 연구 시 고려해야 할 점

**① 거칠기(Roughness) 및 BRDF 일반화**

현재 모델은 완전 거울 반사($\delta$-function BRDF)만 처리합니다. 물리 기반 렌더링 방정식:

$$L_o(\mathbf{v}) = \int_{\Omega} f_r(\mathbf{v}, \mathbf{l}) L_i(\mathbf{l}) (\mathbf{n} \cdot \mathbf{l}) d\mathbf{l}$$

에서 거칠기에 따른 BRDF $f_r(\mathbf{v}, \mathbf{l})$ 처리(Prefiltered Environment Map, Split-Sum Approximation 등)를 deferred shading과 통합하면 glossy 소재까지 일반화 가능합니다.

**② 다층 반사 및 투명 물체**

현재 단일 레이어 deferred shading의 한계를 극복하기 위한 연구 방향:
- **OIT(Order-Independent Transparency)** 기법과의 결합
- **다중 G-buffer** 레이어 활용

**③ 법선 전파의 수학적 정형화**

현재 Normal Propagation은 경험적 규칙(opacity ≥ 0.9, 축 확장 1.5배)에 의존합니다. 이를 **정보 이론적 관점**이나 **그래프 기반 전파 모델**로 정형화하면 더 안정적인 학습이 가능합니다.

**④ Feed-Forward 일반화 모델로의 확장**

pixelSplat [Charatan et al., 2024], MVSplat [Chen et al., 2024] 등 generalizable 3DGS 방법에 deferred reflection 모듈을 통합하여 **단일 추론으로 새 장면에 적용 가능한 반사 렌더링** 모델 개발이 유망합니다.

**⑤ 동적 장면 및 시간적 일관성**

현재는 정적 장면만 대상으로 합니다. 동적 장면으로 확장 시 Gaussian의 변형과 반사 강도 $r_i$, 법선 $n_i$의 **시간적 일관성 유지**가 중요한 연구 과제입니다.

**⑥ 전역 조명 효과와의 통합**

현재 환경 맵 기반 반사는 **무한 원거리 조명**만 가정합니다. 근거리 광원, 상호 반사(inter-reflection), Caustics 등의 처리를 위한 확장이 필요하며, 스크린 스페이스 반사(SSR)나 하드웨어 레이 트레이싱과의 통합이 유망한 방향입니다.

---

## 참고 자료

**주요 논문 (본 문서의 직접 분석 대상):**
- Ye, K., Hou, Q., & Zhou, K. (2024). **3D Gaussian Splatting with Deferred Reflection**. SIGGRAPH Conference Papers '24. arXiv:2404.18454v2

**비교 대상 논문:**
- Mildenhall, B. et al. (2020). **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**. ECCV 2020.
- Verbin, D. et al. (2022). **Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields**. CVPR 2022.
- Kerbl, B. et al. (2023). **3D Gaussian Splatting for Real-Time Radiance Field Rendering**. ACM Trans. Graph. 42(4).
- Jiang, Y. et al. (2023). **GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces**. arXiv:2311.17977.
- Liang, R. et al. (2023). **ENVIDR: Implicit Differentiable Renderer with Neural Environment Lighting**. ICCV 2023.
- Liu, Y. et al. (2023). **NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images**. ACM Trans. Graph. 42(4).
- Kopanas, G. et al. (2022). **Neural Point Catacaustics for Novel-View Synthesis of Reflections**. ACM Trans. Graph. 41(6).
- Müller, T. et al. (2022). **Instant Neural Graphics Primitives with a Multiresolution Hash Encoding**. ACM Trans. Graph. 41(4).
- Munkberg, J. et al. (2022). **Extracting Triangular 3D Models, Materials, and Lighting from Images (NVDiffRec)**. CVPR 2022.

**소스 코드:** https://github.com/gapszju/3DGS-DR
