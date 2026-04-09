# DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

DNGaussian은 **희소 뷰(Sparse-View) 환경에서 3D Gaussian Splatting(3DGS)의 기하학적 열화(geometry degradation) 문제**를 깊이(depth) 정규화를 통해 해결하고, 실시간(300 FPS) 고품질 새로운 뷰 합성(Novel View Synthesis)을 낮은 비용으로 달성할 수 있다고 주장합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **Hard & Soft Depth Regularization** | Gaussian primitives의 위치(center)와 불투명도(opacity)를 독립적으로 조정하여 형상 정보를 보존하면서 기하학 복원 |
| **Global-Local Depth Normalization** | 패치 기반 정규화로 소규모 국소 깊이 변화에 집중하는 스케일-불변 손실 함수 |
| **DNGaussian 프레임워크** | 위 두 기법을 결합한 통합 프레임워크, 훈련 시간 25× 단축, 렌더링 속도 3000× 이상 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: 희소 입력 뷰에서의 기하학적 열화**

3DGS는 밀집 뷰(dense views)에서는 우수한 성능을 보이나, 희소 입력 뷰 환경에서는 뷰 제약(view constraints)이 감소하면서 Gaussian primitives의 위치가 잘못 학습되어 기하학이 크게 손상됩니다.

**문제 2: 기존 NeRF용 깊이 정규화의 3DGS 비적용성**

기존 NeRF 깊이 정규화를 3DGS에 그대로 적용하면:
- 모노큘러 깊이(monocular depth)가 매끄러운(smooth) 반면 색상(color)은 복잡하여, 형상 파라미터가 깊이에 과적합(overfit)되어 **흐릿한 외관(blurry appearance)** 발생
- NeRF는 연속적 밀도(density)를 다루지만, 3DGS는 **이산적(discrete)이고 유연한(flexible) primitive** 구조로 본질적으로 다름

**문제 3: 소규모 깊이 오류 무시**

기존 스케일-불변 손실은 고정된 전역 스케일로 정렬하여 소규모 국소 깊이 변화를 간과하며, 이는 복잡한 텍스처 영역에서 Gaussian primitives의 노이즈 분포를 초래합니다.

---

### 2.2 제안 방법 (수식 포함)

#### 2.2.1 3D Gaussian Splatting 기초

**Gaussian Primitive 기저 함수:**

$$\mathcal{G}_i(x) = e^{-\frac{1}{2}(x - \mu_i)^T \Sigma_i^{-1} (x - \mu_i)} \tag{1}$$

여기서 공분산 행렬 $\Sigma$는 스케일 $s$와 회전 쿼터니언 $q$로부터 계산됩니다.

**픽셀별 색상 렌더링:**

$$\mathcal{C}(x_p) = \sum_{i \in N} c_i \tilde{\alpha}_i \prod_{j=1}^{i-1}(1 - \tilde{\alpha}_j) \tag{2}$$

**렌더링 불투명도:**

$$\tilde{\alpha}_i = \alpha_i \mathcal{G}_i^{proj}(x_p) \tag{3}$$

**픽셀별 깊이 렌더링:**

$$\mathcal{D}(x_p) = \sum_{i \in N} ||\mu_i - o||_2 \times \tilde{\alpha}_i \prod_{j=1}^{i-1}(1 - \tilde{\alpha}_j) \tag{4}$$

---

#### 2.2.2 형상 고정 (Shape Freezing)

장면 기하학의 핵심은 Gaussian primitives의 **위치 분포**에 있으므로, 깊이 정규화 중 스케일링 $s$와 회전 $q$를 **동결(freeze)**하고, center $\mu$와 opacity $\alpha$만 최적화합니다.

---

#### 2.2.3 Hard Depth Regularization

모든 Gaussian에 큰 불투명도 값 $\tau$를 수동 적용하여, 카메라 중심 $o$에서 픽셀 $x_p$를 통과하는 광선 상의 **가장 가까운 Gaussian으로 구성된 "하드 깊이"** 렌더링:

$$\mathcal{D}_{hard}(x_p) = \sum_{i \in N} \tau(1-\tau)^{i-1} \mathcal{G}_i^{proj}(x_p)||\mu_i - o||_2 \tag{5}$$

정규화 손실 (center $\mu$만 최적화):

$$\mathcal{R}_{hard}(\mathcal{P}) = \mathcal{L}_{similar}(\mathcal{D}_{hard}(\mathcal{P}), \tilde{\mathcal{D}}(\mathcal{P})) \tag{6}$$

---

#### 2.2.4 Soft Depth Regularization

Gaussian center $\mu$를 $\check{\mu}$로 동결(frozen)하고, opacity $\alpha$만 조정하여 **실제 렌더링 깊이의 정확성** 보장 (표면의 반투명화 및 빈 공간 방지):

$$\mathcal{D}_{soft}(x_p) = \sum_{i \in N} ||\check{\mu}_i - o||_2 \times \tilde{\alpha}_i \prod_{j=1}^{i-1}(1-\tilde{\alpha}_j) \tag{7a}$$

$$\mathcal{R}_{soft}(\mathcal{P}) = \mathcal{L}_{similar}(\mathcal{D}_{soft}(\mathcal{P}), \tilde{\mathcal{D}}(\mathcal{P})) \tag{7b}$$

---

#### 2.2.5 Global-Local Depth Normalization

**Local Depth Normalization (LN):**

패치 $\mathcal{P}$ 내 깊이를 평균 0, 표준편차 ≈ 1로 정규화:

$$\mathcal{D}^{LN}(x) = \frac{\mathcal{D}(x) - \text{mean}(\mathcal{D}(\mathcal{P}))}{\text{std}(\mathcal{D}(\mathcal{P})) + \epsilon}, \quad \text{s.t.} \quad x \in \mathcal{P} \tag{8}$$

**Global Depth Normalization (GN):**

패치 평균을 빼되, 전체 이미지 $\mathcal{D}_\mathcal{I}$의 표준편차를 사용:

$$\mathcal{D}^{GN}(x) = \frac{\mathcal{D}(x) - \text{mean}(\mathcal{D}(\mathcal{P}))}{\text{std}(\mathcal{D}_\mathcal{I})}, \quad \text{s.t.} \quad x \in \mathcal{P}, \mathcal{P} \subseteq \mathcal{I} \tag{9}$$

두 정규화를 결합하면 **전역 스케일 지식 유지 + 소규모 국소 오류에 재집중** 효과를 동시에 달성합니다.

---

#### 2.2.6 전체 손실 함수

**색상 재구성 손실:**

$$\mathcal{L}_{color} = \mathcal{L}_1(\hat{\mathcal{I}}, \mathcal{I}) + \lambda \mathcal{L}_{D\text{-}SSIM}(\hat{\mathcal{I}}, \mathcal{I}) \tag{10}$$

**깊이 정규화 손실 (Hard/Soft 공통 형태):**

$$\mathcal{R}_T = \mathcal{L}_2(\mathcal{D}_T^{GN}, \tilde{\mathcal{D}}^{GN}) + \gamma \mathcal{L}_2(\mathcal{D}_T^{LN}, \tilde{\mathcal{D}}^{LN}) \tag{11}$$

여기서 $T \in \{hard, soft\}$, $\gamma = 0.1$

**최종 손실:**

$$\mathcal{L} = \mathcal{L}_{color} + \mathcal{R}_{hard} + \mathcal{R}_{soft} \tag{12}$$

---

### 2.3 모델 구조

```
희소 입력 뷰
    │
    ├──▶ [모노큘러 깊이 추정기 (DPT)]
    │         │
    │         ▼
    │    모노큘러 깊이 맵 (사전 생성)
    │
    ▼
[랜덤 초기화 3D Gaussians]
    │
    ├──▶ [색상 지도 모듈]
    │         ├─ Gaussian Splatting 렌더링
    │         ├─ Neural Color Renderer
    │         │   (Hash Encoder + 5-layer MLP)
    │         └─ 색상 손실 (L1 + D-SSIM)
    │
    └──▶ [깊이 정규화 모듈]
              ├─ Hard Depth 렌더링 (center μ 최적화)
              ├─ Soft Depth 렌더링 (opacity α 최적화)
              └─ Global-Local Depth Normalization
                    ├─ Global Norm (전체 이미지 std)
                    └─ Local Norm (패치 std)
    │
    ▼
출력 3D Gaussian Field → 실시간 Novel View Synthesis (300 FPS)
```

**Neural Color Renderer 상세:**
- 구형 조화 함수(Spherical Harmonics) 대신 **Hash Encoder + MLP** 조합 사용
- 16 레벨, 해상도 범위 16~512, 최대 크기 $2^{19}$의 해시 인코더
- 5-layer MLP (hidden dim 64)
- 추론 시: 중간 결과 저장 후 마지막 2개 MLP 레이어만 계산 → 속도 유지 (300 FPS)

---

### 2.4 성능 향상

#### 정량적 성능 (LLFF 3-뷰 설정)

| 방법 | PSNR↑ | LPIPS↓ | SSIM↑ | 학습 시간 | FPS |
|------|--------|--------|-------|----------|-----|
| FreeNeRF | 19.63 | 0.308 | 0.612 | 2.3h | 0.09 |
| SparseNeRF | 19.86 | 0.328 | 0.624 | 1.5h | 0.09 |
| 3DGS | 15.52 | 0.405 | 0.408 | ~2.7min | 280 |
| **DNGaussian** | **19.12** | **0.294** | **0.591** | **3.5min** | **300** |

- LPIPS에서 **모든 방법 중 최고 성능** 달성
- FreeNeRF 대비 학습 시간 **25× 단축**
- NeRF 기반 방법 대비 렌더링 속도 **3000× 이상 향상**
- GPU 메모리 **2GB** (FreeNeRF의 4×48GB 대비 극적 절감)

#### Blender 8-뷰 설정

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|--------|-------|--------|
| FreeNeRF | 24.259 | 0.883 | 0.098 |
| **DNGaussian** | **24.305** | **0.886** | **0.088** |

모든 메트릭에서 최고 성능 달성.

#### Ablation Study 요약 (LLFF 3-뷰)

| 설정 | PSNR↑ | LPIPS↓ | SSIM↑ |
|------|--------|--------|-------|
| AP (all-parameter) only | 18.14 | 0.354 | 0.538 |
| Hard only | 17.90 | 0.351 | 0.525 |
| Hard + Soft | 18.31 | 0.339 | 0.552 |
| Hard + Soft + Global | 18.68 | 0.331 | 0.565 |
| Hard + Soft + Local | 18.55 | 0.324 | 0.562 |
| **Full (Hard+Soft+Global+Local)** | **19.12** | **0.294** | **0.591** |

---

### 2.5 한계점

논문에서 명시한 한계:

1. **입력 뷰 수 증가 시 성능 저하**: 9-뷰 이상에서는 깊이 맵의 오류가 오히려 최적화를 방해. LLFF 9-뷰에서 DNGaussian(23.17) ≈ 3DGS†(23.21)로 이점 소실
2. **단색 평면(Solid Color Planes) 처리 어려움**: 이방성(anisotropic) Gaussian은 스파스 뷰에서 평면 표현이 어려워 ray-like 노이즈 발생
3. **정반사 영역(Specular Regions)**: 일관되지 않은 외관으로 3DGS 자체의 한계 존재
4. **빈 공간과 균열(Hollows and Cracks)**: Gaussian primitives 투영 간 빈 픽셀에서 카메라 포즈 변화 시 결함 발생
5. **카메라 포즈 사전 지식 필요**: 알려진 카메라 포즈를 가정하여 실제 적용에 제약

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능의 근거

DNGaussian은 **다양한 장면 유형에 걸친 일반화 가능성**을 다음 측면에서 입증합니다:

**(1) 다중 데이터셋 검증**
- **LLFF**: 전방향(forward-facing) 실내외 복잡 장면
- **DTU**: 물체 중심(object-centric) 다양한 재질 (반사, 봉제 인형 등)
- **Blender**: 투명/반사 재질 포함 합성 객체 (360° 서라운드 뷰)

각 데이터셋에서 경쟁력 있는 성능을 달성하여 장면 유형에 독립적인 강건함을 입증합니다.

**(2) 깊이 추정기 종류에 대한 강건성**

논문 Table 10에서:

| DPT 타입 | PSNR↑ | LPIPS↓ | SSIM↑ |
|----------|--------|--------|-------|
| dpt_hybrid_384 | 19.12 | 0.294 | 0.591 |
| dpt_large_384 | 19.03 | 0.297 | 0.590 |

성능 차이가 미미하여, **특정 깊이 추정기에 종속되지 않는 일반화** 가능성을 시사합니다.

**(3) 패치 크기 유연성**

Global-Local Depth Normalization의 패치 크기를 고정하지 않고 범위([5,17] for LLFF/Blender, [17,51] for DTU)에서 무작위 샘플링하여, 장면별 세밀 조정 없이 적용 가능합니다.

**(4) 랜덤 초기화**

COLMAP이나 SfM 등 사전 포인트 클라우드 없이 랜덤 초기화에서 시작하여, 데이터 취득 조건에 덜 민감합니다.

### 3.2 일반화 성능 향상 가능성 분석

**(A) 더 강력한 깊이 추정기 통합 가능성**

현재 DPT 기반의 모노큘러 깊이를 사용하지만, 향후 **Depth Anything V2** (Yang et al., 2024)나 **Metric3D** (Yin et al., 2023) 같은 더 정확한 추정기로 교체 시 일반화 성능이 향상될 것으로 예상됩니다. 논문이 특정 추정기에 종속되지 않는 프레임워크 설계를 채택한 점이 이를 뒷받침합니다.

**(B) 불확실성 기반 깊이 필터링**

논문 스스로 미래 연구 방향으로 제시한 "깊이 불확실성 추정을 통한 신뢰할 수 없는 감독 필터링"이 실현되면, 뷰 수가 많아지는 상황에서도 깊이 오류의 부정적 영향을 줄여 더 넓은 입력 조건에서 일반화 가능성이 높아집니다.

**(C) Cross-scene 사전훈련 모델과의 결합**

현재는 장면별(per-scene) 최적화 방식이지만, MVSNeRF, pixelNeRF 등의 **일반화 NeRF**와 결합하거나, 3DGS 기반의 사전훈련 전략(예: FSGS 참조)과 통합 시 zero-shot 또는 few-shot 일반화 성능을 크게 향상시킬 수 있습니다.

**(D) 동적 장면으로의 확장**

논문에서 언급한 Dynamic 3D Gaussians (Luiten et al., 2023)처럼 시간 차원으로 확장 시, 비디오 기반의 희소 뷰 합성으로 일반화 범위를 확대할 수 있습니다.

**(E) 다양한 입력 뷰 수 적응**

Table 11에서 3뷰→6뷰에서는 성능 향상이 지속되지만 9뷰에서는 정체됩니다. 이는 **적응형 깊이 가중치(adaptive depth weighting)** 메커니즘을 도입하여 충분한 색상 제약이 있을 때 깊이 정규화 강도를 동적으로 조절하면 해결 가능합니다.

---

## 4. 연구 영향과 미래 연구 고려사항

### 4.1 연구에 미치는 영향

**(1) 3DGS의 희소 뷰 적용 가능성 개척**

DNGaussian은 3DGS를 희소 뷰 설정에 최초로 체계적으로 적용한 연구 중 하나로, **3DGS + 깊이 정규화**라는 새로운 연구 방향을 제시합니다. 이후 FSGS, SparseGS 등의 연구가 이 방향을 계승하고 있습니다.

**(2) 효율성-품질 트레이드오프의 새로운 기준점 설정**

훈련 3.5분, 렌더링 300 FPS라는 수치는 실시간 응용 가능성을 실증하여, 로봇 공학, AR/VR, 자율주행 등 실용적 분야에서의 적용 가능성을 높입니다.

**(3) 파라미터별 차별적 정규화 개념 도입**

형상 파라미터(scale, rotation)와 위치/점유 파라미터(center, opacity)를 분리하여 정규화하는 개념은 향후 다른 3DGS 개선 연구에서 파라미터 충돌 회피 전략으로 채택될 수 있습니다.

**(4) 계층적 깊이 정규화의 일반성**

Global-Local Depth Normalization은 깊이 추정, 3D 재구성, 스테레오 매칭 등 다양한 3D 비전 태스크에서 스케일-불변 손실 함수 설계의 참조 모델이 될 수 있습니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 주요 방법 | 학습 방식 | 렌더링 속도 | 핵심 차별점 |
|------|----------|----------|------------|------------|
| **NeRF** (Mildenhall et al., 2020) | MLP + 볼륨 렌더링 | Per-scene | 매우 느림 | 기초 연구 |
| **DietNeRF** (Jain et al., 2021, ICCV) | CLIP 의미론적 일관성 | Per-scene | 느림 | 의미적 정규화 |
| **RegNeRF** (Niemeyer et al., 2022, CVPR) | 기하학/외관 정규화 | Per-scene | 느림 | 비어있는 뷰 정규화 |
| **SparseNeRF** (Wang et al., 2023, ICCV) | 깊이 랭킹 증류 | Per-scene | 매우 느림 | 상대적 깊이 순서 보존 |
| **FreeNeRF** (Yang et al., 2023, CVPR) | 주파수 정규화 | Per-scene | 매우 느림 | 고주파 신호 점진적 학습 |
| **3DGS** (Kerbl et al., 2023, SIGGRAPH) | Gaussian Splatting | Per-scene | **280+ FPS** | 실시간 고품질 렌더링 |
| **FSGS** (Zhu et al., 2023, arXiv) | 3DGS + 깊이 | Per-scene | 실시간 | 실시간 희소뷰 3DGS |
| **SparseGS** (Xiong et al., 2023, arXiv) | 3DGS + 깊이 | Per-scene | 실시간 | 360° 희소뷰 |
| **DNGaussian** (Li et al., 2024, CVPR) | Hard/Soft Depth + Global-Local Norm | Per-scene | **300 FPS** | 파라미터 분리 정규화 |

**핵심 차별점 분석:**

- SparseNeRF 대비: 동일한 깊이 감독을 사용하지만, DNGaussian은 3DGS 백본의 특성을 고려한 파라미터 분리 전략으로 더 세밀한 디테일 복원
- FreeNeRF 대비: 주파수 마스킹 대신 깊이 기반 기하학 정규화로 접근, 효율성에서 압도적 우위
- 3DGS 대비: 동일 백본에서 깊이 정규화 추가만으로 PSNR 2.66dB 이상 향상 (LLFF 기준)

---

### 4.3 앞으로의 연구 시 고려할 점

**(1) 깊이 불확실성 모델링**

모노큘러 깊이 추정의 오류를 정량화하여 신뢰도 가중 정규화를 적용하면, 특히 9-뷰 이상 환경에서의 성능 저하 문제를 완화할 수 있습니다.

$$\mathcal{R}_{weighted} = w(x) \cdot \mathcal{R}_{hard} + (1-w(x)) \cdot \mathcal{R}_{soft}$$

여기서 $w(x)$는 깊이 추정 불확실성에 기반한 가중치.

**(2) 카메라 포즈 미지(Unknown Pose) 환경 대응**

현재 방법은 알려진 카메라 포즈를 전제합니다. NOPE-NeRF, BARF 등의 방법과 통합하여 포즈 추정과 장면 재구성을 동시에 수행하는 연구가 필요합니다.

**(3) 동적 장면 확장**

4D Gaussian Splatting과 결합하여 시간적으로 변화하는 장면의 희소 뷰 재구성으로 확장 가능성이 높습니다.

**(4) 일반화 모델(Generalizable Model)로의 전환**

현재 Per-scene 최적화 방식을 벗어나, 대규모 데이터셋에서 사전훈련된 일반화 3DGS 모델(예: pixelSplat, MVSplat 방향)과 DNGaussian의 깊이 정규화 전략을 결합하면 zero-shot 희소 뷰 합성이 가능할 것입니다.

**(5) 고급 깊이 추정기 통합**

Depth Anything V2, UniDepth 등 최신 범용 깊이 추정기와의 통합 및 성능 평가가 필요합니다.

**(6) 단색 평면 및 정반사 영역 처리**

Gaussian primitive의 표현력을 확장하거나 물리 기반 재질 모델(PBR)을 통합하여 어려운 영역에 대한 처리를 개선해야 합니다.

**(7) 패치 크기 자동 적응**

현재 패치 크기를 수동으로 데이터셋별 설정하는데, 장면의 복잡도를 자동 추정하여 적응적으로 조절하는 메커니즘이 필요합니다.

---

## 참고 자료

1. **주 논문**: Li, J., Zhang, J., Bai, X., Zheng, J., Ning, X., Zhou, J., & Gu, L. (2024). "DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization." *arXiv:2403.06912v3*. https://arxiv.org/abs/2403.06912

2. **3D Gaussian Splatting**: Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics (SIGGRAPH 2023)*, 42(4):1–14.

3. **SparseNeRF**: Wang, G., Chen, Z., Loy, C. C., & Liu, Z. (2023). "SparseNeRF: Distilling Depth Ranking for Few-Shot Novel View Synthesis." *ICCV 2023*, pp. 9065–9076.

4. **FreeNeRF**: Yang, J., Pavone, M., & Wang, Y. (2023). "FreeNeRF: Improving Few-Shot Neural Rendering with Free Frequency Regularization." *CVPR 2023*, pp. 8254–8263.

5. **RegNeRF**: Niemeyer, M., Barron, J. T., Mildenhall, B., Sajjadi, M. S. M., Geiger, A., & Radwan, N. (2022). "RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs." *CVPR 2022*, pp. 5480–5490.

6. **DPT (깊이 추정기)**: Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). "Vision Transformers for Dense Prediction." *ICCV 2021*.

7. **FSGS**: Zhu, Z., Fan, Z., Jiang, Y., & Wang, Z. (2023). "FSGS: Real-time Few-Shot View Synthesis using Gaussian Splatting." *arXiv:2312.00451*.

8. **DNGaussian 공식 코드**: https://github.com/Fictionarry/DNGaussian
