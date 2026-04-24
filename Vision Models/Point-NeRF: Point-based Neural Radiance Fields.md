# Point-NeRF: Point-based Neural Radiance Fields

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

Point-NeRF는 **신경망 3D 포인트 클라우드(neural 3D point cloud)** 를 활용하여 연속적인 방사 필드(radiance field)를 모델링함으로써, NeRF의 느린 per-scene 최적화 문제와 Deep MVS 방법의 낮은 렌더링 품질 문제를 동시에 해결한다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **포인트 기반 방사 필드 표현** | 각 포인트에 신경 특징 벡터(neural feature)와 신뢰도(confidence) 부여 |
| **빠른 초기화** | 사전 학습된 deep MVS 네트워크로 feed-forward inference를 통한 초기 포인트 클라우드 생성 |
| **포인트 성장/가지치기** | 구멍(hole)과 이상값(outlier) 자동 보정 메커니즘 |
| **범용성** | COLMAP 등 외부 포인트 클라우드와 연계 가능 |
| **효율성** | NeRF 대비 30× 빠른 학습으로 더 높은 렌더링 품질 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 방법의 한계:**

- **NeRF** (Mildenhall et al., 2020): 씬 전체를 하나의 전역 MLP로 표현 → per-scene 최적화에 수십 시간 소요, 빈 공간 샘플링 낭비
- **Deep MVS** (MVSNet 등): 기하 구조 빠르게 추정 가능하나 고품질 렌더링 어려움
- **기존 포인트 기반 렌더링** (NPBG 등): 래스터화 + 2D CNN 기반 → 흐릿한 결과물

Point-NeRF는 이 두 접근법의 장점을 결합한다.

---

### 2.2 제안하는 방법 및 수식

#### (1) 볼륨 렌더링 기본 수식

픽셀의 방사(radiance)는 레이 마칭을 통해:

$$c = \sum_{M} \tau_j (1 - \exp(-\sigma_j \Delta_j)) r_j, \quad \tau_j = \exp\left(-\sum_{t=1}^{j-1} \sigma_t \Delta_t\right) \tag{1}$$

여기서 $\tau$는 볼륨 투과율, $\sigma_j$와 $r_j$는 각 쉐이딩 포인트 $x_j$에서의 볼륨 밀도와 방사값, $\Delta_t$는 인접 샘플 간 거리이다.

#### (2) 포인트 기반 방사 필드 정의

신경 포인트 클라우드: $P = \{(p_i, f_i, \gamma_i) \mid i = 1, ..., N\}$

- $p_i$: 포인트 위치
- $f_i$: 신경 특징 벡터 (로컬 씬 내용 인코딩)
- $\gamma_i \in [0, 1]$: 포인트 신뢰도 (표면 근처 여부)

임의의 3D 위치 $x$에서 반경 $R$ 내의 $K$개 이웃 포인트로부터 밀도와 방사를 회귀:

$$(\sigma, r) = \text{Point-NeRF}(x, d, p_1, f_1, \gamma_1, ..., p_K, f_K, \gamma_K) \tag{2}$$

#### (3) 포인트별 처리 (Per-point Processing)

MLP $F$를 사용하여 각 이웃 포인트로부터 쉐이딩 위치 $x$에 대한 특징 벡터 추출:

$$f_{i,x} = F(f_i, x - p_i) \tag{3}$$

- 상대 위치 $x - p$를 사용하여 포인트 이동에 대한 **불변성(translation invariance)** 확보 → 일반화 성능 향상

#### (4) 시점 의존 방사 회귀 (View-dependent Radiance)

역거리 가중 평균으로 특징 집계:

$$f_x = \sum_i \gamma_i \frac{w_i}{\sum w_i} f_{i,x}, \quad \text{where } w_i = \frac{1}{\|p_i - x\|} \tag{4}$$

MLP $R$로 시점 방향 $d$를 고려한 방사 회귀:

$$r = R(f_x, d) \tag{5}$$

#### (5) 밀도 회귀 (Density Regression)

MLP $T$로 포인트별 밀도 계산 후 역거리 가중 집계:

$$\sigma_i = T(f_{i,x}) \tag{6}$$

$$\sigma = \sum_i \sigma_i \gamma_i \frac{w_i}{\sum w_i}, \quad w_i = \frac{1}{\|p_i - x\|} \tag{7}$$

#### (6) 포인트 위치 및 신뢰도 생성

MVSNet 기반 네트워크 $G_{p,\gamma}$로 포인트 위치와 신뢰도 예측:

$$\{p_i, \gamma_i\} = G_{p,\gamma}(I_q, \Phi_q, I_{q_1}, \Phi_{q_1}, I_{q_2}, \Phi_{q_2}, ...) \tag{8}$$

2D CNN $G_f$로 이미지 특징 추출:

$$\{f_i\} = G_f(I_q) \tag{9}$$

#### (7) 포인트 가지치기 (Pruning) — Sparsity Loss

$$\mathcal{L}_{\text{sparse}} = \frac{1}{|\gamma|} \sum_{\gamma_i} [\log(\gamma_i) + \log(1 - \gamma_i)] \tag{10}$$

신뢰도를 0 또는 1에 가깝게 강제하여 불필요한 포인트 제거 유도.

#### (8) 포인트 성장 (Growing)

레이 마칭 중 불투명도가 가장 높은 위치 탐지:

$$\alpha_j = 1 - \exp(-\sigma_j \Delta_j), \quad j_g = \arg\max_j \alpha_j \tag{11}$$

$\alpha_{j_g} > T_{\text{opacity}}$ 이고 $\epsilon_{j_g} > T_{\text{dist}}$ (가장 가까운 포인트까지의 거리)일 때 새 포인트 추가.

#### (9) 최종 최적화 손실

$$\mathcal{L}_{\text{opt}} = \mathcal{L}_{\text{render}} + a \mathcal{L}_{\text{sparse}}, \quad a = 2 \times 10^{-3} \tag{12}$$

---

### 2.3 모델 구조

```
[Multi-view Images]
        ↓
┌─────────────────────────────────┐
│  Neural Point Generation        │
│  ┌──────────┐  ┌─────────────┐  │
│  │ G_{p,γ}  │  │    G_f      │  │
│  │(MVSNet)  │  │  (VGG CNN)  │  │
│  └──────────┘  └─────────────┘  │
│   p_i, γ_i         f_i          │
└─────────────────────────────────┘
        ↓ Neural Point Cloud P = {p_i, f_i, γ_i}
┌─────────────────────────────────┐
│  Point-NeRF Representation      │
│  ┌─────┐  ┌─────┐  ┌─────┐    │
│  │MLP F│  │MLP T│  │MLP R│    │
│  └─────┘  └─────┘  └─────┘    │
│  Per-point → Density → Radiance │
└─────────────────────────────────┘
        ↓ Differentiable Ray Marching (Eq. 1)
[Rendered Image] ← L2 Loss → [Ground Truth]
        ↓
┌─────────────────────────────────┐
│  Per-scene Optimization         │
│  Point Pruning (γ_i < 0.1)      │
│  Point Growing (opacity-based)  │
└─────────────────────────────────┘
```

**네트워크 세부 구성:**
- $G_f$: VGG 기반, 3개 다운샘플링 레이어, 56채널 (8+16+32) 멀티스케일 특징
- 최종 per-point 특징: 59채널 벡터 (56 + 시점 방향 3)
- MLP $F$: 2레이어, 중간 채널 256
- MLP $R$: 3레이어, 중간 채널 128
- MLP $T$: 2레이어, 중간 채널 256
- 각 쉐이딩 위치당 $K=8$개 이웃 포인트 쿼리

---

### 2.4 성능 향상

#### DTU Dataset (Table 1)

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ | 학습 시간 |
|------|--------|--------|---------|----------|
| PixelNeRF | 19.31 | 0.789 | 0.382 | - |
| MVSNeRF | 26.63 | 0.931 | 0.168 | - |
| IBRNet | 26.04 | 0.917 | 0.190 | - |
| **Ours (1K)** | **28.43** | **0.929** | **0.183** | **2min** |
| **Ours (10K)** | **30.12** | **0.957** | **0.117** | **20min** |
| NeRF | 27.01 | 0.902 | 0.263 | 10h |

#### NeRF Synthetic Dataset (Table 2)

| 방법 | PSNR↑ | SSIM↑ | LPIPS_Alex↓ |
|------|--------|--------|-------------|
| NeRF | 31.01 | 0.947 | - |
| NSVF | 31.75 | 0.964 | 0.047 |
| IBRNet | 28.14 | 0.942 | - |
| **Point-NeRF $\text{20K}$** | **30.71** | **0.967** | **0.050** |
| **Point-NeRF $\text{200K}$** | **33.31** | **0.978** | **0.027** |

Point-NeRF $\text{20K}$은 단 40분 최적화로 NeRF의 20+ 시간 결과에 근접하며, Point-NeRF $\text{200K}$은 NSVF를 포함한 모든 비교 방법을 능가한다.

---

### 2.5 한계점

논문에서 명시된 한계:

1. **렌더링 속도 미최적화**: 포인트 쿼리 및 특징 집계 연산이 실시간 렌더링에 최적화되지 않음 (NeRF 대비 약 3× 빠르지만 추가 최적화 가능)

2. **배경 없는 장면(unbounded scenes) 처리 불가**: 로컬 방사 표현의 특성상 배경 처리를 위한 추가 컴포넌트 필요 (예: Plenoxels의 background NeRF)

3. **MVSNet의 배경 처리 한계**: MVSNet 기반 포인트 클라우드는 배경 영역에 이상값 포인트를 과도하게 생성할 수 있어 필터링 필요

4. **특징 추출기의 단순성**: 간단한 VGG 기반 특징 추출기 사용으로 IBRNet의 복잡한 분산 기반 특징 추출 대비 직접 추론 품질이 낮음

5. **불완전한 메쉬 초기화 취약성**: ScanNet 실험에서 Scene 101의 메쉬가 극도로 불완전한 경우 품질 저하 관찰

---

## 3. 일반화 성능 향상 관련 심층 분석

Point-NeRF의 일반화 성능은 여러 설계 요소를 통해 달성되며, 이는 논문의 핵심 강점 중 하나이다.

### 3.1 일반화를 가능하게 하는 핵심 메커니즘

#### (a) 상대 위치 기반 로컬 함수

수식 (3)에서 $f_{i,x} = F(f_i, x - p_i)$ 와 같이 **절대 좌표가 아닌 상대 위치** $x - p_i$를 사용한다. 이는 네트워크가 포인트의 절대 위치에 의존하지 않고 로컬 기하 구조를 학습하게 하여, 훈련 시 보지 못한 씬에도 적용 가능한 이동 불변성(translation invariance)을 제공한다.

#### (b) DTU에서 타 데이터셋으로의 Cross-domain 일반화

모델은 **오직 DTU 데이터셋만으로 학습**되었음에도 불구하고:

- **NeRF Synthetic**: 완전히 다른 카메라 배치(360° 분포) → 성공적 일반화
- **Tanks & Temples**: 대규모 실외 씬 → NSVF 대비 PSNR +1.21dB 향상
- **ScanNet**: 대규모 실내 씬 → NSVF 대비 PSNR +4.84dB 향상

이는 포인트 기반 로컬 표현이 씬 타입에 관계없이 공통적인 로컬 기하-외관 관계를 학습한다는 것을 시사한다.

#### (c) 임의 개수의 뷰 처리 가능

MVSNeRF는 **정확히 3개의 소기준선 이미지**만 입력으로 받는 고정 구조인 반면, Point-NeRF는 **임의 개수의 뷰**에서 신경 포인트를 융합할 수 있다. 이는 다양한 촬영 설정에 대한 유연한 일반화를 가능하게 한다.

#### (d) 멀티스케일 특징 추출

$G_f$의 세 해상도 레이어에서 추출한 멀티스케일 특징(8+16+32 채널)이 다양한 크기의 씬 구조를 포착한다.

#### (e) 포인트 성장/가지치기의 도메인 적응 역할

초기 포인트 클라우드의 품질이 낮더라도 (예: 구멍이 많은 COLMAP 포인트), 성장/가지치기 메커니즘이 씬별 최적화 과정에서 자동으로 기하 구조를 보완한다:

- 1000개의 희박한 COLMAP 포인트에서 시작해도 완전한 표면 커버리지 달성 (Fig. 5)
- COLMAP 포인트 기반 최종 결과: PSNR 31.77, SSIM 0.973 (NeRF 31.01, 0.947 초과)

### 3.2 일반화의 한계

- **배경 없는 씬**: 외부 배경 모델 없이는 unbounded 씬 처리 어려움
- **극단적으로 불완전한 초기화**: 매우 불완전한 메쉬 초기화 시 성능 저하 (ScanNet Scene 101 사례)
- **특징 추출기의 도메인 한계**: VGG 기반 단순 특징 추출기는 복잡한 텍스처 씬에서 한계

---

## 4. 최신 연구 비교 분석 (2020년 이후)

Point-NeRF의 위상을 파악하기 위해 동시대 및 이후 관련 연구들을 비교한다.

### 4.1 Point-NeRF와 주요 NeRF 계열 방법 비교

| 방법 | 표현 방식 | 초기화 | 일반화 | 학습 속도 | 렌더링 품질 |
|------|-----------|--------|--------|----------|------------|
| **NeRF** (2020) | 전역 MLP | Per-scene | ✗ | 20+h | 높음 |
| **NSVF** (2020) | Sparse Voxel | Per-scene | ✗ | 긴 시간 | 높음 |
| **PixelNeRF** (2021) | 2D 이미지 특징 | Feed-forward | ✓ | 빠름 | 낮음 |
| **IBRNet** (2021) | 2D 이미지 특징 | Feed-forward+fine-tune | ✓ | 중간 | 중간 |
| **MVSNeRF** (2021) | Voxel (로컬) | Feed-forward | 제한적 | 빠름 | 중간 |
| **Point-NeRF** (2022) | 3D 신경 포인트 | Feed-forward+fine-tune | ✓ | 20-40min | 최고 |

### 4.2 Point-NeRF 이후 발전한 관련 연구

아래는 논문 발표 이후 등장한 관련 연구들이며, Point-NeRF의 영향을 받거나 유사한 방향으로 발전한 것들이다. 단, 이 항목들은 본 논문(arXiv:2201.08845)에 직접 인용되지 않은 이후 연구들이므로, 해당 내용의 구체적인 수치 등은 별도의 원문 확인이 필요하다.

**3D Gaussian Splatting** (Kerbl et al., 2023)
- 3D 가우시안을 장면 표현으로 사용, 실시간 렌더링 달성
- Point-NeRF와 유사하게 포인트 기반 표현을 활용하지만, 볼륨 렌더링 대신 차별화 가능한 래스터화 사용
- Point-NeRF의 포인트 클라우드 초기화 아이디어와 유사한 MVS 기반 초기화 사용

**Instant-NGP** (Müller et al., 2022, SIGGRAPH)
- 해시 기반 다중 해상도 특징 그리드로 수십 초 내 학습 달성
- Point-NeRF의 공간 효율적 샘플링과 보완적 관계

**이 두 연구들은 Point-NeRF의 핵심 통찰—표면 근처에만 샘플링, 명시적 3D 구조 활용—과 맥을 같이 하며 발전하였다.**

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 연구에 미치는 영향

#### (a) 포인트 기반 신경 렌더링의 패러다임 확립

Point-NeRF는 명시적 3D 구조(포인트 클라우드)와 암시적 신경 표현(MLP)의 결합이 효과적임을 증명하였다. 이는 이후 3D Gaussian Splatting 등 명시적 구조 기반 방법들의 발전에 방향을 제시했다.

#### (b) 일반화 가능한 NeRF 연구의 촉진

DTU 훈련 → 다양한 도메인 테스트의 성공 사례는, 씬 일반화 가능한 신경 렌더링 연구의 실현 가능성을 보여주었다. 특히 상대 위치 기반 로컬 처리가 도메인 일반화의 핵심 요소임을 실증했다.

#### (c) 포인트 클라우드와 신경 렌더링의 통합

COLMAP 등 전통적 3D 재구성 방법과 신경 렌더링의 결합 가능성을 보여줌으로써, 기존 컴퓨터 비전 파이프라인과의 통합 연구를 자극했다.

#### (d) 효율적 씬별 최적화 전략

Feed-forward 초기화 + 짧은 per-scene fine-tuning의 2단계 전략은 이후 generalizable NeRF 연구들의 표준적 프레임워크가 되었다.

### 5.2 앞으로 연구 시 고려해야 할 점

#### (a) 실시간 렌더링 최적화

논문 자체에서 인정하듯, 포인트 쿼리 및 특징 집계 연산의 최적화가 미완성 상태이다. KiloNeRF, PlenOctrees 등의 가속화 기법과 Point-NeRF를 결합하는 연구가 유망하다.

#### (b) 동적 씬으로의 확장

현재 Point-NeRF는 정적 씬만 처리한다. 시간적으로 변화하는 포인트 클라우드를 모델링하는 Dynamic Point-NeRF 연구가 필요하다.

#### (c) 배경 처리 (Unbounded Scenes)

실세계 적용을 위해서는 무한 배경을 처리하는 메커니즘이 필요하다. NeRF++나 Mip-NeRF 360의 배경 처리 기법과의 결합을 고려해야 한다.

#### (d) 더 강력한 특징 추출기

VGG 기반의 단순한 특징 추출기를 트랜스포머 기반 혹은 다른 더 강력한 feature extractor로 교체하면 초기 추론 품질을 크게 향상시킬 수 있다. 단, 메모리 및 연산 비용과의 트레이드오프를 고려해야 한다.

#### (e) 포인트 클라우드 품질 의존성 완화

초기 포인트 클라우드 품질에 따라 최적화 시간과 최종 품질이 달라진다. 더 강건한 초기화 방법이나 성장/가지치기 메커니즘의 개선이 필요하다.

#### (f) 공정한 비교를 위한 실험 설정 표준화

논문 appendix에서 언급되듯, MVSNet 배경 필터링 방식(alpha channel vs. background color)에 따라 수치가 달라질 수 있다. 향후 연구에서는 이러한 실험 설정을 명확히 해야 한다.

#### (g) 대규모 씬 확장성

현재 아키텍처는 대규모 야외 씬에서 포인트 클라우드 크기와 메모리 관리 문제가 발생할 수 있다. 계층적 포인트 클라우드 관리나 스트리밍 기반 접근법이 필요하다.

---

## 참고 자료

1. **Xu, Q., Xu, Z., Philip, J., Bi, S., Shu, Z., Sunkavalli, K., & Neumann, U. (2022).** "Point-NeRF: Point-based Neural Radiance Fields." *arXiv:2201.08845v7*. (본 논문 원문 PDF)

2. **Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020).** "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV 2020.* [논문 내 참고문헌 35]

3. **Yao, Y., Luo, Z., Li, S., Fang, T., & Quan, L. (2018).** "MVSNet: Depth Inference for Unstructured Multi-View Stereo." *ECCV 2018.* [논문 내 참고문헌 59]

4. **Yu, A., Ye, V., Tancik, M., & Kanazawa, A. (2021).** "PixelNeRF: Neural Radiance Fields from One or Few Images." *CVPR 2021.* [논문 내 참고문헌 63]

5. **Wang, Q., Wang, Z., Genova, K., Srinivasan, P., Zhou, H., Barron, J. T., Martin-Brualla, R., Snavely, N., & Funkhouser, T. (2021).** "IBRNet: Learning Multi-View Image-Based Rendering." *CVPR 2021.* [논문 내 참고문헌 53]

6. **Chen, A., Xu, Z., Zhao, F., Zhang, X., Xiang, F., Yu, J., & Su, H. (2021).** "MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo." *arXiv:2103.15595.* [논문 내 참고문헌 8]

7. **Liu, L., Gu, J., Lin, K. Z., Chua, T. S., & Theobalt, C. (2020).** "Neural Sparse Voxel Fields." *arXiv:2007.11571.* [논문 내 참고문헌 29]

8. **Aliev, K. A., Sevastopolsky, A., Kolos, M., Ulyanov, D., & Lempitsky, V. (2020).** "Neural Point-Based Graphics." *ECCV 2020.* [논문 내 참고문헌 2]

9. **Schönberger, J. L., Zheng, E., Pollefeys, M., & Frahm, J. M. (2016).** "Pixelwise View Selection for Unstructured Multi-View Stereo." *ECCV 2016.* (COLMAP) [논문 내 참고문헌 44]

10. **Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017).** "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation." *CVPR 2017.* [논문 내 참고문헌 40]
