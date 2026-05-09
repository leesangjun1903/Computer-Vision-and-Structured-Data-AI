# PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting

---

## 1. 핵심 주장 및 주요 기여 요약

**PUP 3D-GS (Principled Uncertainty Pruning for 3D Gaussian Splatting)** 는 사전 학습된 3D Gaussian Splatting (3D-GS) 모델로부터 불필요한 Gaussian을 제거하기 위한 **수학적으로 원리에 기반한(post-hoc) 가지치기(pruning) 파이프라인**입니다 (CVPR 2025 채택).

**핵심 주장**
- 기존 LightGaussian, EAGLES, Compact-3DGS 등은 휴리스틱(가시성·불투명도·전송도 등)에 의존하여 고압축 시 전경(foreground)의 미세 디테일이 크게 손상됩니다.
- 저자들은 **Hessian의 Fisher 근사**를 통해 Gaussian 파라미터에 대한 재구성 손실의 민감도(sensitivity)를 계산하고, 이를 가지치기 기준으로 사용함으로써 더 원리적이고 효과적인 압축이 가능하다고 주장합니다.

**3대 기여**
1. 사전 학습된 3D-GS 모델의 학습 파이프라인을 변경하지 않고 적용 가능한 **post-hoc 가지치기 기법** 제안.
2. 휴리스틱보다 우수한 **공간(spatial) 민감도 점수** 도입 (평균·스케일 파라미터 기반의 Fisher 블록 근사).
3. **Multi-round prune-refine 파이프라인** 제안 — 1회 90% 가지치기보다 (80% → 50%) 두 번에 나눠 가지치기 + fine-tuning이 모든 지표에서 우수.

핵심 결과: **3D-GS Gaussian의 90%를 제거하면서 평균 렌더링 속도 3.56배 향상**, Mip-NeRF 360 / Tanks & Temples / Deep Blending에서 LightGaussian 대비 PSNR·SSIM·LPIPS·FPS 모두 우수.

---

## 2. 상세 설명: 문제·방법·구조·성능·한계

### 2.1 해결하고자 하는 문제

3D-GS는 한 장면당 수백만 개의 3D Gaussian을 사용하기 때문에 단일 unbounded 장면이 1GB 이상을 차지하고, 모바일·AR/VR·임베디드 장치에서의 활용이 제한됩니다. 기존 가지치기 방법들은 다음 한계를 가집니다.

- **LightGaussian**: 가시성(visibility) × 불투명도(opacity) × Gaussian 부피(volume)의 곱을 "global significance"로 사용 → 큰 Gaussian이 더 많이 보이므로 큰 Gaussian을 우선 보존하는 편향이 발생 → 작은 디테일 손실 (LightGaussian 공식 페이지).
- **Compact-3DGS / EAGLES**: 학습 파이프라인 자체를 변경하거나 추가적인 학습 가능한 마스크를 도입해야 함.
- 모든 휴리스틱 기반 방법은 **수학적 정당성 부족**으로 압축률이 높아질수록 시각적 충실도가 급격히 저하됩니다.

### 2.2 3D-GS 배경 수식

각 Gaussian은 다음과 같이 표현됩니다.

$$
\mathcal{G}_i(x) = e^{-\frac{1}{2}(x-\mu_i)^T \Sigma_i^{-1}(x-\mu_i)}, \quad \Sigma_i = R_i S_i S_i^T R_i^T
$$

각 픽셀 $p$의 색은 알파 블렌딩으로 계산됩니다.

$$
C(p) = \sum_{i \in \mathcal{N}} \tilde{c}_i\, \tilde{\alpha}_i(p) \prod_{j=1}^{i-1}\bigl(1 - \tilde{\alpha}_j(p)\bigr)
$$

학습은 다음 손실로 진행됩니다.

$$
L(\mathcal{G} \mid \phi, I_{gt}) = \|I_{\mathcal{G}}(\phi) - I_{gt}\|_1 + L_{\text{SSIM}}\bigl(I_{\mathcal{G}}(\phi), I_{gt}\bigr)
$$

### 2.3 제안 방법 — 민감도 점수 유도

**(1) L2 손실의 Hessian**

$$
L_2 = \tfrac{1}{2}\sum_{\phi \in \mathcal{P}_{gt}} \| I_{\mathcal{G}}(\phi) - I_{gt} \|_2^2
$$

이를 두 번 미분하면:

$$
\nabla_{\mathcal{G}}^2 L_2 = \sum_{\phi}\Bigl[ \nabla_{\mathcal{G}} I_{\mathcal{G}}(\phi)\, \nabla_{\mathcal{G}} I_{\mathcal{G}}(\phi)^T + \bigl(I_{\mathcal{G}}(\phi)-I_{gt}\bigr)\nabla_{\mathcal{G}}^2 I_{\mathcal{G}}(\phi) \Bigr]
$$

**(2) Fisher 근사**

수렴된 모델에서는 잔차 $\|I_{\mathcal{G}} - I_{gt}\|_1 \to 0$ 이므로 두 번째 항이 사라져 다음 형태의 Fisher Information Matrix를 얻습니다.

$$
\nabla_{\mathcal{G}}^2 L_2 \approx \sum_{\phi \in \mathcal{P}_{gt}} \nabla_{\mathcal{G}} I_{\mathcal{G}}(\phi)\, \nabla_{\mathcal{G}} I_{\mathcal{G}}(\phi)^T
$$

중요한 점은 이것이 **ground-truth $I_{gt}$에 의존하지 않고 입력 카메라 포즈 $\mathcal{P}_{gt}$에만 의존**한다는 것입니다.

**(3) 블록 대각 근사 + 공간 파라미터 제한**

전체 $\mathcal{G} \times \mathcal{G}$ Hessian은 비현실적이므로, 각 Gaussian $\mathcal{G}_i$에 대한 블록만 사용:

$$
H_i = \nabla_{\mathcal{G}_i} I_{\mathcal{G}}\, \nabla_{\mathcal{G}_i} I_{\mathcal{G}}^T
$$

이 블록의 **로그 행렬식**을 민감도 점수로 사용:

$$
\tilde{U}_i = \log |H_i| = \log \bigl|\nabla_{\mathcal{G}_i} I_{\mathcal{G}}\, \nabla_{\mathcal{G}_i} I_{\mathcal{G}}^T\bigr|
$$

저자들은 평균 $x_i$ 와 스케일 $s_i$ 만 사용해도 충분함을 실험적으로 검증하여 최종 점수를 다음과 같이 정의합니다.

$$
\boxed{\; U_i = \log \bigl|\nabla_{x_i, s_i} I_{\mathcal{G}}\, \nabla_{x_i, s_i} I_{\mathcal{G}}^T\bigr| \;}
$$

회전 $r_i$를 제외한 이유는 **회전 변환이 3D 기하학적 invariance를 추가로 만들지 않기 때문**이며, 회전을 포함하면 행렬 크기가 $6\times 6 \to 10\times 10$으로 2.78배 증가해 메모리 부담이 커집니다 (RTXA4000 GPU에서 실행 불가).

**(4) Bayesian 해석 (Appendix A.1)**

L2 목적함수를 음의 로그우도로 재해석하고, 수렴된 파라미터 $\hat{\mathcal{G}}$ 주변에서 Laplace 근사를 취하면:

$$
-\log p(\mathcal{G}\mid \mathcal{I},\Phi) \approx -\log p(\hat{\mathcal{G}}\mid \mathcal{I},\Phi) + \tfrac{1}{2}(\mathcal{G}-\hat{\mathcal{G}})^T H(\hat{\mathcal{G}})(\mathcal{G}-\hat{\mathcal{G}})
$$

이때 $\log|H_i|$ 는 사후분포 $p(\mathcal{G}_i \mid \mathcal{I},\Phi)$의 **엔트로피**를 측정하므로, 본 점수는 각 Gaussian의 사후 불확실성에 의한 순위라는 의미를 가집니다.

**(5) Patch-wise 효율화**

픽셀 단위 Fisher 합산 대신 $4\times 4$ 패치 단위 근사를 사용해도 점수 상관성이 매우 높아 거의 동일한 성능을 얻습니다 (Appendix A.2).

### 2.4 모델 구조 — Multi-Round Prune-Refine 파이프라인

1. 사전 학습된 3D-GS 모델 입력
2. 모든 학습 시점에 대해 $U_i$ 계산 (수 초 소요, CUDA 커널 구현)
3. 점수가 낮은 80%의 Gaussian 제거
4. 5,000 iteration fine-tuning (densification 없음)
5. 남은 모델에서 다시 50% 추가 가지치기 → 누적 90% 제거
6. 다시 5,000 iteration fine-tuning
7. (선택) Vectree Quantization 적용 → 약 50× 추가 압축

### 2.5 성능 향상 (Mip-NeRF 360 평균)

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | Size(MB)↓ |
|---|---|---|---|---|---|
| 3D-GS | 27.47 | 0.8123 | 0.2216 | 64.07 | 746.46 |
| LightGaussian (90% prune) | 26.28 | 0.7622 | 0.3054 | 162.12 | 74.65 |
| **PUP 3D-GS (Ours)** | **26.67** | **0.7862** | **0.2719** | **204.81** | 74.65 |
| Ours + Vectree Quant. | 24.93 | 0.7584 | 0.2988 | 205.97 | **14.44** |

데이터 출처: 본 논문 Tables 2–3.

세 데이터셋 평균 **3.56× 렌더링 가속**, Tanks & Temples PSNR 한 가지를 제외하고 모든 지표에서 LightGaussian 대비 우위.

### 2.6 한계

1. **수렴 가정 의존성**: Fisher 근사는 $\|I_{\mathcal{G}} - I_{gt}\|_1 \approx 0$일 때만 정확 → 학습 도중 사용 불가.
2. **메모리 비용**: $N \times 36$ (N개 Gaussian × 6×6 Fisher 블록)에 비례 → 매우 큰 장면에서는 16GB GPU 부족 가능성.
3. **배경(Background) 열화**: 90% 같은 극단 가지치기에서는 배경 영역이 더 많이 손실됩니다(저자들은 이를 전경 우선 trade-off로 정당화하나 명시적 한계).
4. **Anchor 기반 모델 (Scaffold-GS, Octree-GS)에 직접 적용 불가** — 앵커당 고정 개수의 Gaussian이 생성되기 때문.

---

## 3. 모델의 일반화 성능 향상 가능성

본 논문이 일반화에 기여하는 측면을 두 갈래로 정리할 수 있습니다.

### 3.1 모델 자체의 일반화(novel view synthesis 능력) 측면

- **불확실성 기반 가지치기는 학습 분포 의존성이 적은 Gaussian을 보존**합니다. 즉, 다양한 시점에서 일관되게 관측 가능한(low-uncertainty, well-constrained) Gaussian만 남기므로 학습된 시점 이외의 새로운 뷰에서도 안정적인 렌더링을 기대할 수 있습니다. 이는 Figure 2의 spatial uncertainty 분석과 직결됩니다.
- **Foreground-우선 보존**은 사람의 시각적 관심 영역에서의 일반화 품질을 끌어올리는 효과가 있습니다 (Appendix A.9의 L1 residual 시각화).
- **Patch 단위 근사**가 픽셀 단위와 거의 동등한 결과를 낸다는 점은, 점수가 픽셀-스케일 노이즈에 과적합되지 않고 robust함을 시사합니다.

### 3.2 다른 파이프라인으로의 전이(transfer) 가능성

저자들은 직접 다음을 보여 줍니다.

- **EAGLES + PUP 3D-GS**: 베이스 EAGLES 모델 (3D-GS 대비 2.51× 작음)에 본 파이프라인을 적용하면 **25.14× 추가 압축**과 더 높은 화질을 동시에 달성. 즉 **다른 학습-시 압축 기법과 직교(orthogonal)** 합니다.
- **Vectree Quantization과 결합** 가능 → 약 50배 추가 압축에도 LightGaussian보다 우수.
- Appendix A.7에서 저자들은 점수가 ground truth에 의존하지 않으므로 **학습 도중**, **NeRF teacher 없이도**, **다른 3D 표현 (PGSR, Mini-Splatting 등) 으로 확장** 가능하다고 명시합니다 — 이는 일반화의 미래 방향성입니다.

### 3.3 일반화 성능에 대한 신중한 평가

다만 다음 점은 주의가 필요합니다.

- 본 논문은 학습 뷰에 대한 재구성 오차의 Hessian을 사용하므로, **검증 뷰가 학습 뷰와 분포가 매우 다른 경우**(out-of-distribution view)의 일반화는 별도로 검증되지 않았습니다.
- 데이터셋이 모두 정적·실내외 장면이며, **동적 장면(4D)·도심 대규모 장면**에 대한 일반화는 후속 연구가 필요합니다.
- 100% 확신이 아닌 부분: 저자가 직접 "out-of-distribution view에서의 일반화"를 정량 비교한 표는 본 논문에 포함되어 있지 않으므로, 이 영역에서 PUP 3D-GS가 명시적으로 우월하다고 단정하긴 어렵습니다.

---

## 4. 향후 연구에의 영향과 고려할 점

### 4.1 연구계에 미친/미칠 영향

1. **휴리스틱 → 원리 기반 패러다임 전환**: 저자들은 3D-GS 압축 분야에서 처음으로 Bayesian/Fisher 정보 기반의 정형적 점수를 도입했고, 이후 후속 연구들이 학습 가능한 마스크(LP-3DGS, MaskGaussian)나 Hessian 기반 변형(GETA-3DGS) 등으로 다양화되고 있습니다 (GETA-3DGS는 LP-3DGS, RadSplat, Compact-3DGS 등을 함께 비교).
2. **Post-hoc, 학습 파이프라인 비-침습적**이라는 강점은 산업 응용에서 매우 중요 — 이미 학습이 끝난 다수의 3D-GS 자산을 그대로 활용 가능.
3. **Vectree quantization, EAGLES, Mini-Splatting 같은 직교(orthogonal) 기법과 결합** 가능 → 압축 파이프라인의 표준 빌딩 블록으로 사용될 가능성.
4. FisherRF가 active view selection에 사용한 Fisher 정보를 본 논문이 **가지치기**로 재해석한 점은, **하나의 도구(uncertainty)를 다양한 다운스트림 작업에 통합 적용**하는 흐름을 강화합니다.

### 4.2 후속 연구에서 고려해야 할 점

1. **학습 도중(training-time) 적용**: 현재는 수렴 후에만 적용 가능. 학습 중 작은 잔차 구간에서의 적용·근사 정확도 분석이 필요.
2. **앵커 기반 표현 (Scaffold-GS, Octree-GS) 호환성**: 앵커당 Gaussian이 묶여 있어 직접 가지치기가 안 됨 → 점수를 앵커 단위로 집계/매핑하는 전략 필요.
3. **메모리 효율화**: $O(N\times 36)$ 메모리는 수천만 Gaussian 규모 도시 스캔 등에서 병목. 저-랭크 근사, 스트리밍 계산 등이 연구 가치 있음.
4. **배경 보존 균형**: 마스킹 가중치, foreground/background trade-off 하이퍼파라미터 도입 (저자 직접 향후 과제로 명시).
5. **동적 4D Gaussian으로의 확장**: 시간 차원에서의 민감도 정의(시점 간 일관성 가중치 등)는 여전히 열린 문제.
6. **2DGS/Surfel·표면 재구성과의 결합**: 깊이·법선 맵 같은 추가 채널을 Fisher에 포함시키는 일반화가 필요 (PGSR, Mini-Splatting 등). 본 논문 부록도 이 방향을 명시적으로 제안.
7. **Out-of-distribution view 일반화 평가**: 별도 검증 프로토콜 필요.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

다음은 본 논문과 비교 가능한 3D-GS 관련 압축·가지치기 연구들의 핵심 차이입니다. (확실히 검증된 내용만 기재)

### 5.1 휴리스틱 기반 가지치기 계열

| 방법 (연도) | 핵심 기준 | PUP 3D-GS와의 차이 |
|---|---|---|
| **3D-GS** (Kerbl et al., 2023) | Adaptive Density Control: opacity·gradient 임계값 | 압축이 아니라 학습 중 동적 관리. 기준은 gradient·pixel coverage·saliency |
| **LightGaussian** (Fan et al., 2023) | global significance = visibility × opacity × volume | 휴리스틱·큰 Gaussian 편향; PUP은 원리 기반·작은 Gaussian 보존 |
| **EAGLES** (Girish et al., 2023) | least total transmittance per Gaussian + 양자화 | 학습 파이프라인 변경 필요. PUP과 결합 시 25× 추가 압축 가능 |
| **Compact-3DGS** (Lee et al., 2023) | 학습 가능한 mask로 작은·투명 Gaussian 제거 | 학습 시 통합. PUP은 post-hoc |
| **RadSplat** (Niemeyer et al., 2024) | teacher radiance field의 importance score | 사전 학습된 NeRF 필요. PUP은 외부 prior 불필요 |
| **Mini-Splatting** (Fang & Wang, 2024, ECCV) | importance/blur split + 제한된 Gaussian 수 | 밀도화(densification) 전략과 가지치기를 함께 다룸; PUP은 post-hoc 가지치기에 집중 |

### 5.2 학습 가능 마스크 / 자동화 계열 (PUP 이후)

- **LP-3DGS** (Zhang et al., NeurIPS 2024): RadSplat·Mini-Splatting을 베이스로 학습 가능 마스크를 도입해 자동 가지치기 비율 결정. PUP의 비율은 사용자가 선택해야 한다는 점에서 자동화 측면 보완.
- **MaskGaussian / GaussianSpa**: 학습 시 binary 또는 sparsity-inducing 마스크 학습 — 학습 시 통합형.
- **GETA-3DGS** (2025): structured pruning + quantization을 자동 결합하는 프레임워크; PUP 같은 점수 계산보다 그룹 단위 결합 압축에 집중.

### 5.3 양자화·구조화 압축 (직교적 기법)

- **Compressed-3D / Compact3D / RDO-Gaussian**: 10MB 이하 모델 크기 보고 — vector quantization·entropy coding이 핵심.
- **Scaffold-GS / Octree-GS / HAC** (2024): 앵커-그리드 컨텍스트 모델링으로 100× 압축 — 학습 표현 자체를 변경. PUP과는 직교.
- **Reduced-3DGS / Spectrally Pruned Gaussian Fields**: SH·주파수 기반 중복 제거.

### 5.4 최신 OMG, Trim 3D-GS (2024–2025)

- **OMG (Optimized Minimal 3D Gaussian Splatting, 2025)**: local distinctiveness 점수 + 기존 blending weight 점수 조합으로 약 0.4M Gaussian / 4.1MB까지 압축, 600+ FPS — PUP보다 더 공격적 압축 보고. 하지만 학습 파이프라인을 새로 설계.
- **Trim 3DGS** (2024): 정확한 기하 표현을 위해 Gaussian 수 줄이기. 표면 재구성 중심.

### 5.5 서베이/벤치마크

- **3DGS.zip Survey** (Bagdasarian et al., 2025, Computer Graphics Forum): 압축과 compaction을 분리하여 정리, ADC 기반 가지치기와 양자화 계열 분류.
- **SUCCESS-GS** (2025): 3D/4D Gaussian Splatting 압축을 Parameter Compression vs Restructuring Compression로 분류.
- **Splatwizard** (2025): 압축 기법 벤치마킹 툴킷.

### 5.6 비교 요약 테이블 (PUP vs 주요 baselines)

| 측면 | LightGaussian | Compact-3DGS | RadSplat | LP-3DGS | **PUP 3D-GS** |
|---|---|---|---|---|---|
| 점수 유형 | 휴리스틱 | 학습 마스크 | NeRF teacher | 학습 마스크 | **Fisher 민감도** |
| 학습 파이프라인 변경 | X | O | O (NeRF 필요) | O | **X (post-hoc)** |
| 외부 모델 필요 | X | X | O | X | **X** |
| 압축률 (3D-GS 대비) | ~10× | ~7× | ~6× | 가변 | **10× (단독), 50× (+ Vectree)** |
| 작은 Gaussian 보존 | 약함 | 보통 | 보통 | 보통 | **강함** |
| 주요 압축률에서 LPIPS | 높음 | 보통 | 보통 | 보통 | **낮음** |

---

## 참고 자료(출처)

1. **본 논문 (1차 자료)**: A. Hanson, A. Tu, V. Singla, M. Jayawardhana, M. Zwicker, T. Goldstein, *"PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting,"* arXiv:2406.10219v3, CVPR 2025. (사용자 제공 PDF)
2. 공식 프로젝트 페이지: https://pup3dgs.github.io/
3. CVPR 2025 Open Access 버전: https://openaccess.thecvf.com/content/CVPR2025/papers/Hanson_PUP_3D-GS_Principled_Uncertainty_Pruning_for_3D_Gaussian_Splatting_CVPR_2025_paper.pdf
4. **LightGaussian** (Fan et al., 2023), arXiv:2311.17245, 프로젝트 페이지: https://lightgaussian.github.io/
5. **3DGS.zip: A Survey on 3D Gaussian Splatting Compression Methods** (Bagdasarian et al., 2025), Computer Graphics Forum, https://w-m.github.io/3dgs-compression-survey/
6. **SUCCESS-GS: Survey of Compactness and Compression for Efficient Static and Dynamic Gaussian Splatting** (Youn et al., 2025), https://cmlab-korea.github.io/Awesome-Efficient-GS/
7. **Compression in 3D Gaussian Splatting: A Survey of Methods, Trends, and Future Directions** (2025), arXiv:2502.19457
8. **LP-3DGS: Learning to Prune 3D Gaussian Splatting** (Zhang et al., NeurIPS 2024)
9. **GETA-3DGS: Automatic Joint Structured Pruning and Quantization for 3D Gaussian Splatting** (2025), arXiv:2605.02086 (인덱스 번호는 검색 결과 그대로)
10. **Optimized Minimal 3D Gaussian Splatting (OMG)** (2025), arXiv:2503.16924
11. **RadSplat** (Niemeyer et al., 2024), arXiv:2403.13806
12. **Mini-Splatting** (Fang & Wang, ECCV 2024)
13. **Compact 3D Gaussian Representation for Radiance Field** (Lee et al., 2023), arXiv:2311.13681
14. **EAGLES** (Girish et al., 2023), arXiv:2312.04564
15. **FisherRF** (Jiang et al., 2023), arXiv:2311.17874
16. **BayesRays** (Goli et al., CVPR 2024)

---

## 정확도 한계 고지

- 본 답변에서 **본 논문에 직접 명시된 수식·표·실험 수치**는 사용자 제공 PDF에서 확인된 내용입니다.
- **2020년 이후 관련 연구 비교 표**는 각 논문 초록·서베이 논문 기재를 근거로 작성했으나, 일부 베이스라인의 정확한 압축률·FPS는 측정 환경에 따라 달라질 수 있어 직접 비교는 신중해야 합니다.
- "PUP 3D-GS의 OOD view 일반화 우위"와 같이 **본 논문이 정량적으로 직접 비교하지 않은 항목**은 단정적 결론을 피하고 향후 검증 필요로 남겨 두었습니다.
