# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

**논문 정보**: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis (Inria, Université Côte d'Azur / Max-Planck-Institut für Informatik), ACM Transactions on Graphics, Vol. 42, No. 4, August 2023. SIGGRAPH 2023 Best Paper.

---

## 1. 핵심 주장과 주요 기여 (요약)

Radiance Field 방법론은 다중 사진·영상으로 캡처된 장면의 novel-view synthesis를 혁신했으나, 높은 시각적 품질을 달성하려면 학습과 렌더링에 비용이 큰 신경망이 필요하며, 빠른 방법은 속도와 품질을 트레이드오프한다. 특히 비한정(unbounded) 완전 장면에서 1080p 해상도로 실시간 렌더링을 달성하는 방법은 존재하지 않았다.

이 논문은 **세 가지 핵심 요소**를 도입하여 이 문제를 해결한다:

① 카메라 캘리브레이션 중 생성된 희소 포인트로부터 **3D Gaussian으로 장면을 표현**하여 연속 볼류메트릭 래디언스 필드의 바람직한 속성을 보존하면서 빈 공간에서의 불필요한 연산을 회피하고, ② **인터리브된 최적화/밀도 제어**를 수행하여 비등방(anisotropic) 공분산을 최적화하여 장면을 정밀하게 표현하며, ③ 비등방 스플래팅을 지원하는 **빠른 가시성 인식 렌더링 알고리즘**을 개발하여 학습 가속 및 실시간 렌더링을 가능하게 한다.

완전 수렴 모델(30,000 이터레이션)은 Mip-NeRF360과 동등하거나 약간 더 나은 품질을 달성하면서 학습 시간은 35–45분(vs. 48시간)으로 대폭 단축되고, 렌더링 속도는 실시간(vs. 프레임당 10초)이다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

이 연구의 핵심 명성은 제목에서 알 수 있듯 **높은 렌더링 속도**이며, 이는 표현 자체와 맞춤형 CUDA 커널을 사용한 렌더링 알고리즘 덕분이다.

기존 NeRF 기반 방법의 구체적 한계:
- 신경망 기반 연속 3D 장면 복원은 (i) 복원 후 편집 불가능, (ii) MLP와 밀집 레이 샘플링에 의존하여 대규모 장면의 학습 시간이 수십 시간에 달하고 렌더링 속도가 초당 수 프레임에 불과하다.
- 연속 볼류메트릭 래디언스 필드의 강력한 피팅 능력을 보존하면서도, NeRF 기반 방법의 계산 오버헤드(비용이 큰 ray-marching, 빈 공간에서의 불필요한 계산)를 동시에 회피한다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) 3D Gaussian 표현

3DGS는 3D Gaussian이라 불리는 이산 기하 프리미티브 집합으로 3D 데이터를 표현한다. 각 Gaussian은 중심 위치 $\mu \in \mathbb{R}^3$, 스케일링 벡터 $s \in \mathbb{R}^3$, 회전 쿼터니언 $q \in \mathbb{R}^4$로 정의된다. 이 매개변수로부터 공분산 행렬 $\Sigma \in \mathbb{R}^{3\times3}$을 물리적으로 타당한 방식으로 구성한다.

$$
G(x) = e^{-\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu)}
$$

공분산은 타원체의 형태로 해석할 수 있으며, 수학적으로 스케일링 행렬과 회전 행렬로 분해된다:

$$
\Sigma = R S S^T R^T
$$

여기서 $S = \text{diag}(s_1, s_2, s_3)$는 스케일링 행렬, $R$은 쿼터니언 $q$로부터 유도되는 회전 행렬이다.

외형(appearance)을 모델링하기 위해 각 Gaussian은 불투명도 값 $\alpha \in [0, 1]$과 구면 조화 함수(SH) 계수 $c \in \mathbb{R}^C$로 표현되는 뷰 의존 색상 속성을 가진다.

#### (B) 2D 투영 및 렌더링

렌더링 시 3D Gaussian은 2D 이미지 평면에 splat으로 투영된다. 카메라의 뷰 변환 행렬 $W$와 야코비안 $J$를 사용하여, 2D 공분산 $\Sigma'$은 다음과 같이 계산된다:

$$
\Sigma' = J W \Sigma W^T J^T
$$

NeRF와 Gaussian Splatting은 동일한 이미지 형성 모델을 공유한다. 픽셀 색상 $C$는 깊이 순서대로 정렬된 Gaussian들의 $\alpha$-블렌딩으로 계산된다:

$$
C = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

여기서 $c_i$는 $i$번째 Gaussian의 색상(SH에서 계산), $\alpha_i$는 학습된 불투명도와 2D 투영된 Gaussian의 기여를 곱한 값이다.

#### (C) 손실 함수

최적화는 L1 손실과 D-SSIM을 결합한 손실 함수를 확률적 경사 하강법(SGD)으로 최소화하며, 이는 Plenoxels 작업에서 영감을 받았다:

$$
\mathcal{L} = (1 - \lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}
$$

여기서 $\lambda$는 가중치 하이퍼파라미터이다.

#### (D) 적응적 밀도 제어 (Adaptive Density Control)

Gaussian들은 적응적 밀도 제어를 통해 최적화되며, Gaussian의 분포와 정렬을 정제하여 장면을 더 잘 표현한다. 효율적 렌더링을 위해 미분 가능한 타일 기반 래스터라이저가 사용된다.

- **복제(Clone)**: 기하 구조가 부족한 영역(under-reconstruction)에서 작은 Gaussian을 복제
- **분할(Split)**: 작은 규모의 기하가 하나의 큰 splat으로 표현되는 경우 이를 둘로 분할한다.
- **가지치기(Prune)**: 불투명도가 임계값 이하인 Gaussian 제거

### 2.3 모델 구조

MLP조차 없고, "신경(neural)" 요소가 전혀 없으며, 장면은 본질적으로 공간 내 점들의 집합에 불과하다.

| 구성 요소 | 설명 |
|---|---|
| **입력** | 정적 장면의 이미지 세트 + 카메라 위치, 희소 포인트 클라우드로 표현 |
| **3D Gaussian** | 각 Gaussian에 대한 평균, 공분산 행렬, 불투명도 정의 |
| **색상 표현** | 뷰 의존적 외형을 모델링하기 위해 구면 조화 함수(SH) 사용 |
| **래스터라이저** | 빠른 정렬과 역전파를 위한 타일 기반 래스터라이저, Gaussian 컴포넌트의 효율적 블렌딩 |
| **최적화** | 학습 과정은 신경망과 유사하게 확률적 경사 하강법을 사용하되, 레이어가 없다. |

### 2.4 성능 향상

총 13개의 실제 장면(기존 공개 데이터셋)과 합성 Blender 데이터셋에서 테스트되었으며, Mip-NeRF360(당시 NeRF 렌더링 품질 최고 수준), Tanks and Temples 데이터셋 2개 장면, Deep Blending 2개 장면을 포함한다.

Mip-NeRF360, InstantNGP, Plenoxels와 같은 최첨단 기법과 비교하였으며, PSNR, LPIPS, SSIM 정량 평가 지표가 사용되었다.

| 지표 | 3DGS (30K iter) | Mip-NeRF360 | InstantNGP |
|---|---|---|---|
| 학습 시간 | ~35–45분 | ~48시간 | ~5–10분 |
| 렌더링 속도 | **≥100 FPS** | ~10초/프레임 | ~10 FPS |
| 시각적 품질 | Mip-NeRF360 수준 이상 | 최고 | 중간 |

7,000 이터레이션(5–10분 학습) 시점에서 이미 InstantNGP 및 Plenoxels와 비견되는 품질을 달성한다.

### 2.5 한계

기존 포인트 기반 접근법보다 컴팩트하지만, NeRF 기반 솔루션보다 메모리 소비가 훨씬 높다. 대규모 장면 학습 시 GPU 메모리 소비가 최적화되지 않은 프로토타입에서 20GB를 초과할 수 있다.

저자들은 이러한 한계가 향후 더 나은 컬링 접근법, 안티앨리어싱, 정규화, 압축 기술로 해결될 수 있다고 언급한다.

추가 한계점:
- 약한 텍스처 영역 모델링, 동적 장면 적응, 하드웨어 자원 소비의 어려움이 여전히 해결되어야 한다.
- 현재 상태에서 3DGS는 미세한 디테일에 다소 어려움이 있다.
- Gaussian 초기화에 대한 민감성, 제한된 공간 인식, 약한 Gaussian 간 상관관계 등의 한계가 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

3DGS의 가장 중요한 한계 중 하나는 **일반화(generalization)** 문제이다. 원래 3DGS는 장면별(per-scene) 최적화 방식이므로, 학습되지 않은 새로운 장면에 바로 적용할 수 없다.

### 3.1 핵심 일반화 문제

희소 입력 뷰로 감독이 제한될 때, 3DGS는 관찰된 이미지에 과적합하여 보이지 않는 시점에 대한 일반화가 불량해진다.

3DGS에 일반화 능력을 부여하려는 여러 시도가 있었지만, 기존 방법들은 좁은 범위의 장면 수준 뷰 보간과 객체 중심 합성에 제한된다. 주된 이유는 기존 방법들이 다시점 이미지 간 밀집 뷰 매칭에 의존하여 Gaussian 프리미티브를 예측하는데, 이것이 긴 시퀀스에서 계산적으로 비현실적이 되어 감독 범위가 좁은 보간 뷰로 제한되기 때문이다.

### 3.2 일반화 향상을 위한 최신 연구 방향

#### (A) Mip-Splatting (CVPR 2024)

3DGS는 3D 객체를 3D Gaussian으로 표현하여 이미지 평면에 투영한 뒤 2D 확장(dilation)을 수행한다. 그러나 내재적 수축 편향(shrinkage bias)이 샘플링 한계를 넘는 퇴화된 3D Gaussian을 유발하며, 샘플링 레이트가 변경될 때(초점 거리나 카메라 거리를 통해) 강한 확장 효과와 고주파 아티팩트가 관찰된다.

Mip-Splatting에서 제안된 3D Gaussian 표현의 수정은 뛰어난 분포 외(out-of-distribution) 일반화를 가능하게 한다: 단일 샘플링 레이트에서의 학습으로 학습 시 사용되지 않은 다양한 샘플링 레이트에서 충실한 렌더링이 가능하다.

#### (B) FreeSplat (NeurIPS 2024)

3DGS에 일반화 능력을 부여하는 것은 매력적이다. 그러나 기존 일반화 가능한 3DGS 방법은 무거운 백본으로 인해 스테레오 이미지 간 좁은 범위의 보간에 주로 국한된다. FreeSplat은 긴 시퀀스 입력으로부터 기하학적으로 일관된 3D 장면을 복원할 수 있는 프레임워크로, 근접 뷰 간 적응적 코스트 볼륨을 구성하는 Low-cost Cross-View Aggregation과 중복 Gaussian을 제거하는 Pixel-wise Triplet Fusion을 도입한다.

#### (C) Flat Minima Optimization (2025)

Flat minima 최적화 관점에서 접근하여, 작은 매개변수 섭동 하에서도 안정적인 솔루션을 찾는다. Gaussian 매개변수를 훈련 가능한 가중치로 간주하고, Scale-Adaptive Perturbation(SAP) 기법—각 Gaussian의 비등방성에 따라 섭동 크기를 조절—을 도입한다. 또한 확률적 섭동으로 각 Gaussian을 확률적으로 섭동하거나 그대로 두어 과도한 스무딩을 방지하고, 학습 중 섭동 크기를 점진적으로 증가시키는 스케줄링을 적용한다.

#### (D) Feed-Forward 일반화 모델

PixelSplat은 pixelNeRF의 접근을 따라 3DGS의 효율적 학습/렌더링을 활용한다. MVSplat은 깊이 맵 추정 없이 희소 다시점 이미지로부터 3D Gaussian 분포를 예측하는 효율적 feed-forward 3D 복원 모델을 도입하여, PixelSplat 대비 매개변수를 10배 줄이고 추론 속도를 2배 높이면서 크로스 데이터셋 일반화도 향상시켰다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

3DGS는 2023년 4월 말 공개되어 빠르게 인기를 얻었고, 2023년 8월 SIGGRAPH에서 최우수 논문을 수상했다.

3DGS의 도입은 단순한 기술적 진보가 아니라, 컴퓨터 비전 및 그래픽스에서 장면 표현과 렌더링에 접근하는 방식의 근본적 전환을 나타낸다. 시각적 품질을 타협하지 않으면서 실시간 렌더링을 가능하게 함으로써 VR/AR부터 실시간 시네마틱 렌더링까지 다양한 가능성을 열었다.

3DGS는 최근 NeRF의 강력한 대안으로 부상하여, 실시간 성능과 함께 고충실도 포토리얼리스틱 렌더링을 제공한다. novel view synthesis를 넘어, 3DGS의 명시적이고 컴팩트한 특성은 기하학적·의미론적 이해가 필요한 다양한 하류 응용을 가능하게 한다.

주요 응용 분야 확장:
Text-to-3D 생성, 자율 주행 시뮬레이션, SuGaR를 통한 정밀 메시 추출 등으로 광범위하게 확장되었다.

### 4.2 향후 연구 시 고려해야 할 사항

| 연구 방향 | 핵심 과제 |
|---|---|
| **메모리 효율** | Compact3D, LightGaussian 등의 Gaussian 압축 기술 통합으로 모델 컴팩트성과 렌더링 품질의 균형을 맞추어야 한다. |
| **동적 장면** | 3D temporal Gaussian Splatting이 시간 컴포넌트를 통합하여 동적 장면의 실시간 렌더링을 가능하게 하나, 캡처 가능한 모션 길이에 현재 한계가 있다. |
| **반사/투명 표면** | 반사 객체 렌더링은 여전히 상당한 도전이며, 특히 역렌더링과 리라이팅에서 그러하다. |
| **NeRF 통합** | NeRF-GS 프레임워크는 NeRF의 연속 공간 표현을 활용하여 3DGS의 Gaussian 초기화 민감성, 제한된 공간 인식, 약한 Gaussian 간 상관관계를 완화하고 성능을 향상시킨다. |
| **레이 트레이싱** | 2024년 7월 NVIDIA Research가 3D Gaussian Ray Tracing을 공개하여, 래스터화가 아닌 레이 트레이싱을 활용하는 새로운 연구 방향을 제시했다. |
| **하드웨어 최적화** | 병목인 수백만 Gaussian의 정렬은 CUDA 전용 고도로 최적화된 정렬(CUB device radix sort)로 수행되며, 다른 렌더링 파이프라인으로의 이식이 필요하다. |
| **Embodied AI** | 3DGS가 매우 밀집된 3D 공간 표현을 생성한다는 점에서, Embodied AI 연구에 대한 잠재적 의미가 관심을 끌고 있다. |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 기여 | 3DGS와의 관계 |
|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | MLP 기반 암시적 볼류메트릭 장면 표현 | 3DGS가 해결하고자 한 느린 학습/렌더링의 원인 |
| **Plenoxels** (Fridovich-Keil et al.) | 2022 | 신경망 없는 래디언스 필드 | 3DGS 손실 함수에 영감 제공 |
| **Instant-NGP** (Müller et al.) | 2022 | 해시 테이블 인코딩으로 빠른 NeRF | 학습 속도 면에서 3DGS의 비교 대상 |
| **Mip-NeRF360** (Barron et al.) | 2022 | 비한정 장면의 안티앨리어싱 NeRF | 품질 면에서 3DGS의 주요 비교 대상 |
| **Mip-Splatting** (Yu et al.) | CVPR 2024 | 앨리어싱 제거 3DGS, 다중 해상도 일반화 | 3DGS의 분포 외 일반화 문제 직접 해결 |
| **FreeSplat** (NeurIPS 2024) | 2024 | 긴 시퀀스 입력 일반화 가능 3DGS | 자유 시점 합성 범위 확장 |
| **PixelSplat / MVSplat** | 2024 | Feed-forward 일반화 가능 3DGS | 장면별 최적화 없이 즉시 추론 |
| **Street Gaussian** (Yan et al.) | 2024 | 동적 거리 장면 3DGS | 자율 주행에서 움직이는 물체 복원 |
| **3D Gaussian Ray Tracing** (NVIDIA) | 2024 | 래스터화 대신 레이 트레이싱 | Splatting 제한(어안 렌즈 등)을 해소하고, 굴절·그림자·피사계 심도·거울 등 이차 조명 효과를 지원한다. |
| **NeRF-GS** (Fang et al.) | ICCV 2025 | NeRF-3DGS 결합 프레임워크 | 3DGS 초기화 민감성·공간 인식 한계 완화 |
| **Flat Minima Optimization** | 2025 | FM 기반 희소 뷰 일반화 향상 | 기존 3DGS 파이프라인에 아키텍처 변경 없이 원활하게 통합되는 경량 프레임워크 |
| **EVSplitting** (SIGGRAPH Asia 2024) | 2024 | 효율적이고 시각적으로 일관된 3DGS 분할 알고리즘으로, 산업 생산에 용이하도록 설계 |

---

## 참고자료 출처

1. **Kerbl, B. et al.** "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics*, Vol. 42, No. 4, 2023. ([arXiv:2308.04079](https://arxiv.org/abs/2308.04079), [Project Page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/))
2. **Wikipedia** - "Gaussian splatting." ([en.wikipedia.org/wiki/Gaussian_splatting](https://en.wikipedia.org/wiki/Gaussian_splatting))
3. **Hugging Face Blog** - "Introduction to 3D Gaussian Splatting." ([huggingface.co/blog/gaussian-splatting](https://huggingface.co/blog/gaussian-splatting))
4. **LearnOpenCV** - "3D Gaussian Splatting - Paper Explained." ([learnopencv.com/3d-gaussian-splatting](https://learnopencv.com/3d-gaussian-splatting/))
5. **Towards Data Science** - Kate Feingold, "A Comprehensive Overview of Gaussian Splatting." ([towardsdatascience.com](https://towardsdatascience.com/a-comprehensive-overview-of-gaussian-splatting-e7d570081362/))
6. **KIRI Engine** - "3D Gaussian Splatting: A Technical Guide." ([kiriengine.app](https://www.kiriengine.app/blog/3d-gaussian-splatting-a-technical-guide-to-real-time-neural-rendering))
7. **Chen, G. and Wang, W.** "A Survey on 3D Gaussian Splatting." *arXiv:2401.03890*, 2024. ([arxiv.org](https://arxiv.org/html/2401.03890v7))
8. **FreeSplat** - "Generalizable 3D Gaussian Splatting." *NeurIPS 2024*. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/file/c2166d01fe4bcd694aba89f608737678-Paper-Conference.pdf))
9. **Mip-Splatting** - Yu, Z. et al. "Alias-free 3D Gaussian Splatting." *CVPR 2024*. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf))
10. **Frontiers in AI** - "Human reconstruction using 3D Gaussian Splatting: a brief survey." 2025. ([frontiersin.org](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1709229/full))
11. **PMC** - "Enhanced 3D Gaussian Splatting for Real-Scene Reconstruction." 2025. ([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656154/))
12. **OpenReview** - "Improving Sparse-View 3DGS Generalization via Flat Minima Optimization." 2025. ([openreview.net](https://openreview.net/forum?id=eH9Wlahibz))
13. **Nature Scientific Reports** - "Single view generalizable 3D reconstruction based on 3D Gaussian splatting." 2025. ([nature.com](https://www.nature.com/articles/s41598-025-03200-7))
14. **Fang, S. et al.** "NeRF Is a Valuable Assistant for 3D Gaussian Splatting." *ICCV 2025*. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/papers/Fang_NeRF_Is_a_Valuable_Assistant_for_3D_Gaussian_Splatting_ICCV_2025_paper.pdf))
15. **Radiance Fields** - 산업 동향 정리 사이트. ([radiancefields.com](https://radiancefields.com/))
16. **EVSplitting** - "An Efficient and Visually Consistent Splitting Algorithm." *SIGGRAPH Asia 2024*. ([dl.acm.org](https://dl.acm.org/doi/10.1145/3680528.3687592))
17. **PyImageSearch** - "3D Gaussian Splatting vs NeRF." 2024. ([pyimagesearch.com](https://pyimagesearch.com/2024/12/09/3d-gaussian-splatting-vs-nerf-the-end-game-of-3d-reconstruction/))
18. **arXiv** - "A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation." 2025. ([arxiv.org](https://arxiv.org/html/2508.09977v1))

> **참고**: 본 답변은 논문 원문, 공식 프로젝트 페이지, 학술 서베이, 그리고 신뢰할 수 있는 기술 블로그를 기반으로 작성되었습니다. 수식은 논문 및 후속 해설 자료에서 확인된 내용을 바탕으로 한 것이며, 특정 수치(예: FPS, 학습 시간 등)는 원 논문 및 Wikipedia의 정량적 요약에 근거합니다.

# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

## 1. 핵심 주장과 주요 기여 요약

이 논문은 실시간 radiance field 렌더링을 위한 혁신적인 접근법을 제시하며, 다음과 같은 핵심 주장과 기여를 담고 있습니다:[1]

**핵심 주장**: 기존 NeRF 방식의 고품질 렌더링과 최신 고속 방법의 효율성을 동시에 달성할 수 있다는 것입니다. 이는 3D Gaussian을 장면 표현의 기본 단위로 사용하고, 타일 기반 래스터화를 통해 가능해졌습니다.[1]

**주요 기여**:

1. **비등방성 3D Gaussian 표현**: 고품질의 비구조화된 radiance field 표현으로 3D Gaussian을 도입했습니다. 이는 미분 가능한 볼륨 표현의 속성을 유지하면서도 빠른 GPU 래스터화가 가능합니다.[1]

2. **적응형 밀도 제어를 통한 최적화**: 3D Gaussian의 위치, 불투명도 $$\alpha$$, 비등방성 공분산, 구면 조화(SH) 계수를 최적화하는 방법을 제안했습니다. 이 과정은 Gaussian의 추가 및 제거를 통한 적응형 밀도 제어와 교차로 진행됩니다.[1]

3. **고속 미분 가능 렌더러**: 가시성을 인식하고 비등방성 스플래팅을 지원하는 GPU 기반 렌더링 방식을 개발했습니다. 이는 빠른 역전파를 가능하게 하여 고품질 novel view synthesis를 달성합니다.[1]

**성능 요약**: 6분의 학습으로 PSNR 23.6을 달성하며 135fps로 렌더링하고, 51분 학습으로 PSNR 25.2를 달성하며 93fps로 렌더링합니다. 이는 Mip-NeRF360(48시간 학습, 0.071 fps)와 동등하거나 더 나은 품질을 보입니다.[1]

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**핵심 문제**: 기존 radiance field 방법들은 고품질과 실시간 렌더링 사이의 트레이드오프가 존재했습니다:[1]

- **NeRF 기반 방법**: Mip-NeRF360 같은 방법은 뛰어난 품질(PSNR 27.69)을 보이지만, 48시간의 학습 시간과 0.06 fps의 느린 렌더링 속도를 가집니다.[1]
- **고속 방법**: InstantNGP와 Plenoxels는 빠른 학습(5-7분)과 개선된 렌더링 속도(9-17 fps)를 제공하지만, 품질이 떨어지고(PSNR 21-25) 1080p 해상도에서 실시간 렌더링이 불가능합니다.[1]

**기술적 제약사항**:
- 볼륨 ray-marching은 많은 샘플링이 필요하여 계산 비용이 높고 노이즈를 발생시킵니다.[1]
- 구조화된 그리드 기반 가속 방법은 빈 공간 표현에 어려움을 겪습니다.[1]
- 기존 포인트 기반 방법은 MVS 데이터가 필요하여 over/under-reconstruction 문제를 상속받습니다.[1]

### 2.2 제안하는 방법과 수식

#### 3D Gaussian 표현

각 3D Gaussian은 평균 $$\mu$$와 공분산 행렬 $$\Sigma$$로 정의됩니다:[1]

$$
G(x) = e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}
$$

**공분산 행렬의 최적화 가능한 표현**: 직접 $$\Sigma$$를 최적화하면 양의 준정부호 제약이 위반될 수 있으므로, 스케일 행렬 $$S$$와 회전 행렬 $$R$$로 분해합니다:[1]

$$
\Sigma = RSS^TR^T
$$

여기서:
- $$s$$는 3D 스케일 벡터
- $$q$$는 회전을 나타내는 단위 쿼터니언

이 표현은 gradient descent에 적합하며 유효한 공분산 행렬을 보장합니다.[1]

#### 2D 투영

3D Gaussian을 화면 공간으로 투영하기 위해, viewing transformation $$W$$와 affine approximation의 Jacobian $$J$$를 사용합니다:[1]

$$
\Sigma' = JW\Sigma W^TJ^T
$$

여기서 $$\Sigma'$$는 카메라 좌표계에서의 공분산 행렬입니다. $$\Sigma'$$의 3번째 행과 열을 제거하면 2D 분산 행렬을 얻습니다.[1]

#### 렌더링 수식

픽셀 색상 $$C$$는 정렬된 $$N$$개의 Gaussian을 블렌딩하여 계산됩니다:[1]

$$
C = \sum_{i \in N} c_i\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)
$$

여기서:
- $$c_i$$는 각 포인트의 색상 (구면 조화 계수로 표현)
- $$\alpha_i$$는 2D Gaussian 평가와 학습된 불투명도의 곱

**손실 함수**: L1과 D-SSIM의 조합을 사용합니다:[1]

$$
\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}
$$

여기서 $$\lambda = 0.2$$를 모든 실험에서 사용합니다.[1]

#### 적응형 밀도 제어

**Densification 전략**: view-space positional gradient가 임계값 $$\tau_{pos} = 0.0002$$를 초과하는 Gaussian을 대상으로 합니다:[1]

1. **Clone (복제)**: 작은 Gaussian은 복제하여 위치 gradient 방향으로 이동시킵니다. 이는 under-reconstruction 영역을 처리합니다.[1]

2. **Split (분할)**: 큰 Gaussian($$\|S\| > \tau_S$$)은 두 개로 분할하고, 스케일을 $$\phi = 1.6$$으로 나눕니다. 원래 Gaussian을 PDF로 사용하여 새 위치를 샘플링합니다[1].

**Pruning (가지치기)**: 100 iteration마다 densification을 수행하고, 불투명도가 $$\epsilon_\alpha$$ 미만인 Gaussian을 제거합니다. 또한 3000 iteration마다 $$\alpha$$를 0에 가깝게 설정하여 필요한 Gaussian만 유지합니다.[1]

### 2.3 모델 구조 및 파이프라인

#### 초기화

Structure-from-Motion (SfM)으로 생성된 sparse point cloud로 시작합니다. 각 포인트에서 3D Gaussian을 생성하며, 초기 공분산 행렬은 가장 가까운 3개 포인트까지의 평균 거리를 축으로 하는 등방성 Gaussian으로 설정합니다.[1]

#### 타일 기반 래스터화

**핵심 설계**:[1]

1. **화면 분할**: 화면을 16×16 픽셀 타일로 분할합니다.

2. **Frustum culling**: 99% 신뢰 구간이 view frustum과 교차하는 Gaussian만 유지합니다.

3. **키 할당**: 각 Gaussian 인스턴스에 view space depth와 tile ID를 결합한 키를 할당합니다.

4. **GPU Radix sort**: 단일 고속 GPU Radix sort로 모든 Gaussian을 정렬합니다.

5. **타일당 렌더링**: 각 타일에 대해 thread block을 실행하여 front-to-back으로 색상과 $$\alpha$$ 값을 누적합니다. 픽셀이 포화($$\alpha \rightarrow 1$$)에 도달하면 종료합니다.

**역전파**: forward pass에서 사용된 정렬된 배열과 타일 범위를 재사용하여 back-to-front로 순회합니다. 최종 누적 불투명도만 저장하고, 각 포인트의 $$\alpha$$로 나누어 중간 불투명도를 복원합니다.[1]

**차별점**: 이전 방법(Pulsar)과 달리, gradient를 받는 Gaussian 수에 제한이 없으며, 픽셀당 일정한 메모리 오버헤드만 필요합니다.[1]

#### 최적화 세부사항

- **Warm-up**: 4배 작은 해상도로 시작하여 250, 500 iteration 후 업샘플링합니다.[1]
- **구면 조화 최적화**: 0차 성분부터 시작하여 1000 iteration마다 한 밴드씩 추가하여 4개 밴드까지 확장합니다.[1]
- **활성화 함수**: $$\alpha$$는 sigmoid, 공분산 스케일은 exponential 활성화를 사용합니다.[1]
- **학습률 스케줄링**: 위치에 대해서만 exponential decay를 적용합니다.[1]

### 2.4 성능 향상

#### 정량적 결과

**Mip-NeRF360 데이터셋** (평균 30K iterations):[1]
- **SSIM**: 0.815 (Mip-NeRF360: 0.792, InstantNGP-Big: 0.699)
- **PSNR**: 27.21 dB (Mip-NeRF360: 27.69 dB, InstantNGP-Big: 25.59 dB)
- **LPIPS**: 0.214 (Mip-NeRF360: 0.237, InstantNGP-Big: 0.331)
- **학습 시간**: 41분 33초 (Mip-NeRF360: 48시간)
- **FPS**: 134 (Mip-NeRF360: 0.06)

**Tanks&Temples 데이터셋** (30K iterations):[1]
- **SSIM**: 0.841, **PSNR**: 23.14 dB, **LPIPS**: 0.183
- **학습 시간**: 26분 54초, **FPS**: 154

**Deep Blending 데이터셋** (30K iterations):[1]
- **SSIM**: 0.903, **PSNR**: 29.41 dB, **LPIPS**: 0.243
- **학습 시간**: 36분 2초, **FPS**: 137

#### Ablation Study 결과

**Anisotropic covariance의 중요성**: isotropic Gaussian을 사용하면 평균 PSNR이 26.05에서 25.23으로 감소합니다. 비등방성은 표면 정렬을 크게 개선하여 동일한 포인트 수로 더 높은 품질을 달성합니다.[1]

**Densification 전략**: split 없이는 평균 PSNR이 23.90으로 감소하고(특히 배경 재구성 저하), clone 없이는 25.91로 감소합니다(얇은 구조 처리 어려움).[1]

**Gradient 제한 제거**: 10개 Gaussian으로 gradient를 제한하면 Truck 장면에서 PSNR이 11dB 감소합니다(22.71 → 14.66 at 5K iterations).[1]

**구면 조화**: SH 없이는 평균 PSNR이 25.35로 감소하여, view-dependent 효과 보상의 중요성을 보여줍니다.[1]

### 2.5 한계점

논문에서 명시한 주요 한계:[1]

1. **관찰되지 않은 영역의 아티팩트**: 장면이 잘 관찰되지 않은 영역에서 아티팩트가 발생합니다. Mip-NeRF360도 유사한 문제를 겪습니다.

2. **Elongated/splotchy Gaussians**: 비등방성 Gaussian의 장점에도 불구하고, 때때로 길쭉하거나 얼룩진 Gaussian이 생성됩니다.

3. **Popping artifacts**: 큰 Gaussian 생성 시 popping 아티팩트가 발생할 수 있습니다. 이는:
   - 래스터화기의 trivial guard band rejection
   - 단순한 가시성 알고리즘으로 인한 갑작스러운 depth/blending order 변경 때문입니다.

4. **정규화 부재**: 현재 최적화에 정규화를 적용하지 않습니다. 정규화는 위의 문제들을 완화할 수 있습니다.

5. **대규모 장면의 하이퍼파라미터**: 동일한 하이퍼파라미터를 모든 평가에 사용했지만, 초기 실험에서 매우 큰 장면(예: 도시 데이터셋)에서는 위치 학습률 감소가 필요할 수 있습니다.

6. **높은 메모리 소비**: 이전 포인트 기반 방법보다는 compact하지만, NeRF 기반 솔루션보다 메모리 소비가 현저히 높습니다:
   - **학습 시**: 대규모 장면에서 20GB 이상의 GPU 메모리 필요
   - **렌더링 시**: 모델 저장에 수백 MB + 래스터화에 30-500 MB 필요
   - 그러나 point cloud 압축 기술을 적용하여 개선 가능성이 있습니다.[1]

## 3. 모델의 일반화 성능 향상

### 3.1 다양한 장면 타입에서의 일반화

논문은 다음과 같은 다양한 장면 타입에서 일관된 성능을 보여줍니다:[1]

**실내 bounded 장면**: Room, Counter, Kitchen 장면에서 PSNR 28.7-30.6 달성[1]

**실외 unbounded 장면**: Bicycle, Garden, Stump 장면에서 PSNR 21.5-27.4 달성[1]

**합성 장면**: 100K 랜덤 초기화로 시작하여 NeRF-synthetic 데이터셋에서 평균 PSNR 33.32 달성 (Point-NeRF: 33.30, Mip-NeRF: 33.09)[1]

### 3.2 초기화 유연성

**SfM 포인트 없이도 작동**: 랜덤 초기화 ablation에서, 방법이 완전 실패를 피하고 합리적인 성능을 유지합니다. SfM 초기화 대비 품질 저하가 있지만(평균 PSNR: 20.42 vs 26.05 at 30K), 주로 배경에서 발생합니다.[1]

**합성 장면에서의 강건성**: NeRF-synthetic 데이터셋에서 exact camera parameters와 exhaustive view set으로 인해 랜덤 초기화로도 state-of-the-art 결과 달성합니다.[1]

### 3.3 Compactness와 효율성

**적은 primitive 수로 고품질 달성**: Zhang et al. 의 highly compact point-based 모델과 비교하여, 약 1/4의 포인트 수로 동일한 PSNR 달성 (평균 모델 크기: 3.8 MB vs 9 MB).[1]

**장면당 Gaussian 수**: 모든 테스트 장면에서 1-5백만 Gaussian으로 합리적으로 compact한 표현 달성. 합성 장면은 30K iteration 후 200-500K Gaussian으로 수렴합니다.[1]

### 3.4 일반화 제한사항 및 개선 방향

**현재 제한사항**:

1. **Angular coverage 의존성**: 각도 정보가 부족한 캡처(예: 장면 코너, inside-out 캡처)에서 SH 0차 성분이 부정확할 수 있습니다. 이를 완화하기 위해 progressive SH band 최적화를 사용합니다.[1]

2. **MVS 데이터 불필요**: 대부분 포인트 기반 방법과 달리 MVS 데이터가 필요하지 않지만, SfM sparse points에 여전히 의존합니다.[1]

**일반화 향상 메커니즘**:

- **적응형 밀도 제어**: clone과 split 전략은 다양한 기하학적 복잡도에 자동으로 적응합니다.[1]
- **동일한 하이퍼파라미터**: 모든 평가에서 동일한 하이퍼파라미터 설정을 사용하여 강건성을 입증했습니다.[1]

## 4. 향후 연구에 미치는 영향과 고려사항 (최신 연구 기반)

### 4.1 후속 연구 동향 (2024-2025)

3D Gaussian Splatting은 2023년 발표 이후 폭발적인 연구 성장을 보였습니다. 최신 연구들은 다음 방향으로 발전하고 있습니다:

#### 동적 장면 확장

**BARD-GS (2025)**: 모션 블러와 부정확한 카메라 포즈를 처리하는 robust dynamic scene reconstruction을 제안합니다. 카메라 모션 블러와 객체 모션 블러를 명시적으로 분리하여 모델링합니다.[2]

**Sliding Windows for Dynamic 3DGS (ECCV)**: temporally-local dynamic MLP를 사용하여 동적 장면을 재구성합니다. 각 sliding window에 대해 별도의 canonical representation을 학습하여 significant geometric changes를 처리합니다.[3]

**Temporally Compressed 3DGS (TC3DGS, 2024)**: 동적 3D Gaussian 표현을 효과적으로 압축하여 AR/VR, 게임에 적합하도록 합니다. 최대 67배 압축을 달성하면서도 복잡한 동작을 고충실도로 표현합니다.[4]

#### 표현 방식 개선

**3D Convex Splatting (2024)**: Gaussian 대신 3D smooth convexes를 primitive로 사용합니다. Hard edges와 dense volumes를 더 적은 primitive로 표현할 수 있으며, 3DGS 대비 PSNR 0.81, LPIPS 0.026 개선을 달성합니다.[5]

**Deformable Beta Splatting (2025)**: Beta distribution을 사용한 새로운 radiance field 표현으로 3DGS보다 큰 장점을 제공합니다.[6]

**Triangle Splatting (2025)**: 실시간 radiance field 렌더링을 위한 또 다른 새로운 표현 방법입니다.[6]

#### 압축 및 효율성

**Compression in 3DGS Survey (2025)**: 3DGS의 압축 방법, 트렌드, 향후 방향을 종합적으로 조사합니다. 효율적인 NeRF 표현의 발전이 향후 3DGS 최적화에 영감을 줄 수 있다고 제안합니다.[7]

**DOGS (2024)**: 대규모 3D 재구성을 위한 distributed-oriented Gaussian Splatting을 제안합니다. Scene decomposition을 통해 학습 시간을 6배 이상 가속화합니다.[8]

#### 강건성 향상

**Robust Gaussian Splatting (2024)**: blur, 부정확한 카메라 포즈, 색상 불일치 등 일반적인 오류 소스를 해결합니다. Scannet++과 Deblur-NeRF 벤치마크에서 state-of-the-art 결과를 달성합니다.[9]

**GP-GS (2025)**: Gaussian Processes를 통합하여 sparse SfM point cloud의 한계를 극복합니다. Dense point clouds를 생성하여 고품질 초기 3D Gaussian을 제공합니다.[10]

**SelfSplat (2025)**: Pose-free와 3D prior-free generalizable 3D reconstruction을 수행합니다. Self-supervised depth와 pose estimation을 통합하여 상호 개선을 달성합니다.[11]

#### 응용 분야 확장

**Extended Reality (XR)**: 3DGS의 XR 적용에 대한 연구가 증가하고 있습니다. SAGE (2025)는 semantic-driven adaptive Gaussian Splatting을 XR에 적용합니다.[12][13]

**자율주행**: Dynamic radiance field framework가 자율주행 시나리오에 특화되어 개발되고 있습니다. SDF 기반 formulation을 통해 동적 객체와 정적 배경을 효과적으로 분리합니다.[14]

**물리 기반 시뮬레이션**: Physics-integrated Gaussian Splatting을 통한 editable dynamic scene modeling이 제안되었습니다. 물리적 속성을 임베딩하여 realistic, complex motion modeling을 가능하게 합니다.[15]

### 4.2 향후 연구 시 고려사항

#### 1. 메모리 효율성 개선

**압축 기술 통합**: Point cloud 압축은 잘 연구된 분야이며, 이러한 접근법을 3DGS 표현에 적용할 수 있는 많은 기회가 있습니다. 최신 연구들은 gradient-aware mixed-precision quantization 같은 기술을 탐구하고 있습니다.[4][1]

**적응형 해상도**: 장면의 복잡도에 따라 Gaussian 밀도를 동적으로 조절하는 방법이 필요합니다.

#### 2. 정규화 및 Prior 통합

**기하학적 prior**: 현재 방법은 정규화를 적용하지 않지만, depth maps나 다른 geometric priors를 통합하면 unseen regions와 popping artifacts를 완화할 수 있습니다.[10][1]

**Semantic information**: Semantic-driven approaches는 더 의미 있는 장면 분해를 가능하게 합니다.[13][16]

#### 3. 동적 장면 처리

**Temporal consistency**: 동적 장면에서 temporal flickering을 방지하고 일관된 motion modeling을 보장하는 것이 중요합니다.[2][3]

**Physics-based constraints**: 물리 법칙을 통합하면 더 현실적인 동작 예측과 편집이 가능합니다.[15]

#### 4. 일반화 능력 향상

**Pose estimation 통합**: Pose-free 방법은 실제 애플리케이션에서 매우 유용하며, self-supervised 기술과의 통합이 promising합니다.[11]

**Multi-scale representation**: Multi-scale Gaussian이나 hierarchical representations는 다양한 장면 스케일에 더 잘 적응할 수 있습니다.[17]

#### 5. 렌더링 품질 개선

**Antialiasing**: 간단한 visibility 알고리즘을 개선하고 antialiasing을 추가하면 popping artifacts를 완화할 수 있습니다.[1]

**Advanced rasterization**: 더 정교한 culling과 blending 전략이 시각적 품질을 향상시킬 수 있습니다.[5]

#### 6. 실용적 애플리케이션

**Real-time editing**: 실시간 장면 조작과 편집 기능은 XR과 게임 애플리케이션에 필수적입니다.[16][12]

**Hardware optimization**: Mobile devices와 저전력 플랫폼에서의 효율적인 실행을 위한 최적화가 필요합니다.[4]

### 4.3 연구 커뮤니티에 미친 영향

3D Gaussian Splatting은 radiance field 연구의 패러다임 전환을 촉발했습니다:[17]

1. **Explicit vs Implicit**: "연속 표현이 고품질 radiance field 학습에 필수적"이라는 통념을 깨뜨렸습니다.[1]

2. **Real-time rendering**: 처음으로 SOTA 품질과 실시간 렌더링을 동시에 달성했습니다.[1]

3. **Editability**: Explicit representation은 전례 없는 수준의 장면 제어와 편집 가능성을 제공합니다.[17]

4. **다양한 응용**: XR, 자율주행, 로봇공학 등 다양한 분야에서 실질적 영향을 미치고 있습니다.[12][13][14]

### 4.4 결론

3D Gaussian Splatting은 radiance field rendering의 혁신적 돌파구를 제공했으며, 실시간 고품질 렌더링이라는 오랜 과제를 해결했습니다. 2024-2025년 최신 연구들은 동적 장면, 압축, 강건성, XR 응용 등 다양한 방향으로 빠르게 발전하고 있습니다.[7][8][13][9][3][10][12][11][2][4][1]

향후 연구는 메모리 효율성, 일반화 능력, 동적 장면 처리, 실용적 애플리케이션 최적화에 집중해야 합니다. 특히 물리 기반 시뮬레이션, semantic information 통합, pose-free 방법, 그리고 다양한 표현 방식(convex, beta distribution 등)의 탐구가 promising한 방향입니다.[11][15][6][5]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/48b06790-56b2-490d-9764-e2a02a800fee/2308.04079v1.pdf)
[2](https://arxiv.org/abs/2503.15835)
[3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07170.pdf)
[4](https://arxiv.org/abs/2412.05700)
[5](https://arxiv.org/abs/2411.14974)
[6](https://radiancefields.com/research)
[7](https://arxiv.org/html/2502.19457v1)
[8](https://arxiv.org/abs/2405.13943)
[9](https://arxiv.org/abs/2404.04211)
[10](https://arxiv.org/html/2502.02283v3)
[11](https://arxiv.org/html/2411.17190)
[12](http://arxiv.org/pdf/2412.06257.pdf)
[13](http://arxiv.org/pdf/2503.16747.pdf)
[14](https://www.sciencedirect.com/science/article/abs/pii/S0167865525002752)
[15](https://viplab.snu.ac.kr/viplab/courses/mlvu_2024_1/projects/11.pdf)
[16](https://arxiv.org/html/2503.11601v1)
[17](https://arxiv.org/html/2401.03890v8)
[18](https://github.com/Lee-JaeWon/2025-Arxiv-Paper-List-Gaussian-Splatting)
[19](https://www.sciencedirect.com/science/article/abs/pii/S092523122502301X)
[20](https://isprs-archives.copernicus.org/articles/XLVIII-G-2025/891/2025/isprs-archives-XLVIII-G-2025-891-2025.pdf)
[21](https://arxiv.org/html/2511.06408v1)
[22](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)
[23](https://www.themoonlight.io/ko/review/temporally-compressed-3d-gaussian-splatting-for-dynamic-scenes)
[24](https://dl.acm.org/doi/10.1145/3687897)
[25](https://ieeexplore.ieee.org/iel8/6287639/10820123/10884729.pdf)
[26](https://github.com/hustvl/4DGaussians)
[27](https://ieeexplore.ieee.org/iel8/76/11223720/11016927.pdf)
[28](https://dl.acm.org/doi/10.1145/3728302)
[29](https://dynamic3dgaussians.github.io)
