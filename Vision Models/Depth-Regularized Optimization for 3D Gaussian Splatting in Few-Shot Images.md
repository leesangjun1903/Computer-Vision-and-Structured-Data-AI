# Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
3D Gaussian Splatting(3DGS)은 소수의 이미지만 사용할 경우 **강한 지역성(locality)**으로 인해 과적합(overfitting)과 부유 아티팩트(floating artifacts)가 발생한다. 본 논문은 **사전 학습된 단안 깊이 추정(monocular depth estimation) 모델**로부터 얻은 밀집 깊이 맵(dense depth map)을 기하학적 가이드로 활용하여, 소수 이미지 환경에서도 안정적인 3DGS 최적화를 달성할 수 있음을 주장한다.

### 주요 기여
1. **깊이 정규화 기반 3DGS 최적화 전략**: COLMAP의 희소 포인트 클라우드로 스케일 보정된 밀집 깊이를 기하학적 정규화 도구로 활용
2. **조기 종료(Early Stop) 전략**: 깊이 가이드 손실이 상승하기 시작할 때 학습을 중단하여 과적합 방지
3. **비지도 평활도 제약(Unsupervised Smoothness Constraint)**: 인접 픽셀 간 깊이 일관성을 보장하여 올바른 기하 구조 탐색 유도

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

3DGS는 수백만 개의 독립적인 가우시안 스플랫(Gaussian splat)으로 3D 장면을 표현하며, 다중 뷰 컬러 감독(multi-view color supervision)을 통해 최적화된다. 그러나:

- **글로벌 구조 부재**: 각 스플랫이 독립적으로 최적화되어 전역 기하학적 제약이 없음
- **소수 이미지 문제**: 이미지 수가 적으면 기하학적 정보가 부족하여 **지역 최적해(local optimum)**에 수렴, 최적화 실패, 또는 부유 아티팩트 발생
- **COLMAP 희소 포인트의 한계**: 19장 이미지로부터 SfM을 수행해도 평균 0.04%의 유효 픽셀만 생성되며, 소수 이미지에서는 더욱 부족

### 2.2 제안하는 방법

#### (A) 밀집 깊이 사전 정보 준비 (Section 3.1)

**SfM 기반 희소 깊이 맵 생성**: COLMAP으로부터 카메라 포즈 $\mathbf{R}_i \in \mathbb{R}^{3 \times 3}$, $\mathbf{t}_i \in \mathbb{R}^3$, 내부 파라미터 $K_i \in \mathbb{R}^{3 \times 3}$, 포인트 클라우드 $P \in \mathbb{R}^{n \times 3}$를 획득하고, 포인트를 이미지 평면에 투영하여 희소 깊이를 얻는다:

$$p = P_{\text{homog}}[\mathbf{R}_i | \mathbf{t}_i]$$

$$D_{\text{sparse},i} = p_z \in [0, \infty]^{H \times W}$$

**단안 깊이 추정 및 스케일 보정**: 사전 학습된 단안 깊이 추정 모델 $F_\theta$ (ZoeDepth)로부터 밀집 깊이를 추정한다:

$$D_{\text{dense}} = s \cdot F_\theta(I) + t$$

스케일 모호성(scale ambiguity)을 해결하기 위해, 추정된 밀집 깊이를 COLMAP 희소 깊이에 맞추는 최소 자승 피팅을 수행한다:

$$s^\*, t^* = \arg\min_{s,t} \sum_{p \in D_{\text{sparse}}} \left\| w(p) \cdot D_{\text{sparse}}(p) - D_{\text{dense}}(p; s, t) \right\|^2$$

여기서 $w \in [0,1]$은 SfM 재투영 오차의 역수를 정규화한 가중치이다. 보정된 밀집 깊이는:

```math
D^*_{\text{dense}} = s^* \cdot F_\theta(I) + t^*
```

#### (B) 래스터화를 통한 깊이 렌더링 (Section 3.2)

3DGS의 α-블렌딩 래스터화 파이프라인을 활용하여 컬러와 동일한 방식으로 깊이를 렌더링한다. 픽셀 컬러 렌더링:

$$C = \sum_{i \in N} c_i \alpha_i T_i, \quad \text{where } T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$$

깊이 렌더링은 $c_i$를 $d_i$로 대체하여:

$$D = \sum_{i \in N} d_i \alpha_i T_i$$

여기서 $d_i = (\mathbf{R}_i \mathbf{p}_i + \mathbf{t}_i)_z$는 카메라로부터 각 스플랫의 깊이이다. 이렇게 렌더링된 깊이와 보정된 밀집 깊이 사이의 L1 거리로 깊이 손실을 정의한다:

$$\mathcal{L}_{\text{depth}} = \left\| D - D^*_{\text{dense}} \right\|_1$$

#### (C) 비지도 평활도 제약 (Section 3.3)

독립적으로 추정된 깊이 간의 충돌을 완화하기 위해, 인접 픽셀 간 깊이 차이를 정규화한다:

$$\mathcal{L}_{\text{smooth}} = \sum_{d_j \in \text{adj}(d_i)} \mathbb{1}_{ne}(d_i, d_j) \cdot \left\| d_i - d_j \right\|^2$$

여기서 $\mathbb{1}_{ne}$는 두 깊이 값이 모두 **에지 영역이 아닌** 경우를 나타내는 지시 함수이며, Canny edge detector를 마스크로 사용하여 경계 영역에서의 과도한 평활화를 방지한다.

#### (D) 최종 손실 함수

$$\mathcal{L} = (1 - \lambda_{\text{ssim}}) \mathcal{L}_{\text{color}} + \lambda_{\text{ssim}} \mathcal{L}_{\text{D-SSIM}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}$$

#### (E) Few-Shot 학습을 위한 수정 사항 (Section 3.4)

1. **구면 조화 함수(SH) 최대 차수를 1로 제한**: 불충분한 정보로 인한 고주파 과적합 방지
2. **깊이 손실 기반 조기 종료**: 이동 평균된 깊이 손실이 상승할 때 최적화 중단
3. **주기적 불투명도 리셋 제거**: 소수 이미지 환경에서 불투명도 리셋이 비가역적 손상을 유발하므로 제거

### 2.3 모델 구조

본 논문은 새로운 신경망 아키텍처를 제안하는 것이 아니라, **기존 3DGS 파이프라인 위에 깊이 정규화 모듈을 추가**하는 접근 방식이다:

```
[소수 이미지] → [COLMAP(SfM)] → [카메라 포즈 + 희소 포인트 클라우드]
                                          ↓
[단안 깊이 추정(ZoeDepth)] → [밀집 깊이] → [스케일 보정] → [보정된 밀집 깊이 D*_dense]
                                          ↓
[3DGS 최적화] ← [컬러 손실 + 깊이 손실 + 평활도 손실] + [조기 종료]
```

- 깊이 래스터라이저는 CUDA 기반으로 구현하여, 컬러 래스터화에서 계산된 $\alpha_i$와 $T_i$를 재사용함으로써 최소한의 추가 연산 부담

### 2.4 성능 향상

NeRF-LLFF 데이터셋에서 2~5-view 설정으로 평가한 결과 (평균값):

| 방법 | 2-view PSNR↑ | 3-view PSNR↑ | 4-view PSNR↑ | 5-view PSNR↑ |
|------|-------------|-------------|-------------|-------------|
| 3DGS | 12.25 | 13.75 | 15.26 | 16.17 |
| **Ours** | **15.94** | **17.17** | **18.15** | **18.74** |
| Oracle | 18.29 | 19.95 | 21.02 | 22.05 |

- **2-view 기준 PSNR 약 +3.69dB 향상** (12.25 → 15.94)
- 특히 실내 장면(Fortress, Room, Fern)에서 큰 폭의 성능 향상
- Fortress 2-view: 13.87 → 19.80 (**+5.93dB**)
- Room 2-view: 10.18 → 17.21 (**+7.03dB**)

#### Ablation Study 결과 (Horns 장면):

| 방법 | 2-view PSNR | 5-view PSNR |
|------|------------|------------|
| w/o Adjustment | 7.86 | 10.01 |
| w/o $\mathcal{L}_{\text{depth}}$ | 11.49 | 12.97 |
| w/o $\mathcal{L}_{\text{smooth}}$ | 14.75 | 17.79 |
| w/o early stop | 13.99 | 17.28 |
| **Full (Ours)** | **15.91** | **18.39** |

→ 스케일 보정이 가장 중요하며, 모든 구성 요소가 성능 향상에 기여함을 확인

### 2.5 한계

1. **단안 깊이 추정 모델에 대한 강한 의존성**: ZoeDepth가 학습된 도메인(NYU Depth v2 실내, KITTI 도시)과 다른 장면에서는 성능 저하 (예: Orchids, Flower 등 자연 장면)
2. **COLMAP 성능 의존**: 텍스처가 없는 평면이나 도전적인 표면에서 COLMAP이 실패하면 연쇄적으로 성능 저하
3. **밀집 깊이의 부정확성**: 단안 추정 깊이는 여전히 거칠고(coarse), 세밀한 디테일을 포착하지 못함 (Oracle과의 성능 차이: 2~3dB)
4. **하늘 등 깊이 추정이 어려운 영역**에 대한 처리 미비

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화의 병목

본 논문의 일반화 성능은 크게 **두 가지 외부 모듈**에 의해 제약된다:

**(1) 단안 깊이 추정 모델의 도메인 편향**
- ZoeDepth는 NYU Depth v2 (실내)와 KITTI (도시 주행)에서 학습되었으므로, 이 도메인과 유사한 장면(Fortress, Room)에서는 높은 성능을, 자연 장면(Orchids, Flower)에서는 상대적으로 낮은 성능을 보임
- 논문에서 명시적으로 언급: *"our model reports relatively higher performance in indoor scenes (Fortress, Room, Fern) and comparatively worse results for natural scenes (Orchids, Flower)"*

**(2) COLMAP의 희소 포인트 품질**
- Leaves 장면의 경우 COLMAP 자체가 어려움을 겪어 전반적으로 학습 실패
- 소수 이미지에서 3개 이상의 뷰에서 관찰 가능한 특징점만 사용하므로, 유효 포인트가 극히 적을 수 있음

### 3.2 일반화 성능 향상을 위한 방향

**(1) Foundation Model 기반 깊이 추정 활용**
- Depth Anything (Yang et al., 2024), Depth Anything V2 (Yang et al., 2024), Marigold (Ke et al., 2024) 등 최신 foundation depth model은 다양한 도메인에서 강건한 상대 깊이를 제공
- 이러한 모델로 ZoeDepth를 대체하면 도메인 의존성을 크게 완화 가능

**(2) 뷰 간 일관성(Cross-view Consistency) 강화**
- 현재 각 이미지의 깊이를 독립적으로 추정하고 COLMAP 포인트로 개별 보정하므로, 뷰 간 기하학적 충돌이 발생 가능
- 다중 뷰 깊이 일관성 제약을 추가하거나, 최적화 과정에서 깊이 맵을 공동으로 갱신하는 전략이 일반화에 기여 가능

**(3) 다양한 장면 유형으로의 확장**
- 현재 NeRF-LLFF (forward-facing) 데이터셋에서만 평가되었으며, 360° 장면(Mip-NeRF 360 데이터셋), 대규모 실외 장면, 동적 장면 등으로의 확장이 필요
- Oracle 실험 결과가 보여주듯, 정확한 깊이가 제공되면 성능이 대폭 향상되므로, 깊이 추정 모델의 일반화가 곧 전체 파이프라인의 일반화로 직결

**(4) COLMAP-free 접근**
- 논문 자체에서 future work로 언급: COLMAP 포인트 대신 독립적으로 추정된 깊이들 간의 상호 보정을 통해 COLMAP 의존성 제거
- DUSt3R (Wang et al., 2024) 등 포즈 추정과 밀집 재구성을 동시에 수행하는 최신 기법과의 통합이 유망

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**(1) 3DGS의 Few-Shot 최적화 연구의 촉발**
- 이 논문은 3DGS가 소수 이미지에서 근본적으로 과적합에 취약하다는 문제를 처음으로 체계적으로 다루며, 이후 FSGS (Zhu et al., 2024), DNGaussian (Li et al., 2024), SparseGS (Xiong et al., 2024) 등 다수의 후속 연구에 영향

**(2) 깊이 사전 정보의 유효성 확인**
- "거친(coarse) 깊이도 기하학적 가이드로서 효과적"이라는 실험적 발견은, 이후 연구에서 다양한 깊이 소스(확산 모델 기반, stereo 기반 등)를 활용하는 동기 부여

**(3) 포인트 기반 표현에서의 정규화 전략**
- NeRF의 글로벌 MLP와 달리 지역적(local) 표현인 3DGS에서의 정규화 방법론을 제시하여, 유사한 지역적 표현(sparse voxel, feature point cloud 등)에도 적용 가능한 범용적 프레임워크 제공

### 4.2 향후 연구 시 고려할 점

1. **깊이 추정 모델의 선택과 도메인 적응**: 타겟 장면의 특성에 맞는 깊이 모델 선택이 최종 성능에 결정적 영향
2. **깊이 손실의 가중치 스케줄링**: 깊이 정규화가 너무 강하면 컬러 재현을 방해하고, 너무 약하면 기하 가이드 효과가 미미 → 적응적 가중치 전략 필요
3. **초기화 포인트의 영향**: Table 3에서 확인되듯, 초기 포인트 클라우드의 품질이 최종 성능에 큰 영향 → 초기화 전략 연구 필요
4. **스케일 보정의 강건성**: 소수 이미지에서 COLMAP 포인트가 극히 적을 때 스케일 피팅의 신뢰성 보장 방안
5. **평가 다양성**: NeRF-LLFF 외에 다양한 벤치마크에서의 검증 필요 (Mip-NeRF 360, DTU, Tanks and Temples 등)

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Few-Shot NeRF 계열

| 연구 | 연도 | 핵심 아이디어 | 본 논문과의 관계 |
|------|------|------------|-------------|
| **RegNeRF** (Niemeyer et al.) | CVPR 2022 | 랜덤 뷰에서의 렌더링 정규화 + depth smoothness | NeRF 기반; MLP의 글로벌성에 의존하므로 3DGS에 직접 적용 어려움 |
| **DietNeRF** (Jain et al.) | ICCV 2021 | CLIP 기반 의미적 일관성 제약 | 의미적 정규화는 기하학적 정규화와 상보적 |
| **InfoNeRF** (Kim et al.) | CVPR 2022 | Ray entropy 최소화 | 볼륨 렌더링 기반; 래스터화 기반 3DGS와 호환 어려움 |
| **Depth-supervised NeRF** (Deng et al.) | CVPR 2022 | COLMAP 희소 깊이 감독 | 본 논문의 motivation과 유사하나 NeRF 기반이며 밀집 깊이 미사용 |
| **Dense Depth Priors** (Roessle et al.) | CVPR 2022 | COLMAP 포인트 → 깊이 완성 네트워크 → NeRF 감독 | 깊이 완성 방식의 차이; 본 논문은 단안 추정 + 스케일 보정 |

### 5.2 Few-Shot 3DGS 계열 (본 논문 이후 발전)

| 연구 | 연도 | 핵심 아이디어 | 본 논문 대비 발전 |
|------|------|------------|-------------|
| **FSGS** (Zhu et al.) | CVPR 2024 | Proximity-guided Gaussian Unpooling + 확산 모델 기반 뷰 증강 | 깊이 + 생성 모델 결합; 더 적극적인 데이터 증강 |
| **DNGaussian** (Li et al.) | CVPR 2024 | Depth-regularized + Hard/Soft depth regularization | 깊이 정규화의 정교한 이중 전략 |
| **SparseGS** (Xiong et al.) | 2024 | CLIP 기반 floater 제거 + depth distortion 정규화 | 의미적 + 기하학적 정규화 통합 |
| **InstantSplat** (Fan et al.) | 2024 | DUSt3R 기반 포즈-프리 초기화 → 3DGS | COLMAP 의존성 완전 제거 |
| **DepthSplat** (Xu et al.) | 2024 | Feed-forward 방식의 깊이 기반 3DGS | 최적화 없이 직접 예측; 일반화 성능 극대화 |

### 5.3 깊이 추정 모델의 발전

| 모델 | 연도 | 특징 | 본 논문 적용 시 기대 효과 |
|------|------|------|-----------------|
| **ZoeDepth** (Bhat et al.) | 2023 | 상대+절대 깊이 결합 | 본 논문에서 사용 |
| **Depth Anything** (Yang et al.) | CVPR 2024 | 대규모 라벨리스 학습, 강건한 상대 깊이 | 도메인 일반화 대폭 향상 |
| **Depth Anything V2** (Yang et al.) | 2024 | 합성 데이터 학습, metric depth 지원 | 스케일 보정 필요성 감소 가능 |
| **Marigold** (Ke et al.) | CVPR 2024 | Stable Diffusion 기반 affine-invariant depth | 확산 모델의 사전 지식 활용; 자연 장면에서도 강건 |
| **Metric3D v2** (Hu et al.) | 2024 | 범용 metric depth | COLMAP 없이도 절대 스케일 깊이 제공 가능 |

### 5.4 핵심 비교 분석

본 논문은 3DGS의 few-shot 문제를 **최초로 체계적으로 다룬 연구** 중 하나이다. 이후 연구들은 본 논문의 깊이 정규화 아이디어를 기반으로 다음과 같이 발전시켰다:

- **더 정교한 깊이 정규화**: DNGaussian의 hard/soft regularization
- **생성 모델과의 결합**: FSGS의 확산 모델 기반 뷰 증강
- **COLMAP 의존성 제거**: InstantSplat의 DUSt3R 기반 접근
- **Feed-forward 방식**: DepthSplat, pixelSplat 등 최적화 없이 직접 예측

본 논문의 가장 큰 가치는 **"거친 깊이라도 기하학적 가이드로서 효과적"**이라는 실증적 발견과, 3DGS의 강한 지역성을 극복하기 위한 **깊이 정규화 프레임워크의 기초**를 제공한 데 있다.

---

## 참고자료

1. Chung, J., Oh, J., & Lee, K. M. (2024). "Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images." arXiv:2311.13398v3.
2. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." ACM TOG (SIGGRAPH 2023).
3. Bhat, S. F., et al. (2023). "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth." arXiv:2302.12288.
4. Niemeyer, M., et al. (2022). "RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs." CVPR 2022.
5. Deng, K., et al. (2022). "Depth-supervised NeRF: Fewer Views and Faster Training for Free." CVPR 2022.
6. Roessle, B., et al. (2022). "Dense Depth Priors for Neural Radiance Fields from Sparse Input Views." CVPR 2022.
7. Zhu, Z., et al. (2024). "FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting." CVPR 2024.
8. Li, J., et al. (2024). "DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization." CVPR 2024.
9. Yang, L., et al. (2024). "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data." CVPR 2024.
10. Ke, B., et al. (2024). "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation (Marigold)." CVPR 2024.
11. Schönberger, J. L., & Frahm, J. M. (2016). "Structure-from-Motion Revisited." CVPR 2016.
12. Godard, C., Mac Aodha, O., & Brostow, G. J. (2017). "Unsupervised Monocular Depth Estimation with Left-Right Consistency." CVPR 2017.
13. Kim, M., Seo, S., & Han, B. (2022). "InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering." CVPR 2022.
