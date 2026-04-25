
# ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Model

> **논문 정보**
> - **제목**: ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Model
> - **저자**: Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, Yueqi Duan
> - **arXiv**: [2408.16767](https://arxiv.org/abs/2408.16767) (2024년 8월 제출)
> - **프로젝트 페이지**: https://liuff19.github.io/ReconX
> - **GitHub**: https://github.com/liuff19/ReconX
> - **OpenReview (ICLR 2025 제출)**: https://openreview.net/forum?id=Z30Mdbv5jO

---

## 1. 핵심 주장과 주요 기여 (Executive Summary)

3D 장면 재구성 분야에서 밀집된(dense) 입력 뷰 시나리오는 큰 성공을 거뒀지만, 부족한 뷰에서 세밀한 장면을 렌더링하는 것은 여전히 ill-posed 최적화 문제로 아티팩트와 왜곡을 유발한다. 이를 해결하기 위해 ReconX는 이 모호한 재구성 문제를 **시간적(temporal) 생성 태스크**로 재정의(reframe)하는 새로운 3D 장면 재구성 패러다임을 제안한다. 핵심 통찰은 대규모 사전 학습된 비디오 확산 모델(video diffusion model)의 강력한 생성적 사전(generative prior)을 Sparse-view 재구성에 활용하는 것이다.

### 주요 기여 요약

| 기여 | 내용 |
|---|---|
| **패러다임 전환** | Sparse-view 3D 재구성을 생성 문제로 재정의 |
| **3D 구조 조건화** | 글로벌 포인트 클라우드 → 비디오 확산 조건으로 주입 |
| **3D 일관성 생성** | 비디오 프레임에서 3D 뷰 일관성 확보 |
| **신뢰도 기반 최적화** | Confidence-aware 3D Gaussian Splatting 최적화 |
| **일반화 성능** | 타 데이터셋 zero-shot 일반화 우수 성능 |

---

## 2. 해결하고자 하는 문제

### 2.1 문제 정의

Sparse-view 재구성은 ill-posed 문제로, 두 장의 이미지처럼 매우 제한된 시점 정보에서 복잡한 3D 구조를 복원해야 하며, 이는 여러 해(solution)에 대응될 수 있다.

최근 효율적이고 표현력 높은 3DGS를 활용한 feed-forward Gaussian Splatting 방법들이 제안되어 왔다. 그러나 이러한 방법들은 에피폴라 트랜스포머 등의 feature 추출 모듈로 장면 사전 지식을 학습하여 보간(interpolation) 결과를 낼 수 있지만, 불충분한 장면 캡처로 인해 ill-posed 최적화 문제가 지속된다. 결과적으로 특히 미관측 영역에서 심각한 아티팩트와 비현실적 이미지 문제가 발생한다.

### 2.2 핵심 도전 과제

사전 학습된 모델로 직접 생성된 비디오 프레임에서는 3D 뷰 일관성(3D view consistency)을 정확히 보존하기 어렵다는 문제가 있다.

---

## 3. 제안하는 방법 (수식 포함)

### 3.1 전체 파이프라인

이론적 분석을 바탕으로, 3D 구조 조건을 비디오 생성 과정에 통합하는 가능성을 탐구한다. 이는 under-determined 3D 생성 문제와 완전히 관측된 3D 재구성 환경 사이의 간극을 메운다. 구체적으로, 스파스 이미지들로부터 포즈-프리(pose-free) 스테레오 재구성 방법을 통해 글로벌 포인트 클라우드를 구축한다. 그런 다음 이를 풍부한 컨텍스트 표현 공간으로 인코딩하여 cross-attention 레이어에서 3D 조건으로 사용한다. 마지막으로 생성된 비디오로부터 3D confidence-aware Gaussian Splatting 최적화 스킴을 통해 3D 장면을 재구성한다.

파이프라인은 크게 **세 단계**로 구성된다:

---

### Stage 1: 글로벌 포인트 클라우드 구축

프레임워크에서 DUSt3R(Wang et al., 2024)을 포즈-없는(unconstrained) 스테레오 3D 재구성 백본으로, I2V(Image-to-Video) 모델인 DynamiCrafter(Xing et al., 2023)를 비디오 확산 백본으로 사용한다.

스파스 입력 이미지 $\{I_1, I_2, \ldots, I_k\}$로부터 DUSt3R를 이용해 포인트 클라우드를 구축한다:

$$\mathcal{P} = \text{DUSt3R}(I_1, I_2, \ldots, I_k)$$

각 픽셀 $(u, v)$에 대해 3D 포인트맵 $X \in \mathbb{R}^{H \times W \times 3}$이 출력되며, 이를 통합하여 전역(global) 포인트 클라우드 $\mathcal{P}$를 구성한다.

---

### Stage 2: 비디오 확산 모델 조건화

포인트 클라우드를 풍부한 컨텍스트 표현 공간으로 인코딩하여 cross-attention 레이어에서 3D 조건으로 활용하며, 비디오 확산 모델이 세부 사항을 보존하면서 3D 일관성 있는 프레임을 합성하도록 유도한다.

3D 구조 조건 $\mathbf{c}_{3D}$는 포인트 클라우드 $\mathcal{P}$로부터 인코더 $\mathcal{E}$를 통해 계산된다:

$$\mathbf{c}_{3D} = \mathcal{E}(\mathcal{P}) \in \mathbb{R}^{T \times H \times W \times C}$$

Video Diffusion (DDPM 기반) 학습 목적함수:

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{x_0, \epsilon, t}\left[\left\|\epsilon - \epsilon_\theta(x_t, t, \mathbf{c}_{3D})\right\|^2\right]$$

여기서:
- $x_0$: 원본 비디오 프레임
- $\epsilon \sim \mathcal{N}(0, I)$: 노이즈
- $t$: 확산 timestep
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$: 노이즈가 추가된 프레임
- $\epsilon_\theta$: 3D 조건 $\mathbf{c}_{3D}$를 cross-attention을 통해 조건으로 받는 노이즈 예측 네트워크

조건의 안내를 통해 비디오 확산 모델은 세부 사항을 보존하면서 높은 수준의 3D 일관성을 보이는 비디오 프레임을 합성하여, 다양한 시점에서 장면의 일관성을 보장한다.

---

### Stage 3: Confidence-Aware 3D Gaussian Splatting 최적화

최종적으로 생성된 비디오로부터 confidence-aware 3D Gaussian Splatting 최적화 스킴을 통해 3D 장면을 복원한다.

3D Gaussian Splatting에서 각 Gaussian은 다음 파라미터로 정의된다:

$$\mathcal{G} = \{\mu_i, \Sigma_i, c_i, \alpha_i\}_{i=1}^{N}$$

- $\mu_i \in \mathbb{R}^3$: 위치(center)
- $\Sigma_i \in \mathbb{R}^{3 \times 3}$: 공분산 행렬(covariance)
- $c_i$: 색상(spherical harmonics)
- $\alpha_i$: 불투명도(opacity)

렌더링 수식 (alpha-compositing):

$$\hat{I}(u,v) = \sum_{i \in \mathcal{N}} c_i \cdot \alpha_i \cdot \prod_{j < i}(1 - \alpha_j)$$

Confidence-aware 최적화에서는 비디오 생성 프레임의 불확실성(uncertainty)을 반영한 신뢰도 맵 $w_i$를 도입하여 손실 함수를 가중:

$$\mathcal{L}_{\text{total}} = \lambda_1 \sum_{i} w_i \cdot \mathcal{L}_{\text{rgb}}(\hat{I}_i, I_i^{\text{gen}}) + \lambda_2 \mathcal{L}_{\text{LPIPS}}(\hat{I}, I^{\text{gen}})$$

Ablation study에서는 3D 구조 가이던스, confidence-aware 최적화, LPIPS 손실 등 설계 선택들의 효과를 검증했다.

---

## 4. 모델 구조 상세

```
[입력: 2~3장의 Sparse Views]
          │
          ▼
┌─────────────────────────┐
│   Stage 1: DUSt3R        │  ← Pose-free stereo reconstruction
│   Global Point Cloud 생성 │
└───────────┬─────────────┘
            │  𝒫 (Point Cloud)
            ▼
┌─────────────────────────┐
│  Stage 2: 3D Context    │  ← 포인트 클라우드 → Context 공간 인코딩
│  Representation 인코딩   │
└───────────┬─────────────┘
            │  c_3D (3D structure condition)
            ▼
┌─────────────────────────┐
│  DynamiCrafter          │  ← I2V Video Diffusion Model
│  (Video Diffusion, 512²)│  ← cross-attention으로 c_3D 주입
│  3D-consistent 프레임 생성│
└───────────┬─────────────┘
            │  생성된 비디오 프레임
            ▼
┌─────────────────────────┐
│  Stage 3: Confidence-   │  ← Gaussian Splatting
│  Aware 3DGS 최적화       │  ← LPIPS + weighted RGB loss
└───────────┬─────────────┘
            │
            ▼
     [최종 3D 장면 재구성]
```

---

## 5. 성능 향상 및 한계

### 5.1 성능 향상

다양한 실세계 데이터셋에 대한 광범위한 실험을 통해 품질 및 일반화 측면에서 최신 방법들보다 ReconX의 우월성이 검증되었다.

실험은 아래 데이터셋에서 진행되었다:
- **RealEstate10K** (in-distribution 학습/테스트)
- **NeRF-LLFF** (cross-dataset, zero-shot)
- **DTU** (cross-dataset, zero-shot)
- **Tank-and-Temples, DL3DV, Mip-NeRF 360** (추가 평가)

공정한 비교를 위해 모델을 RealEstate10K에서만 훈련하고 NeRF-LLFF 및 DTU 데이터셋에서 직접 테스트했다. 결과적으로 경쟁 베이스라인 방법인 MVSplat과 pixelSplat은 다른 카메라 분포와 이미지 외관을 가진 out-of-distribution(OOD) 데이터셋에서 급격한 성능 저하를 보였다.

성능 지표(PSNR, SSIM, LPIPS)에서 기존 방법 대비 개선:

| 비교 방법 | 특징 | ReconX 대비 |
|---|---|---|
| **pixelSplat** (Charatan, CVPR 2024) | Epipolar transformer 기반 | OOD에서 급격히 성능 저하 |
| **MVSplat** (Chen, ECCV 2024) | Cost volume 기반 | OOD에서 급격히 성능 저하 |
| **NeRF 기반 방법** (per-scene) | 장면별 최적화 필요 | 일반화 불가 |

### 5.2 한계

- **추론 속도**: ViewCrafter(Yu et al., 2024)와 같은 확산 기반 모델들은 많은 샘플링 단계로 인해 수 초에서 수 분에 이르는 금지적(prohibitive) 지연(latency) 문제가 있어 실시간 응용에 부적합하다. ReconX 역시 비디오 확산 모델을 사용하므로 동일한 속도 제약을 가진다.
- **비디오 프레임 품질 의존성**: 비디오 확산 모델이 생성한 프레임의 품질에 최종 3D 재구성이 의존하므로, 생성 품질이 낮은 경우 재구성 품질도 저하된다.
- **복잡한 파이프라인**: DUSt3R + DynamiCrafter + 3DGS의 세 단계 파이프라인으로 인해 end-to-end 학습이 어렵다.
- **해상도 제약**: 비디오 확산 백본으로 사용하는 DynamiCrafter는 $512 \times 512$ 해상도에서 동작하며, image cross-attention 레이어를 2000 스텝 미세 조정한다.

---

## 6. 모델의 일반화 성능 향상 가능성 (핵심 분석)

### 6.1 일반화 우월성의 근거

비디오 확산 모델의 3D 구조 조건을 통해 강력한 생성력을 발휘하는 ReconX는 본질적으로 out-of-distribution 새로운 장면으로의 일반화에 우수한 성능을 보인다.

이러한 일반화 성능의 근거는 세 가지로 분석된다:

**① 대규모 비디오 데이터로 학습된 생성 Prior 활용**

기존 feed-forward 방법(pixelSplat, MVSplat)은 3D 데이터 기반 학습에 의존하지만, ReconX는 인터넷 수준의 방대한 비디오 데이터로 학습된 비디오 확산 모델의 prior를 활용한다. 이는 훈련 분포 외 장면에서도 합리적인 시각적 콘텐츠를 생성할 수 있게 한다.

**② 포인트 클라우드 기반 구조적 조건화**

ReconX는 대규모 사전 학습된 비디오 확산 모델을 활용하여 스파스 뷰 재구성을 위한 추가 관측치를 생성한다. 글로벌 포인트 클라우드를 구축하고 이를 3D 구조 조건으로 인코딩하여 비디오 확산 모델이 3D 일관성 있는 프레임을 합성하도록 유도한다.

**③ Cross-dataset 제로샷 일반화 검증**

모델은 소스 데이터셋 RealEstate10K에서 훈련하고, 별도의 파인튜닝 없이 NeRF-LLFF와 DTU 데이터셋의 미관측 장면에서 직접 테스트된다.

### 6.2 일반화 한계 및 개선 방향

- **도메인 갭**: 비디오 확산 모델이 실내 장면에 편향될 경우 실외·산업 장면 일반화에 한계 존재
- **카메라 분포 변화**: DUSt3R 기반 포즈 추정은 극단적 뷰 변화 시 포인트 클라우드 품질 저하 가능
- **개선 가능성**: 더 큰 비디오 확산 모델(예: Sora류 모델) 또는 3D-native generative prior 도입 시 일반화 성능 추가 향상 기대

---

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 연구 계보 분류

```
Sparse-view 3D 재구성 연구 흐름

NeRF 기반
├── pixelNeRF (Yu et al., CVPR 2021)    ─ 단일 이미지 기반 NeRF
├── MVSNeRF (Chen et al., ICCV 2021)    ─ 다중 뷰 스테레오 + NeRF
├── SparseNeRF (Wang et al., 2023)      ─ 깊이 랭킹 증류
└── ReconFusion (Wu et al., 2023)       ─ 확산 prior + NeRF

Feed-forward Gaussian Splatting 기반
├── pixelSplat (Charatan, CVPR 2024)    ─ Epipolar transformer
├── MVSplat (Chen, ECCV 2024)           ─ Cost volume 기반
├── TranSplat (2024)                    ─ Transformer 기반
├── InstantSplat (Fan, 2024)            ─ DUSt3R + 빠른 최적화
└── FreeSplat (2024)                    ─ Pose-free

Generative Prior 통합 기반
├── CAT3D (Gao, 2024)                   ─ 멀티뷰 확산 + 3D
├── ViewCrafter (Yu, 2024)              ─ 비디오 확산 기반 NVS
└── ReconX (Liu, 2024)  ← [본 논문]    ─ 비디오 확산 + 3DGS

포즈-프리 다중 뷰 재구성
├── DUSt3R (Wang, CVPR 2024)            ─ 비제약 스테레오 재구성
├── MASt3R (Leroy, 2024)                ─ DUSt3R 개선판
└── MV-DUSt3R+ (2024)                  ─ 단일 패스 다중 뷰 재구성
```

### 7.2 방법별 상세 비교

| 방법 | 연도 | 표현 | 일반화 | 속도 | 포즈 필요 | 특징 |
|---|---|---|---|---|---|---|
| **pixelNeRF** | CVPR 2021 | NeRF | △ | 느림 | O | 최초 일반화 NeRF |
| **MVSNeRF** | ICCV 2021 | NeRF | △ | 느림 | O | MVS + NeRF 결합 |
| **pixelSplat** | CVPR 2024 | 3DGS | △ | 빠름 | O | Epipolar 어텐션 |
| **MVSplat** | ECCV 2024 | 3DGS | △ | 매우 빠름 | O | Cost volume 매칭 |
| **DUSt3R** | CVPR 2024 | 포인트맵 | ○ | 빠름 | **X** | 포즈-프리 |
| **ReconX** | 2024 | 3DGS | **◎** | 느림 | **X** | 비디오 확산 prior |
| **MV-DUSt3R+** | 2024 | 3DGS | ○ | **매우 빠름** | **X** | 단일 패스 |
| **FreeSplat** | 2024 | 3DGS | ○ | 빠름 | **X** | Pose-free |

MVSplat은 pixelSplat 대비 10배 적은 파라미터와 2배 이상 빠른 추론 속도로 외관과 기하학 품질 모두에서 우수한 성능을 보인다.

pixelSplat과 MVSplat은 대응 뷰의 깊이 맵을 3D Gaussian의 중심으로 투영하는 방식을 사용하지만, 이 방법들의 성능은 깊이 기반 픽셀 수준 매칭의 정확도에 의존하며, 이를 실현하기 어려워 재구성 정밀도를 제한한다.

최근 연구들은 신경 암묵 모델(NeRF), 명시적 포인트 클라우드 방법(3DGS), 확산 및 비전 기반 모델의 prior를 활용하는 하이브리드 프레임워크를 포괄하며, 기하학적 정규화, 명시적 형상 모델링, 생성적 추론이 스파스 뷰 환경의 아티팩트를 완화하는 데 어떻게 활용되는지를 분석하고 있다.

---

## 8. 앞으로의 연구에 미치는 영향과 고려할 점

### 8.1 연구에 미치는 영향

**① 패러다임 전환의 선도**

ReconX는 3D 재구성 문제를 "최적화 문제"가 아닌 "생성 문제"로 재정의함으로써, 기존 feed-forward 방법의 한계를 뛰어넘는 새로운 방향을 제시했다. 이 패러다임은 후속 연구인 MVSplat360, TranSplat 등에서 이미 참조·인용되고 있다.

**② 비디오 확산 모델과 3D 재구성의 연결**

MVSplat360은 geometry-aware 3D 재구성과 temporal 일관성 있는 비디오 생성을 효과적으로 결합한다. feed-forward 3DGS 모델이 특징을 사전 학습된 Stable Video Diffusion(SVD) 모델의 잠재 공간으로 직접 렌더링하며, 이 특징들이 포즈 및 시각적 단서로 작동한다.

**③ 일반화 가능 3D 재구성 연구 촉진**

지속적인 도전 과제인 도메인 일반화 및 포즈-프리 재구성이 강조되고 있으며, 3D-native 생성 prior 개발과 실시간·비제약 스파스 뷰 재구성 달성을 위한 미래 방향이 제시되고 있다.

### 8.2 앞으로 연구 시 고려할 점

#### 🔬 기술적 고려사항

1. **추론 속도 개선**: ReconX는 비디오 확산의 다단계 샘플링으로 인해 느리다. DDIM/consistency model 등 빠른 샘플링 기법 적용이 필요하다.

2. **End-to-End 학습**: 현재의 DUSt3R → DynamiCrafter → 3DGS 세 단계 파이프라인을 통합적으로 학습하는 구조 탐색이 중요하다.

3. **비디오 확산 모델의 발전 활용**: 더욱 강력한 비디오 생성 모델(예: Wan2.1, CogVideoX)이 등장하고 있으며, 이를 백본으로 교체 시 품질/일반화 성능 추가 향상이 기대된다.

4. **포즈-프리 파이프라인 완전화**: 이러한 방법들은 모두 알려진 카메라 내재 및 외재 파라미터가 필요한데, 이는 실세계에서 실용적이지 않다. DUSt3R의 이점을 활용한 InstantSplat(Fan et al., 2024)은 포즈 없는 스파스 뷰 입력에서 정확한 카메라 파라미터와 초기 3D 표현을 획득할 수 있다.

5. **동적 장면(Dynamic Scene) 확장**: 현재 ReconX는 정적 장면만 대상으로 한다. 동적 장면(움직이는 물체, 사람 등) 재구성으로의 확장이 필요하다.

#### 📊 평가 및 벤치마크 관련 고려사항

6. **다양한 도메인 평가**: 실내/실외, 의료, 위성 이미지 등 다양한 도메인에서의 일반화 성능 체계적 평가가 필요하다.

7. **계산 효율성 지표 강화**: PSNR/SSIM뿐 아니라 추론 시간, 메모리 사용량, FLOPs 등 효율성 지표를 포함한 종합 평가가 중요하다.

#### 🌐 응용 관련 고려사항

8. **로보틱스·자율주행 통합**: 스파스 뷰 3D 재구성은 로보틱스, AR/VR, 자율 시스템 등 밀집 이미지 획득이 비현실적인 응용에 필수적이다. 이러한 환경에서 최소한의 이미지 겹침은 신뢰할 수 있는 대응점 매칭을 방해하여 전통적인 SfM/MVS 방법이 실패한다.

9. **3D 콘텐츠 생성 파이프라인 통합**: 메타버스, 게임, 영화 산업의 3D 자산 생성 파이프라인과의 통합 가능성 탐색이 중요하다.

---

## 9. 참고 자료 목록

| # | 참고 자료 |
|---|---|
| 1 | **ReconX 논문 (주요)**: Fangfu Liu et al., "ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Model," arXiv:2408.16767, 2024. https://arxiv.org/abs/2408.16767 |
| 2 | **ReconX 프로젝트 페이지**: https://liuff19.github.io/ReconX |
| 3 | **ReconX GitHub**: https://github.com/liuff19/ReconX |
| 4 | **ReconX OpenReview (ICLR 2025)**: https://openreview.net/forum?id=Z30Mdbv5jO |
| 5 | **ReconX IEEE Xplore**: https://ieeexplore.ieee.org/document/11415357 |
| 6 | **MVSplat (ECCV 2024)**: Yuedong Chen et al., "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images," ECCV 2024. https://arxiv.org/abs/2403.14627 |
| 7 | **pixelSplat (CVPR 2024)**: David Charatan et al., "pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction," CVPR 2024. |
| 8 | **DUSt3R (CVPR 2024)**: Shuzhe Wang et al., "DUSt3R: Geometric 3D Vision Made Easy," CVPR 2024. |
| 9 | **FreeSplat**: "FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction," arXiv:2412.09573, 2024. https://arxiv.org/html/2412.09573v1 |
| 10 | **MV-DUSt3R+**: "MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds," arXiv:2412.06974, 2024. |
| 11 | **TranSplat**: "TranSplat: Generalizable 3D Gaussian Splatting from Sparse Multi-View Images with Transformers," arXiv:2408.13770, 2024. |
| 12 | **MVSplat360**: "MVSplat360: Feed-Forward 360 Scene Synthesis from Sparse Views," arXiv:2411.04924, 2024. |
| 13 | **Sparse-View 3D 재구성 서베이 (2025)**: "Sparse-View 3D Reconstruction: Recent Advances and Open Challenges," arXiv:2507.16406, 2025. https://arxiv.org/html/2507.16406v1 |
| 14 | **3DGS 스파스 뷰 리뷰 (Springer 2025)**: "A review on 3D Gaussian splatting for sparse view reconstruction," Artificial Intelligence Review, Springer, 2025. https://link.springer.com/article/10.1007/s10462-025-11171-4 |
| 15 | **HuggingFace 논문 페이지**: https://huggingface.co/papers/2408.16767 |

---

> ⚠️ **정확도 관련 안내**: 논문의 상세 수식(특히 confidence weight $w_i$의 구체적 계산식, 학습 hyperparameter 등)은 공개된 HTML/PDF 전문을 직접 확인하지 못한 부분이 있어, 핵심 원리에 기반한 표준적 수식으로 표현했습니다. 정확한 수식은 [arXiv 전문](https://arxiv.org/abs/2408.16767) 또는 [OpenReview PDF](https://openreview.net/forum?id=Z30Mdbv5jO)를 직접 참조하시길 권장합니다.
