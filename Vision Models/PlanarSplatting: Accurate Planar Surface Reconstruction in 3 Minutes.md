
# PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes

> **논문 정보**
> - **제목**: PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes
> - **저자**: Bin Tan, Rui Yu, Yujun Shen, Nan Xue
> - **발표**: CVPR 2025 Highlight
> - **arXiv**: [2412.03451](https://arxiv.org/abs/2412.03451) (2024년 12월 4일)
> - **GitHub**: [ant-research/PlanarSplatting](https://github.com/ant-research/PlanarSplatting)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 멀티뷰 실내 이미지를 위한 초고속·고정밀 표면 재구성 기법인 PlanarSplatting을 제안하며, 3D 평면을 실내 장면의 주된 표현 객체로 삼아 3D 평면들을 2.5D 깊이(depth) 및 법선(normal) 맵으로 splatting하여 실내 장면의 표면을 근사하는 명시적 최적화 프레임워크를 개발한다.

PlanarSplatting은 3D 평면 기본 요소(primitives)에 직접 작동하기 때문에 2D/3D 평면 검출, 평면 매칭 및 추적에 대한 의존성을 제거하며, 평면 기반 표현의 장점과 CUDA 기반 구현을 결합하여 3분 이내에 실내 장면을 재구성하면서도 현저히 더 나은 기하학적 정밀도를 달성한다.

### 주요 기여 요약

| 기여 항목 | 내용 |
|---|---|
| **표현 방식** | 3D 직사각형 평면 primitive의 명시적 최적화 |
| **렌더링 방식** | 미분 가능한 평면 splatting → 2.5D 깊이/법선 맵 |
| **속도** | 단일 GPU, 3분 이내 재구성 완료 |
| **독립성** | 평면 검출·매칭·추적 사전 처리 불필요 |
| **평가 규모** | ScanNet·ScanNet++ 수백 장면에 걸친 최대 규모 정량 평가 |
| **응용 확장** | Gaussian Splatting과 통합하여 Novel View Synthesis 품질 향상 |

---

## 2. 해결하고자 하는 문제, 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

평면적 3D 재구성은 오랫동안 모델 피팅 문제로 연구되어 왔으나, 이미지 기반 방법들은 최근에 단일 뷰 및 멀티뷰 3D 재구성을 통해 점차 단순화되고 있다. 그러나 기존 방법들은 여전히 각 입력 이미지에 대해 3D 평면을 검출하고, 시점 간에 평면을 매칭·추적하며, 최종적으로 3D 평면을 재구성·병합하는 2D/3D 이미지 레벨 특징에 의존했다.

특히 학습 기반 방법들은 2D/3D 평면 어노테이션을 지도 학습 신호로 요구하여 대규모 평면 어노테이션 획득의 어려움으로 인한 성능 병목이 존재했다. 반면 PlanarSplatting은 추가적인 평면 검출이나 매칭 없이 멀티뷰 이미지로부터 직접 3D 평면 기본 요소들을 최적화하여 정확하고 완전한 실내 평면 표면을 재구성한다.

---

### 2.2 제안하는 방법 및 핵심 수식

#### (A) 학습 가능한 평면 Primitive 표현

PlanarSplatting의 핵심은 3D 평면을 직사각형 고체 primitive로 표현하는 것이며, 각 primitive는 다음과 같은 학습 가능한 파라미터들로 구성된다: **평면 중심(Plane Center, $p_\pi$)**: 3D 공간에서 평면 중심 위치를 나타내는 $\mathbb{R}^3$ 벡터, **평면 회전(Plane Rotation, $q_\pi$)**: 법선 방향으로의 회전을 나타내는 $\mathbb{R}^4$ 쿼터니언.

이 외에도 각 primitive는 반경(radii)과 같은 파라미터를 포함하며, 이를 종합하면 하나의 평면 primitive $\pi$는 다음과 같이 정의된다:

$$\pi = \{p_\pi \in \mathbb{R}^3, \; q_\pi \in \mathbb{R}^4, \; r_\pi \in \mathbb{R}^2\}$$

여기서 $p_\pi$는 평면 중심, $q_\pi$는 회전 쿼터니언, $r_\pi$는 평면의 가로·세로 반경이다.

이러한 학습 가능한 파라미터를 통해 3D 평면 primitive는 최적화 과정에서 잠재적인 장면 표면에 맞게 이동하고, 표면 형상에 맞게 변형될 수 있다.

#### (B) 장면 초기화

PlanarSplatting은 최적화 초기에 최신 기초 모델로부터 단안 깊이를 사용하여 3D 평면 primitive를 빠르게 초기화한다. 구체적으로, Metric3Dv2로부터 추정된 깊이를 이용해 매우 거친 장면 형상을 얻고, 이 거친 메시 위에서 2,000개의 점을 무작위로 샘플링하여 3D 평면 primitive의 평면 중심을 설정한다. 거친 메시의 법선 방향을 이용해 평면 회전도 초기화한다.

#### (C) 미분 가능한 평면 Splatting 렌더링

PlanarSplatting은 3D 공간에서 직사각형 평면 primitive들을 2.5D 깊이 및 법선 맵으로 미분 가능하게 splatting하여 명시적으로 최적화한다.

각 카메라 뷰에서 렌더링된 깊이 $D$는 일종의 가중 평균으로 표현될 수 있다. primitive $\pi_i$가 픽셀 $u$에 기여하는 가중치 $w_i(u)$와 해당 깊이 $d_i(u)$를 통해:

$$\hat{D}(u) = \frac{\sum_i w_i(u) \cdot d_i(u)}{\sum_i w_i(u)}$$

마찬가지로 법선 맵 $\hat{N}(u)$은:

$$\hat{N}(u) = \frac{\sum_i w_i(u) \cdot n_i}{\|\sum_i w_i(u) \cdot n_i\|}$$

여기서 $n_i$는 primitive $\pi_i$의 법선 벡터이다.

이러한 primitive 기반 방법의 핵심은 gradient descent로 primitive의 속성을 최적화하는 미분 가능한 렌더링 프로세스를 설계하는 것이며, 전형적인 패러다임은 primitive에 정의된 방사 기저 함수(예: Gaussian 함수)로 실현되는 splatting 기법으로 primitive로부터 이미지를 렌더링하는 것이다.

#### (D) 최적화 손실 함수

PlanarSplatting은 정교하게 설계된 평면 splatting 함수 덕분에 현대 기초 모델로부터 얻은 단안 기하 단서를 활용하며, 미분 가능한 평면 primitive 렌더링을 통해 평면 어노테이션 없이 현대 기초 모델의 단안 깊이/법선 단서를 직접 활용한다.

전체 최적화 손실은 깊이 손실과 법선 손실의 합으로 구성된다:

$$\mathcal{L} = \mathcal{L}_{\text{depth}} + \lambda \mathcal{L}_{\text{normal}}$$

깊이 손실:

$$\mathcal{L}_{\text{depth}} = \sum_{u} \left\| \hat{D}(u) - D^*(u) \right\|_1$$

법선 손실:

$$\mathcal{L}_{\text{normal}} = \sum_{u} \left( 1 - \hat{N}(u) \cdot N^*(u) \right)$$

여기서 $D^\*(u)$와 $N^\*(u)$는 각각 Metric3Dv2 등의 기초 모델에서 제공된 단안 깊이·법선 의사 레이블(pseudo-label)이며, $\lambda$는 두 손실 항 사이의 균형 하이퍼파라미터이다.

#### (E) 평면 분할 (Plane Splitting)

최적화 중에 장면 기하를 더 잘 적합시키기 위해 Plane Splitting을 도입하며, 이 연산은 3D 평면 primitive의 X축 및 Y축 방향을 따라 수행된다.

#### (F) 평면 병합 (Plane Merging) 및 최종 재구성

실내 장면을 멀티뷰 입력 이미지로부터 수집한 고체 3D 평면 primitive 집합으로 근사하여, 일관된 3D 평면을 갖도록 직접 최적화하되 평면 기본 요소(예: 평면 마스크)의 비최적 사전 계산을 제거한다.

---

### 2.3 모델 구조

```
멀티뷰 이미지 입력 (posed)
         │
         ▼
[기초 모델 단안 추정 – Metric3Dv2]
 → 깊이 맵, 법선 맵 → 초기 coarse 메시 생성
         │
         ▼
[3D 평면 Primitive 초기화]
 → K=2,000개의 직사각형 평면: {p_π, q_π, r_π}
         │
         ▼
[CUDA 기반 미분 가능 평면 Splatting 렌더링]
 → 각 카메라 뷰로 깊이 맵 & 법선 맵 렌더링
         │
         ▼
[손실 계산 & Gradient Descent 최적화]
 → L_depth + λ·L_normal
         │
         ▼
[Plane Splitting / 적응적 primitive 분할]
         │
         ▼
[유사 평면 병합 (Plane Merging)]
 → 어노테이션 없이 유사한 3D 평면 기본 요소 병합
         │
         ▼
고품질 평면적 실내 표면 재구성 결과
```

문제를 평면 splatting을 이용한 미분 가능한 렌더링으로 공식화하여 정확한 형상 재구성과 컴팩트한 구조적 장면 모델링 양쪽에서 3D 평면 표현의 강력한 능력을 보여주며, 효율적인 CUDA 구현은 단일 GPU를 사용하여 몇 시간 내에 100개 이상의 장면에 걸친 포괄적인 평가를 가능하게 한다.

---

### 2.4 성능 향상

초고속 재구성 속도 덕분에 ScanNet 및 ScanNet++ 데이터셋에서 수백 개의 장면에 걸친 최대 규모의 정량적 평가에서 해당 방법의 장점이 명확히 입증되었으며, 저자들은 정확하고 초고속인 평면 표면 재구성 방법이 미래에 표면 재구성을 위한 구조화된 데이터 큐레이션에 적용될 것이라 믿는다.

또한 Gaussian Splatting과의 통합에서도 성능 향상이 확인되었다:

PlanarSplatting은 3D 평면 primitive에서 직접 설계되고 CUDA로 효율적으로 구현되어 고품질 실내 novel view synthesis를 위한 최신 Gaussian Splatting 방법과 원활하게 통합될 수 있으며, GS 기반 방법들이 densification 없이 잘 초기화되고 최적화되어 더 나은 렌더링 결과와 훨씬 적은 학습 시간을 달성한다.

학습률 설정:
학습 가능한 평면 중심, 평면 반경, 평면 회전의 학습률은 모두 0.001로 고정된다.

---

### 2.5 한계

PlanarSplatting은 정확한 실내 평면 표면을 재구성할 수 있지만, 곡면과 같은 복잡한 형태에는 적합하지 않으며, 이 도전적인 문제는 보다 유연한 기하학적 모델링을 위한 향후 연구 과제로 남겨두었다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Pose-Free 입력 지원

PlanarSplatting은 VGG-T에 기반한 포즈 없는(pose-free) 멀티뷰 입력을 지원한다. 이는 카메라 포즈 정보가 없는 환경에서도 재구성이 가능함을 의미하며, 실제 배포 환경에서의 일반화 가능성을 크게 높인다.

### 3.2 어노테이션 없는 최적화

PlanarSplatting은 추가적인 평면 검출이나 매칭 없이 포즈가 있는 멀티뷰 이미지로부터 직접 3D 평면 primitive 집합을 최적화하며, 미분 가능한 평면 primitive 렌더링 덕분에 평면 어노테이션 없이 현대 기초 모델의 단안 깊이/법선 단서를 직접 활용할 수 있다.

이는 다음과 같은 일반화 이점을 제공한다:
- 대규모 어노테이션 없이 새로운 도메인에 적용 가능
- 다양한 실내 환경에 확장 적용 용이

### 3.3 대규모 기초 모델 활용을 통한 일반화

잘 설계된 평면 splatting 함수 덕분에, PlanarSplatting은 정확한 평면 최적화를 위해 현대 기초 모델들로부터의 단안 기하 단서를 효과적으로 활용한다.

특히 초기화에 사용되는 Metric3Dv2는 다음과 같은 강력한 일반화 능력을 가진다:
이 모델은 다양한 카메라 모델로부터 수집된 1,600만 장 이상의 이미지로 안정적으로 학습되어, 학습에 보지 않은 카메라 설정의 인터넷 이미지에 제로샷 일반화가 가능하며, 임의로 수집된 인터넷 이미지에서 정확한 metric 3D 구조 복원을 가능하게 한다.

### 3.4 PLANA3R로의 발전: 직접적 일반화 확장

후속 연구 PLANA3R(NeurIPS 2025)은 PlanarSplatting에서 도입된 평면 primitive 위에 구축되어 CUDA 기반 미분 가능 렌더러를 지도 학습 신호로 활용하며, 희소 평면 primitive를 직접 출력하면서도 여전히 조밀한 깊이/법선 렌더링을 지원한다.

이는 PlanarSplatting의 일반화 가능성이 피드포워드 학습 기반 방법으로 확장되고 있음을 보여주며, 미지의 장면에 대한 제로샷 재구성으로의 진화 방향을 제시한다.

### 3.5 실내 한계를 넘기 위한 방향

일반화 능력의 주요 제한 요소로서, 대규모 고품질 3D 평면 어노테이션의 부족이 지적되며, 이는 지도 학습을 제한하고 대규모 데이터로 학습된 모델의 일반화 능력을 한계 짓는다. PlanarSplatting의 어노테이션 불필요 설계는 이 문제를 우회하는 효과적인 전략이다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4.1 관련 방법 비교표

| 방법 | 유형 | 속도 | 평면 어노테이션 필요 | 사전 검출 필요 | 주요 특징 |
|---|---|---|---|---|---|
| **PlanarSplatting (2024)** | 최적화 기반 | 3분 | ❌ | ❌ | 직사각형 평면 primitive |
| **PGSR (2024)** | 3DGS 기반 | 빠름 | ❌ | ❌ | Unbiased depth rendering |
| **PlanarRecon (2022)** | 학습 기반 | 실시간 | ✅ | ✅ | 3D 볼륨 기반 end-to-end |
| **AirPlanes (2023)** | 학습 기반 | 느림 | ✅ | ✅ | Dense mesh + plane embedding |
| **PlaneFormers (2022)** | 트랜스포머 | 빠름 | ✅ | ✅ | Transformer + 3D-aware token |
| **PLANA3R (2025)** | 피드포워드 | 빠름 | ❌ | ❌ | Zero-shot, pose-free |

### 4.2 PGSR과의 비교

3DGS는 고품질 렌더링과 초고속 학습·렌더링으로 광범위한 주목을 받았으나, 비구조적이고 불규칙한 Gaussian 점 구름의 특성으로 인해 이미지 재구성 손실만으로는 기하학적 재구성 정밀도와 멀티뷰 일관성을 보장하기 어렵고 최근 많은 3DGS 기반 표면 재구성 연구들이 등장했음에도 메시 품질이 전반적으로 만족스럽지 않은 문제가 있었다. 이를 해결하기 위해 PGSR은 고충실도 표면 재구성과 고품질 렌더링을 동시에 달성하는 빠른 평면 기반 Gaussian splatting 재구성 표현을 제안한다.

PGSR은 구체적으로 카메라 원점에서 Gaussian 평면까지의 거리와 해당 법선 맵을 점 구름의 Gaussian 분포에 기반하여 직접 렌더링하는 편향 없는 깊이 렌더링 방법을 도입하고, 전역 기하 정밀도 보존을 위한 단일뷰 기하, 멀티뷰 측광, 기하 정규화를 도입한다.

**PlanarSplatting vs PGSR의 핵심 차이**:
- PGSR은 3DGS 위에 평면 정규화를 추가한 방식인 반면, PlanarSplatting은 처음부터 평면 primitive를 명시적으로 최적화하는 방식
- PlanarSplatting은 컴팩트한 구조적 표현(평면 병합 포함)에 집중하는 반면, PGSR은 렌더링 품질과 메시 품질 모두를 목표

### 4.3 Planar Gaussian Splatting(PGS, 2024)과의 비교

Planar Gaussian Splatting(PGS)은 여러 RGB 이미지로부터 장면의 3D 기하와 3D 평면을 직접 학습하는 새로운 신경 렌더링 접근법이다. PGS는 Gaussian primitive를 활용하여 계층적 Gaussian 혼합(hierarchical Gaussian mixture) 방식을 채용한 반면, PlanarSplatting은 독립적인 직사각형 평면 primitive를 명시적으로 최적화한다는 점에서 구별된다.

### 4.4 PlanarGS (2025)와의 비교

PlanarGS는 실내 장면 재구성에 맞춰진 3DGS 기반 프레임워크를 도입하며, 사전 학습된 비전-언어 세그멘테이션 모델을 활용한 언어 프롬프트 평면 사전(Language-Prompted Planar Priors) 파이프라인을 설계하고, 평면 일관성을 강화하는 평면 사전 지도 항과 깊이·법선 단서로 Gaussian을 유도하는 기하 사전 지도 항 두 가지 항을 추가하여 3D Gaussian을 최적화한다.

---

## 5. 연구에 미치는 영향 및 향후 고려 사항

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 초고속 재구성 패러다임의 확립

효율적인 CUDA 구현으로 단일 GPU를 사용하여 몇 시간 내에 100개 이상의 장면에 걸친 포괄적인 평가가 가능해졌다. 이는 실제 산업 응용에서 실내 장면 3D 재구성의 실용적 배포 가능성을 크게 높인다.

#### (2) 어노테이션 불필요 최적화 방향의 제시

미분 가능한 평면 primitive 렌더링 덕분에 PlanarSplatting은 평면 어노테이션 없이 현대 기초 모델의 단안 깊이/법선 단서를 직접 활용할 수 있다. 이는 대규모 어노테이션 없이 고품질 재구성이 가능함을 보여주어, 향후 자기 지도 학습(self-supervised) 방식 3D 재구성 연구에 영향을 미칠 것이다.

#### (3) Gaussian Splatting 생태계와의 시너지

Gaussian Splatting 방법과의 통합을 통해 향상된 novel view synthesis(NVS) 품질을 달성함으로써, PlanarSplatting이 구조화된 데이터 큐레이션과 표면 모델링의 미래 응용을 위한 기초 도구로서의 가능성을 입증한다.

#### (4) 피드포워드 학습 방향으로의 진화

PLANA3R(NeurIPS 2025)은 PlanarSplatting에서 도입된 평면 primitive 위에 구축되어 CUDA 기반 미분 가능 렌더러를 지도 학습 신호로 활용하며, 희소 평면 primitive를 직접 출력하면서도 조밀한 깊이/법선 렌더링을 지원한다. 이는 PlanarSplatting이 차세대 피드포워드 제로샷 재구성 연구의 직접적 토대가 됨을 보여준다.

#### (5) 구조화된 데이터 큐레이션 기여

정확하고 초고속인 평면 표면 재구성 방법이 미래에 표면 재구성을 위한 구조화된 데이터 큐레이션에 적용될 것이다. 대규모 3D 실내 데이터셋 구축을 위한 자동화된 도구로 활용될 가능성이 높다.

---

### 5.2 향후 연구 시 고려할 점

#### ① 곡면·비평면 기하 처리
모델이 평면 표면 재구성에서 탁월하지만, 곡면과 같은 복잡한 비평면 기하 표현에 대한 한계를 저자들 스스로 인정하고 있어, 보다 유연한 모델링 증가를 목표로 하는 향후 연구 방향을 제시한다. 예를 들어, 평면 primitive와 곡선 primitive를 혼합하는 하이브리드 표현 연구가 필요하다.

#### ② 실외 장면으로의 확장
현재 PlanarSplatting은 실내 장면에 특화되어 있다. 도로면, 건물 외벽 등 실외 대규모 평면 구조물로의 확장 가능성을 탐색할 필요가 있다.

#### ③ 동적 장면 처리
현재 프레임워크는 정적 장면을 가정하므로, 움직이는 객체가 존재하는 동적 실내 환경에서의 강건성 연구가 필요하다.

#### ④ 포즈 없는 (Pose-Free) 입력의 강화
VGG-T에 기반한 포즈 없는 멀티뷰 입력을 지원한다. 그러나 포즈 추정 오차가 누적될 경우 재구성 정밀도가 저하될 수 있으므로, 포즈 추정과 평면 재구성을 공동 최적화하는 연구가 필요하다.

#### ⑤ 평면 어노테이션 부족 문제
대규모 고품질 3D 평면 어노테이션의 부족이 지도 학습을 제한하고 모델의 일반화 능력을 한계 짓는다. 합성 데이터 생성이나 준지도 학습 방식을 통한 데이터 증강 전략이 향후 주요 연구 주제가 될 것이다.

#### ⑥ 렌더링 품질(NVS)과 재구성 정밀도의 공동 최적화
PlanarSplatting과 Gaussian Splatting의 원활한 통합은 실내 novel view synthesis의 품질과 효율성을 모두 향상시키며 광범위한 잠재력을 보여준다. 재구성 정밀도와 렌더링 품질을 동시에 최적화하는 통합 프레임워크 연구가 유망하다.

---

## 참고 자료 및 출처

1. **[주 논문]** Tan, B., Yu, R., Shen, Y., Xue, N. "PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes." *CVPR 2025*, pp. 1190–1199. — [arXiv:2412.03451](https://arxiv.org/abs/2412.03451)

2. **[공식 프로젝트 페이지]** PlanarSplatting Project Page — https://icetttb.github.io/PlanarSplatting/

3. **[공식 GitHub]** ant-research/PlanarSplatting — https://github.com/ant-research/PlanarSplatting

4. **[CVPR 2025 Open Access]** https://openaccess.thecvf.com/content/CVPR2025/html/Tan_PlanarSplatting_Accurate_Planar_Surface_Reconstruction_in_3_Minutes_CVPR_2025_paper.html

5. **[비교 연구 1]** Chen, D. et al. "PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction." *IEEE TVCG 2024.* — [arXiv:2406.06521](https://arxiv.org/abs/2406.06521)

6. **[비교 연구 2]** Zanjani, F.G. et al. "Planar Gaussian Splatting (PGS)." *WACV 2025.* — [arXiv:2412.01931](https://arxiv.org/abs/2412.01931)

7. **[비교 연구 3]** "PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors." — [arXiv:2510.23930](https://arxiv.org/abs/2510.23930)

8. **[후속 연구]** "PLANA3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting." *NeurIPS 2025.* — GitHub: lck666666/plana3r

9. **[GSPlane 후속 연구]** "GSPlane: Concise and Accurate Planar Reconstruction via Structured Representation." — [arXiv:2510.17095](https://arxiv.org/abs/2510.17095)

10. **[기초 모델 참조]** Hu, M. et al. "Metric3Dv2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation." *IEEE TPAMI 2024.* — [arXiv:2404.15506](https://arxiv.org/abs/2404.15506)

11. **[리터러처 리뷰]** Moonlight Literature Review: PlanarSplatting — https://www.themoonlight.io/en/review/planarsplatting-accurate-planar-surface-reconstruction-in-3-minutes

> ⚠️ **정확도 주의사항**: 손실 함수의 구체적 수식 형태(가중치 $w_i(u)$의 정확한 정의, $\lambda$ 값 등)는 논문의 전체 본문을 직접 확인해야 합니다. 본 보고서의 수식 일부는 논문의 방법론적 설명을 토대로 합리적으로 재구성한 것이며, 논문 원본의 수식과 세부 표기가 다를 수 있습니다.
