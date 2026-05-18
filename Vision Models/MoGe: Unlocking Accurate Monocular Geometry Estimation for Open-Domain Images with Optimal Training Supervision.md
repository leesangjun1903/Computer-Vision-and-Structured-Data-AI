
# MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision

> **논문 정보:**
> - **저자:** Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, Jiaolong Yang
> - **소속:** USTC, Microsoft Research, Harvard, Tsinghua University
> - **발표:** CVPR 2025 Oral
> - **arXiv:** [2410.19115](https://arxiv.org/abs/2410.19115)
> - **코드:** [github.com/microsoft/MoGe](https://github.com/microsoft/MoGe)

---

## 1. 핵심 주장 및 주요 기여 요약

### 📌 핵심 주장

MoGe는 단안(monocular) 오픈 도메인 이미지에서 3D 기하학을 복원하는 강력한 모델로, 단일 이미지가 주어지면 **affine-invariant 표현**으로 캡처된 장면의 3D point map을 직접 예측한다. 이 새로운 표현 방식은 전역 스케일(scale)과 이동(shift)에 무관하며, 학습 시 모호한 지도(supervision)를 제거하고 효과적인 기하학 학습을 가능하게 한다.

### 🎯 주요 기여 (3가지)

1. **새로운 직접 MGE 방법 제안:** 오픈 도메인 이미지에 대한 affine-invariant point map을 사용하는 직접적인 단안 기하학 추정(MGE) 방법을 제안한다.
2. **새로운 전역 및 로컬 감독(supervision) 체계 구축:** 강건하고 정밀한 기하학 복원을 위한 새로운 전역·로컬 감독 기법들을 확립한다.

3. 제로샷(zero-shot) 평가에서 8개의 미학습 데이터셋에 대해 이전 최고 성능 대비 **35% 이상의 오류 감소**를 달성하였으며, 단안 깊이 추정(MDE)과 카메라 FoV 추정에서도 각각 **20~30%, 20% 이상**의 오류 감소를 보였다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

단안 기하학 추정(MGE)은 2D 이미지로부터 3D 구조를 복원하는 과정에서 고유한 모호성 때문에 오랫동안 컴퓨터 비전의 난제였다. 기존 방법들은 다수의 이미지나 카메라 파라미터에 대한 사전 지식을 필요로 했으며, 이는 많은 실제 시나리오에서 얻기 어렵다.

**구체적인 두 가지 문제:**

**① 초점-거리 모호성 (Focal-Distance Ambiguity)**

기존 MGE 방법들은 스케일 불명확성이 있는 깊이 맵을 먼저 추정한 뒤, 카메라 내부 파라미터와 결합해 역투영(unprojection)으로 3D 형상을 복원하는 방식을 따랐다. 대규모 데이터 학습으로 단안 깊이 추정은 크게 발전했지만, 단일 이미지에서 카메라 초점 거리(focal length) 같은 내부 파라미터를 추론하는 것은 기하학적 단서가 없을 때 매우 높은 모호성으로 인해 여전히 어렵다. 부정확한 카메라 파라미터는 정확한 깊이 맵과 결합되어도 심각한 기하학 왜곡을 초래한다.

**② 기존 정렬 방법의 부정확성**

학습 시 예측 결과와 정답(GT)을 맞추기 위해 전역 스케일링 및 이동 계수를 계산해야 하는데, 기존의 전역 정렬 계산 방법들은 이상치(outlier)에 민감하거나 근사값으로만 계산하여 만족스럽지 못한 지도(supervision)를 초래했다. 이를 해결하기 위해 affine-invariant pointmap 손실에서 스케일과 이동을 해결하는 **ROE(Robust, Optimal, Efficient) 전역 정렬 솔버**를 제안하며, 이는 학습 효율성과 최종 정확도를 크게 향상시킨다.

---

### 2.2 제안하는 방법 (수식 포함)

#### ① Affine-Invariant Point Map 표현

MoGe는 예측 point map $\hat{P} \in \mathbb{R}^{H \times W \times 3}$을 정답 기하학 $P$에 대해 미지의 전역 스케일과 이동을 가진 **affine-invariant** 표현으로 학습한다:

$$\hat{P} = s \cdot P + t$$

여기서 $s$는 전역 스케일, $t$는 3D 이동(shift) 벡터를 의미한다.

DUSt3R의 scale-invariant point map과 달리, MoGe는 전역 스케일과 3D 이동 모두를 미지수로 두는 affine-invariant point map을 예측한다. 이 변경은 네트워크 학습에 해로운 **focal-distance ambiguity를 제거**하는 데 중요한 역할을 한다.

#### ② ROE 전역 정렬 솔버 (Global Alignment Loss)

MoGe는 예측 point map과 정답 기하학 $P$ 사이에서, 유효 마스크 $M$과 역 깊이 가중치 $1/z_i$를 사용해 최적 스케일 $s^\*$와 이동 $t^*$를 결정하며, 이를 위해 효율적인 병렬 탐색 알고리즘 기반의 **ROE(Robust Optimal Efficient) 정렬 솔버**를 사용한다.

전역 정렬 손실의 일반 형식은:

$$\mathcal{L}_S = \frac{1}{|M|} \sum_{i \in M} \frac{1}{z_i} \cdot \left\| \hat{P}_i - (s^* P_i + t^*) \right\|$$

여기서:
- $M$: 정답 point map의 유효 마스크
- $z_i$: 픽셀 $i$의 정답 깊이 (역 깊이 가중치 적용)
- $s^\*, t^*$: ROE 솔버로 구한 최적 스케일 및 이동

#### ③ 멀티스케일 로컬 기하학 손실 (Multi-Scale Local Geometry Loss)

멀티스케일 로컬 기하학 손실에서, 단안 기하학 추정 시 서로 다른 객체 간의 상대적 거리는 모호하고 예측하기 어렵다. 이는 모든 객체를 포함한 전역 정렬이 적용될 때 정밀한 로컬 기하학 학습을 방해한다. 이를 향상시키기 위해 **로컬 구형 영역(local spherical regions)의 정확도를 측정하는 손실 함수**를 제안한다.

로컬 손실 $\mathcal{L}_N$은 다양한 스케일의 로컬 패치 $\mathcal{R}_k$에 대해 각각 정렬을 계산하는 방식으로:

$$\mathcal{L}_N = \sum_{k} \frac{1}{|\mathcal{R}_k|} \sum_{i \in \mathcal{R}_k} \left\| \hat{P}_i - (s_k^* P_i + t_k^*) \right\|$$

이와 함께 ROE 전역 정렬 솔버로 affine-invariant pointmap 손실의 스케일과 이동을 해결하며, 멀티스케일 로컬 기하학 손실은 3D 포인트 클라우드의 로컬 불일치에 페널티를 부과함으로써 정밀한 로컬 기하학 지도를 촉진한다.

#### ④ 법선 손실 및 마스크 손실

더 나은 표면 품질을 위해 예측된 포인트로부터 계산된 법선(normal)을 감독하는 **법선 손실(Normal Loss)**을 추가로 적용한다. 또한 야외 장면의 하늘 영역이나 객체 전용 이미지의 단순 배경처럼 무한 거리 영역(infinity region)에 대해서는 단일 채널 헤드를 추가하는 **마스크 손실(Mask Loss)**을 사용한다.

이를 통해 모델은 하늘 영역을 올바르게 예측할 수 있으며, 마스크 없이 큰 거리 레이블을 할당하면 전경 예측 정확도에 부정적인 영향을 미친다는 점을 검증했다.

#### ⑤ 카메라 FoV 추정

모델은 등방성 초점 거리와 중심 주점을 갖는 단순 핀홀 카메라 모델을 가정하며, 이미지 평면을 중심 기반으로 파라미터화한다. 각 픽셀 $(u_i, v_i)$에 대응하는 예측 3D 포인트 $\mathbf{p}_i = (x_i, y_i, z_i)$를 기반으로 투영 오차를 최소화해 초점 거리와 이동을 추정한다.

---

### 2.3 모델 구조

MoGe는 특징 추출을 위한 **Vision Transformer(ViT) 인코더**와 **CNN 기반의 경량 업샘플러(decoder)**로 구성되며, 약 900만 프레임 이상의 다양한 도메인으로 구성된 대규모 혼합 데이터셋에서 학습된다.

모델의 주요 특징:
- **단일 이미지 → point map, depth map, normal map 동시 추정** (하나의 모델, 하나의 순전파)
- 실제 FOV 입력을 통해 모델 정확도 향상 가능 (선택적 조건)
- 2:1~1:2 비율의 다양한 해상도를 원활하게 지원
- **A100 또는 RTX3090(FP16, ViT-L 기준)에서 이미지당 60ms** 지연시간

MoGe-2(후속 모델)의 설명에서 확인되듯, 기반 인코더는 보간 가능한 위치 임베딩을 갖는 **DINOv2 기반 이미지 인코더**로, 임의의 이미지 해상도(예: 14픽셀의 배수)를 수용한다.

**모델 구조 요약:**

```
입력 이미지 (단일)
     │
     ▼
[ViT 인코더 (DINOv2 기반)]
  - 보간 가능한 위치 임베딩
  - 다양한 해상도 지원
     │
     ▼
[경량 CNN Decoder / 업샘플러]
     │
     ├──→ 3D Point Map (Affine-Invariant)
     ├──→ Depth Map
     ├──→ Normal Map
     ├──→ Camera FoV
     └──→ 유효 영역 마스크 (Infinity Mask)
```

---

### 2.4 성능 향상

MoGe는 실내 장면(NYUv2), 도로 뷰(KITTI, DDAD), 객체 스캔(GSO), 합성 동영상(Sintel)을 포함한 8개의 다양한 데이터셋에서 모노큘러 3D point map 추정 분야 최고 성능을 달성했다. Scale-invariant point map에서 평균 **Rel[p] 8.32%, δ₁[p] 93.1%** 로 UniDepth(12.9%, 87.7%)와 DUSt3R(14.7%, 81.7%) 대비 우수한 성능을 보였다.

Affine-invariant point map에서는 평균 **Rel[p] 6.43%, δ₁[p] 94.4%**를 달성하며, UniDepth(9.86%, 90.4%)와 DUSt3R(12.1%, 83.9%) 대비 상당한 개선을 보여주어 전역 스케일 및 이동 모호성 처리의 효과성을 입증했다.

로컬 기하학 추정에서도 기존 방법 대비 region-wise Rel[p]를 약 **30% 감소**(7.96% → 5.33%)시켰으며, 이는 객체 간 상대 거리가 모호한 시나리오에서 멀티스케일 로컬 기하학 손실의 효과를 강조한다.

**방법별 성능 비교 정리표:**

| 방법 | Scale-Inv Rel[p] (↓) | Scale-Inv δ₁[p] (↑) | Affine-Inv Rel[p] (↓) | Affine-Inv δ₁[p] (↑) |
|------|----------------------|----------------------|----------------------|----------------------|
| DUSt3R | 14.7% | 81.7% | 12.1% | 83.9% |
| UniDepth | 12.9% | 87.7% | 9.86% | 90.4% |
| **MoGe** | **8.32%** | **93.1%** | **6.43%** | **94.4%** |

---

### 2.5 한계점

MoGe(v1)는 **metric scale(절대 거리)을 예측하지 못하며**, 세밀한 디테일이 부족하여 많은 다운스트림 작업에서의 적용에 제한이 있다.

후속 연구(MoGe-2)에서도 언급되듯, **얇은 선이나 머리카락 같은 극도로 미세한 구조물을 포착하거나**, 전경과 배경 사이에 큰 스케일 차이가 있을 때 직선적이고 정렬된 구조를 유지하는 데 어려움이 있다. 실세계 metric scale의 모호성은 분포 밖(out-of-distribution) 시나리오에서 편차를 초래할 수 있다.

또한 다양한 장면에 대한 일반화 평가가 제한적이며, 도전적인 조명, 폐색(occlusion), 극단적인 장면 구성을 가진 이미지에 대한 모델의 일반화 능력 평가가 추가로 필요하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능케 하는 핵심 설계 원칙

MoGe는 **affine-invariant 표현**을 활용함으로써, 추정 과정에서 스케일과 위치 이동과 관련된 모호성을 크게 줄여 일반화 성능 향상의 기반을 마련한다.

ROE 전역 정렬 솔버와 affine-invariant point map 표현의 결합이 MoGe 성능에 핵심적이며, Ablation study를 통해 affine-invariant 표현이 scale-invariant 방식이나 depth+카메라 파라미터 예측 방식보다 일관되게 우수하여 **focal-distance 모호성 제거가 효과적인 기하학 학습의 핵심**임을 확인했다. 최적 정렬 전략 또한 중앙값(median) 정렬 전략보다 크게 우수하여 강건한 학습 지도에서의 중요성을 강조한다.

### 3.2 대규모 혼합 데이터셋을 통한 일반화

모델은 대규모 혼합 데이터셋에서 학습되며, 강한 일반화 능력과 높은 정확도를 보여준다. 8개의 미학습 데이터셋에 대한 포괄적 평가에서 3D point map, depth map, 카메라 FoV 추정을 포함한 모든 태스크에서 최신 기술 방법들을 크게 능가한다.

제로샷 평가를 통해 NYUv2, KITTI, DIODE 등 다양한 미학습 데이터셋에서 point map 정확도 지표(Relp, δp1)와 깊이 추정 정확도(Reld) 모두에서 다른 최신 방법들을 일관되게 능가하며 강한 일반화 능력을 입증했다.

### 3.3 후속 연구(MoGe-2)에서의 일반화 확장

MoGe-2는 MoGe를 직접 확장하는 고급 단안 기하학 추정 프레임워크로, 단일 RGB 이미지에서 정밀한 3D metric 기하학을 날카로운 디테일로 복원할 수 있다. 이 방법은 **상대 기하학(형상) 예측과 metric 스케일 복원을 신중하게 분리**하고, 개선된 데이터 큐레이션 프로토콜을 도입하여 오픈 도메인 기하학 추정에서 새로운 성능 기준을 세운다.

실제 데이터의 노이즈와 오류가 예측 기하학의 세밀한 디테일을 저하시킨다는 점을 발견하고, **실제 데이터를 합성 레이블로 필터링하고 완성하는 통합 데이터 정제 접근법**을 개발하여 전체 정확도를 유지하면서 복원 기하학의 세밀도를 크게 향상시킨다.

모델은 DINOv2의 보간 가능한 위치 임베딩을 활용하여 **가변 입력 해상도를 지원**함으로써 다양한 실제 환경에서의 범용성을 확보한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

향상된 단안 기하학 추정 능력은 자율 주행, 증강 현실, 로봇 공학 등 정확한 3D 장면 이해에 의존하는 다양한 응용에 시사점을 가진다. 이러한 시스템이 단일 카메라만으로 실세계를 더 잘 인식하고 상호작용할 수 있게 해주며, 이 연구에서 얻은 통찰은 다양한 3D 인식 태스크를 위한 더 효과적인 학습 전략 및 아키텍처 설계 연구에 영감을 줄 수 있다.

MoGe는 실세계 응용에서 단안 기하학 추정의 잠재력을 높이며, **3D 세계 모델링, 자율 주행 등 다양한 태스크를 촉진하는 기초 도구**로 기능할 수 있다.

**영향을 받는 연구 분야:**

- **3D 재구성 및 Novel View Synthesis:** 단안 입력만으로 고품질 3D 복원이 가능해져 NeRF, 3DGS 계열 연구의 초기화 품질 향상
- **비디오 기하학 추정:** GeometryCrafter(2025) 등 일관된 비디오 기하학 추정 연구로 확장
- **자율 주행:** 단일 카메라로부터의 신뢰성 있는 깊이/포인트 추정으로 인식 시스템 강화
- **로보틱스/XR:** 실시간 단안 기하학 추정을 통한 환경 이해 가능

### 4.2 앞으로 연구 시 고려할 점

#### ① Metric Scale 문제

MoGe는 단안 입력에 대한 affine-invariant point map을 예측하며 robust한 정렬 솔버로 최고 성능을 달성했지만, **metric scale을 처리하지 못하고 세밀한 디테일이 부족하여 많은 다운스트림 태스크에서 적용이 제한된다.**

MoGe-2에서 이를 해결하기 위해 상대 기하학 복원과 전역 스케일 예측을 **분리(decoupling)**하는 방식을 채택하여 affine-invariant 표현의 장점을 유지하면서도 정확한 metric 재구성을 가능하게 했다.

#### ② 세밀한 디테일 향상 방향

합성 레이블을 활용해 실제 데이터를 강화하는 **실용적인 데이터 정제 파이프라인**을 제안하여 전체 정확도를 손상시키지 않고 기하학적 세밀도를 크게 향상시킬 수 있었다.

#### ③ 데이터 확장성 및 표준화

현재 이 분야에는 공유된 표준 학습 세트가 없으며, 사용되는 학습 데이터의 양과 성능이 반드시 상관관계를 갖지는 않는다. 향후 연구에서 **표준화된 벤치마크와 데이터 프로토콜 구축**이 중요한 과제이다.

#### ④ 아키텍처 확장 방향

MoGe-2 아키텍처가 다중 뷰나 시간적 데이터 같은 추가적인 모달리티를 처리하도록 적응되거나 확장될 수 있는지, 그리고 다양한 컴퓨팅 제한을 가진 장치에서 입력 이미지 해상도 변화에 대한 모델의 강건성이 실제 응용에 어떤 영향을 미치는지가 중요한 연구 방향이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 아이디어 | 한계 |
|------|------|--------------|------|
| **LeReS** | 2021 | Affine-invariant 깊이 + 포인트 클라우드 모듈로 shift/focal 복원 | 2단계 파이프라인 |
| **Metric3D (v1/v2)** | 2023/2024 | 정준 카메라 변환 모듈로 metric 모호성 처리, 대규모 데이터 | 카메라 파라미터 의존성 |
| **DUSt3R** | 2024 | 2-view 이미지 → 카메라 공간 point map 직접 예측 (End-to-End) | 단안 비최적화, focal-distance 모호성 |
| **UniDepth (v1/v2)** | 2024/2025 | 자기-프롬프트 가능한 카메라 모듈 + 밀집 카메라 표현으로 metric depth 추정 | 복잡한 카메라 모듈 |
| **Depth Pro** | 2024 | 멀티패치 ViT + 합성 데이터 학습으로 경계 선명도 향상 | 기하학 정확도 한계 |
| **MoGe** | 2024 | Affine-invariant point map + ROE 솔버 + 멀티스케일 로컬 손실 | Metric scale 없음, 디테일 부족 |
| **MoGe-2** | 2025 | Metric scale 분리 예측 + 데이터 정제 파이프라인 | 극세밀 구조 처리 한계 |

LeReS는 affine-invariant 깊이 예측 모듈과 포인트 클라우드 모듈로 구성된 2단계 파이프라인을 도입했으며, UniDepth는 이후 깊이 추정 모듈을 조건화하기 위한 밀집 카메라 표현을 예측하는 자기-프롬프트 카메라 모듈을 제안했다. DUSt3R는 두 뷰 이미지를 카메라 공간 point map으로 직접 변환하는 end-to-end 모델을 채택했고 단안 시나리오에서도 사용 가능하지만, 카메라 공간 scale-invariant pointmap 방식은 focal-distance 모호성에 취약하여 pointmap 학습을 저해한다. 이에 반해 MoGe는 단안 입력에 최적화되어 affine-invariant point map과 정밀하게 설계된 학습 감독을 사용해 더 효과적인 기하학 학습을 달성한다.

2024~2025년에는 UniDepth, Marigold 등 범용적이고 확산 기반의 MMDE 모델 개발에 연구가 수렴되어 metric 정확도와 생성적 적응성 모두를 추구하고 있다. 이러한 변화는 데이터 특화 지도에서 기반 수준의 인식으로의 더 넓은 전환을 반영하며, 모델 아키텍처, 손실 설계, 대규모 사전 학습의 발전에 의해 주도된다.

---

## 📚 참고 자료 및 출처

1. **arXiv 논문 원문:** Wang et al., "MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision," arXiv:2410.19115, 2024. [https://arxiv.org/abs/2410.19115](https://arxiv.org/abs/2410.19115)
2. **CVPR 2025 공식 게재본:** OpenAccess CVPR 2025, pages 5261–5271. [https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_MoGe_...](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_MoGe_Unlocking_Accurate_Monocular_Geometry_Estimation_for_Open-Domain_Images_with_CVPR_2025_paper.pdf)
3. **공식 GitHub 저장소 (Microsoft Research):** [https://github.com/microsoft/MoGe](https://github.com/microsoft/MoGe)
4. **arXiv HTML 전문:** [https://arxiv.org/html/2410.19115v1](https://arxiv.org/html/2410.19115v1)
5. **IEEE Xplore 게재본:** [https://ieeexplore.ieee.org/document/11094249](https://ieeexplore.ieee.org/iel8/11091818/11091608/11094249.pdf)
6. **MoGe-2 후속 논문:** Wang et al., "MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details," arXiv:2507.02546, 2025. [https://arxiv.org/abs/2507.02546](https://arxiv.org/abs/2507.02546)
7. **Quick Review (liner.com):** [https://liner.com/review/moge-unlocking-accurate-monocular-geometry-estimation-for-opendomain-images-with](https://liner.com/review/moge-unlocking-accurate-monocular-geometry-estimation-for-opendomain-images-with)
8. **Literature Review (themoonlight.io):** [https://www.themoonlight.io/en/review/moge-unlocking-accurate-monocular-geometry-estimation-for-open-domain-images-with-optimal-training-supervision](https://www.themoonlight.io/en/review/moge-unlocking-accurate-monocular-geometry-estimation-for-open-domain-images-with-optimal-training-supervision)
9. **AI Models FYI 논문 상세:** [https://www.aimodels.fyi/papers/arxiv/moge-unlocking-accurate-monocular-geometry-estimation-open](https://www.aimodels.fyi/papers/arxiv/moge-unlocking-accurate-monocular-geometry-estimation-open)
10. **Emergent Mind (MoGe-2 분석):** [https://www.emergentmind.com/topics/moge-2](https://www.emergentmind.com/topics/moge-2)
11. **Survey on Monocular Metric Depth Estimation (MDPI, 2025):** [https://www.mdpi.com/2073-431X/14/11/502](https://www.mdpi.com/2073-431X/14/11/502)
12. **UniDepth 논문:** Piccinelli et al., "UniDepth: Universal Monocular Metric Depth Estimation," CVPR 2024.
13. **CVPR 2025 포스터 페이지:** [https://cvpr.thecvf.com/virtual/2025/poster/34233](https://cvpr.thecvf.com/virtual/2025/poster/34233)
