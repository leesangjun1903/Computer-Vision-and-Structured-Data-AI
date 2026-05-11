
# Distilling Monocular Foundation Model for Fine-grained Depth Completion (DMD³C)
**[CVPR 2025] — Yingping Liang, Yutao Hu, Wenqi Shao, Ying Fu**

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

Depth completion은 희소(Sparse) LiDAR 입력으로부터 밀집(Dense) 깊이 맵을 예측하는 태스크인데, 센서의 희소한 깊이 어노테이션이 세밀한 기하학적 특징 학습에 필요한 밀집 감독(Dense Supervision)의 가용성을 제한한다.

이 문제를 해결하기 위해 저자들은 강력한 단안(Monocular) Foundation Model을 활용하여 depth completion에 밀집 감독을 제공하는 **2단계 지식 증류(Knowledge Distillation) 프레임워크**를 제안한다.

### 🏆 주요 기여

| 기여 항목 | 내용 |
|---|---|
| 2단계 KD 프레임워크 | Monocular Foundation Model로부터 Depth Completion 모델로 지식 전달 |
| LiDAR 시뮬레이션 데이터 생성 | 그라운드 트루스 없이 학습 데이터 자동 생성 |
| SSI Loss | 스케일·시프트 불변 손실 함수 설계 |
| SOTA 달성 | KITTI Depth Completion 벤치마크 1위 |

KITTI 리더보드에서 제안된 **DMD³C**가 제출 당시 1차 RMSE 지표 기준 모든 다른 방법을 능가하며 **1위**를 기록하였다.

---

## 2. 문제 정의, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

실외 환경에서는 LiDAR 등 감지 기술의 한계로 밀집 깊이 어노테이션 획득이 어렵다. KITTI 데이터셋은 실외 Depth Completion의 핵심 벤치마크이지만 이미지의 약 5%만 어노테이션되어 있어 매우 희소하다. 이를 해결하기 위해 복잡한 후처리 및 다중 프레임 융합 기법을 사용하더라도 최대 약 20% 수준의 커버리지밖에 달성하지 못한다.

더불어 동적 객체나 원거리 객체의 깊이 그라운드 트루스는 포함될 수 없으며, 희소한 그라운드 트루스는 세밀한 Depth Completion 모델 훈련에 큰 도전을 제기한다.

### 2-2. 제안 방법 — 2단계 지식 증류 프레임워크

#### 🔵 Stage 1: Pre-training (데이터 생성 전략)

1단계에서는 단안 깊이 추정(Monocular Depth Estimation)과 메쉬 재구성(Mesh Reconstruction)을 활용하여 자연 이미지로부터 학습 데이터를 생성하는 데이터 생성 전략을 제안하며, 이는 어떠한 LiDAR나 그라운드 트루스도 없이 모델이 기하학적 특징을 학습할 수 있게 한다.

구체적으로, 추정된 단안 깊이를 이용해 장면을 재구성한 뒤 LiDAR 스캔 과정을 시뮬레이션하여 훈련용 희소 포인트를 생성한다.

#### 🟠 Stage 2: Fine-tuning (SSI Loss 적용)

2단계에서는 레이블된 데이터셋에서 파인튜닝할 때 Monocular Depth Estimation을 위한 Foundation Model을 활용한다. 희소 그라운드 트루스는 L1 Loss를 통해 실제 깊이 스케일을 제공하고, 이 방법은 세밀한 감독을 위해 밀집 단안 깊이를 통합하여 이 과정을 강화한다. 그러나 단안 깊이 맵에는 고유한 스케일 및 시프트 모호성이 존재한다. 이러한 문제를 해결하기 위해, **SSI Loss(Scale- and Shift-Invariant Loss)**를 사용하여 예측을 밀집 단안 깊이에 정렬시키고 실제 세계의 깊이 스케일에 맞추어 더 정확한 Depth Completion을 보장한다.

#### 📐 SSI Loss 수식

SSI Loss는 $D_f$와 $D_m$ 사이의 스케일 및 시프트 차이에 불변하며, 다음과 같이 정식화된다:

$$\mathcal{L}_{\text{SSIL}} = \min_{s, b} \left| D_f - (s \cdot D_m + b) \right|$$

여기서:
- $D_f$: 모델이 예측한 밀집 깊이 맵 (Depth Completion 출력)
- $D_m$: Monocular Foundation Model이 추정한 깊이 맵 (Teacher)
- $s$: 스케일 파라미터
- $b$: 시프트(바이어스) 파라미터

**전체 학습 손실 함수:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{L1}} + \lambda_1 \mathcal{L}_{\text{SSIL}} + \lambda_2 \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{L1}}$: 희소 그라운드 트루스와의 지도 학습 손실
- $\mathcal{L}_{\text{SSIL}}$: SSI 기반 증류 손실
- $\mathcal{L}_{\text{reg}}$: 정규화 항(Gradient Matching 등)

이 프레임워크는 SSI Loss와 **그라디언트 매칭(Gradient Matching) 항**을 통합하여 선명도를 보존하고 깊이 불연속성과 정렬시킴으로써 완성된 깊이 맵의 충실도를 향상시킨다.

### 2-3. 모델 구조

증류에 사용되는 Monocular Foundation Model로는 강건한 성능을 바탕으로 **Depth Anything V2**를 채택하였다. Depth Completion 네트워크 아키텍처는 주로 **ResNet 블록**으로 구성된 **BP-Net**을 기본 모델로 사용한다.

2단계에서는 깊이 어노테이션과 함께 지도 손실(L1 Loss) 및 제안된 SSI Loss를 사용하여 레이블된 데이터셋에 사전 훈련된 모델을 적응시킨다. 희소 그라운드 트루스와의 L1 Loss를 통해 Depth Completion 모델이 희소 감독하에 실제 세계 깊이 스케일에 적응할 수 있게 하며, 제안된 단안 모델 증류는 SSI Loss를 사용해 Monocular Foundation Model을 증류하여 밀집 감독으로 세밀한 디테일을 유지한다.

**모델 구조 요약:**

```
[Stage 1 Pre-training]
 Unlabeled RGB Images
       ↓
 Monocular Depth (Depth Anything V2)
       ↓
 Mesh Reconstruction
       ↓
 LiDAR Scan Simulation → Sparse Points + Pseudo Dense GT
       ↓
 BP-Net (Base Architecture, ResNet Blocks) Pre-training

[Stage 2 Fine-tuning]
 Real-world Dataset (KITTI / NYUv2)
 Sparse LiDAR Input + RGB Image
       ↓
 BP-Net (Pre-trained)
       ↓ ← SSI Loss (Teacher: Depth Anything V2)
       ↓ ← L1 Loss (Sparse GT)
       ↓ ← Gradient Matching (Regularization)
 Fine-grained Dense Depth Map
```

### 2-4. 성능 향상

제안된 2단계 증류 프레임워크인 DMD³C는 KITTI 벤치마크에서 **RMSE 678.12mm**를 달성하며 1위를 기록하였다. 이는 이전 최고 성능 방법인 ImprovingDC(686.46mm) 및 BP-Net(684.90mm)에 비해 주목할 만한 개선이다. RMSE 외에도 MAE(194.46), iRMSE(1.82), iMAE(0.85)를 포함한 다른 평가 지표에서도 경쟁력 있거나 우월한 성능을 보인다.

NYUv2 데이터셋(실내 장면)에서도 최고 RMSE(**0.085**)와 가장 높은 $\delta_{1.25}$ 비율(**99.7%**)을 달성하였다.

DMD³C 모델은 다른 모델들이 어려움을 겪는 복잡한 장면, 특히 날카로운 객체 경계 유지 및 세밀한 디테일 포착에서 뛰어난 성능을 보인다.

### 2-5. 한계점

단안 깊이 추정은 실제 환경에서 고유한 스케일 모호성(Scale Ambiguity)의 문제를 안고 있으며, 이를 해결하기 위해 SSI Loss를 도입했지만, 이는 스케일 정렬에 추가적인 하이퍼파라미터 조정이 필요함을 의미한다.

추가적인 한계로는:
- **Teacher 모델 의존성**: Depth Anything V2와 같은 강력한 외부 Foundation Model에 의존하므로, Teacher 모델의 품질이 성능에 직접적인 영향을 미침
- **계산 비용**: 모델 훈련에 4개의 NVIDIA RTX A100 GPU가 필요하므로, 경량화 측면에서의 개선 여지가 존재한다.
- **도메인 특화성**: KITTI와 NYUv2 중심의 평가로, 의료 영상이나 수중 환경 등 완전히 다른 도메인에서의 성능은 별도 검증이 필요

---

## 3. 일반화 성능 향상 가능성

### 3-1. 다양한 도메인에서의 Zero-shot 일반화

이 논문은 제안된 방법이 다양한 네트워크 아키텍처로 어떻게 일반화되는지, 그리고 **Zero-shot 설정에서 Out-of-domain 데이터셋**에 어떻게 일반화되는지를 핵심 연구 질문으로 탐구한다.

NYUv2 실내 데이터셋에서도 최고 RMSE와 $\delta_{1.25}$ 비율을 달성하였는데, 이는 자율주행 실외 환경을 넘어 **다양한 실내 환경에까지 프레임워크의 강건성과 일반화 능력이 확장됨**을 나타내어, 그 다용도성을 보여준다.

### 3-2. 일반화 성능 향상의 핵심 요소

| 핵심 요소 | 일반화에 미치는 영향 |
|---|---|
| **다양한 자연 이미지 사전 학습** | 특정 도메인(KITTI)의 분포에 과적합되지 않고 넓은 범위의 기하학적 패턴 학습 |
| **레이블 없는 데이터 활용** | 추가적인 어노테이션 없이 어느 도메인에서든 Pre-training 가능 |
| **SSI Loss** | 스케일·시프트 불변이므로, 스케일이 다른 새로운 환경에서도 안정적 증류 가능 |
| **Foundation Model (Depth Anything V2)** | 대규모 사전 학습으로 강건한 기하학적 표현을 갖춘 Teacher 제공 |

DMD³C는 Monocular Foundation Model로부터의 지식 증류를 통해 세밀한 Depth Completion을 위한 새로운 프레임워크를 제시하며, 이 접근법은 **그라운드 트루스 감독이 없는 희소 데이터 영역에서도** 깊이 추정 정확도를 크게 향상시킨다.

### 3-3. 다양한 아키텍처로의 확장 가능성

KITTI와 NYUv2 모두에서 일관된 성능 향상이, Monocular Foundation Model 및 Scale-and-Shift-Invariant Loss를 활용한 2단계 증류 프레임워크가 희소 감독 및 스케일 모호성 과제를 극복하는 데 효과적임을 확인시켜 준다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 미래 연구에 미치는 영향

#### ① Foundation Model 기반 Dense Supervision의 패러다임 전환
본 논문은 **레이블이 없는 데이터로부터 Foundation Model을 통해 Dense Pseudo-GT를 생성하고, 이를 지식 증류 형태로 활용**하는 새로운 패러다임을 제시한다. 이는 어노테이션 비용이 높은 3D 비전 태스크 전반(표면 법선 추정, 광학 흐름, 점군 완성 등)에 유사한 방식이 적용될 수 있음을 시사한다.

#### ② SSI Loss의 광범위한 활용 가능성
스케일·시프트 불변 손실 함수는 단안 깊이 추정 모델들이 공통적으로 가지는 **Affine Ambiguity** 문제를 다루므로, 다른 Metric Depth Estimation 연구에서도 이 손실 함수의 변형된 형태가 널리 사용될 가능성이 높다.

#### ③ 비지도 사전 학습의 확장
1단계에서 단안 깊이 추정과 메쉬 재구성으로 자연 이미지로부터 학습 데이터를 생성하여 **어떠한 LiDAR나 그라운드 트루스도 필요하지 않은** 방식은 웹에서 수집된 이미지를 사용한 대규모 사전 훈련 파이프라인 개발로 이어질 수 있다.

### 4-2. 앞으로 연구 시 고려할 점

#### 🔬 기술적 고려사항

| 고려사항 | 세부 내용 |
|---|---|
| **Teacher 모델 선택** | Depth Anything V2 외에도 더 강력하거나 경량화된 Foundation Model이 등장할 경우, 교체만으로 성능 향상 가능 |
| **Multi-Teacher 증류** | 다중 교사 증류 프레임워크는 여러 깊이 추정 모델의 강점을 결합하는 방향으로 발전하고 있어, DMD³C에 Multi-Teacher 방식 통합 시 추가 성능 향상이 기대됨 |
| **경량화** | ResNet 기반 BP-Net이 아닌 Transformer 기반 아키텍처나 경량 백본 적용 시의 성능·속도 Trade-off 검토 필요 |
| **스케일 문제** | SSI Loss에서 $s$와 $b$를 최적화할 때의 수렴 안정성과 최적화 전략에 대한 추가 연구 필요 |

#### 🌐 일반화 관련 고려사항

- **도메인 다양화**: KITTI·NYUv2 이외에도 **의료 영상(내시경 깊이), 수중, 항공 영상** 등 이질적 도메인에서의 제로샷 성능 검증 필요
- **센서 다양성**: Depth Completion 시스템이 진정으로 유용하기 위해서는 정확성뿐 아니라 장면 유형 및 희소 깊이의 희소성 패턴 등 다양한 입력 분포에서도 계속 잘 동작하는 강건성이 필요하며, 이는 단일 시스템이 다양한 조건에서 잘 동작할 수 있게 해준다.
- **동적 객체**: 이동 중인 자동차, 보행자 등 동적 객체에서의 메쉬 재구성 품질 저하 문제에 대한 연구 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도/학회 | 방법론 | KITTI RMSE (mm) | 특징 |
|---|---|---|---|---|
| **DMD³C** (본 논문) | CVPR 2025 | 2단계 KD + SSI Loss | **678.12** | Foundation Model 증류, LiDAR 시뮬레이션 |
| **ImprovingDC** | ~2024 | — | 686.46 | 이전 SOTA |
| **BP-Net** | ~2024 | ResNet 기반 SPN | 684.90 | DMD³C의 기반 모델 |
| **OGNI-DC** | ECCV 2024 | Optimization-Guided Neural Iteration | — | 강한 일반화 성능; NYUv2로 훈련 후 VOID에서 MAE 35.5% 감소, KITTI로 훈련 후 DDAD에서 RMSE 25% 감소 |
| **Flexible DC** | CVPR 2024 | ASC Module | — | 희소 및 다양한 포인트 밀도에서 특히 더 희소한 설정에서 SOTA 성능 달성 |

Depth Completion은 희소 깊이 맵으로부터 밀집 깊이 맵을 생성하는 태스크로, 초기에는 희소 깊이 맵에서 직접 희소성을 채우는 방식에 집중했으나, 현대 기법들은 RGB 이미지를 가이드 도구로 활용하여 이 문제를 해결하고 있다.

---

## 📚 참고 자료 및 출처

1. **arXiv 논문 원문**: [arXiv:2503.16970](https://arxiv.org/abs/2503.16970) — *Distilling Monocular Foundation Model for Fine-grained Depth Completion* (March 2025)
2. **CVPR 2025 Open Access**: [CVPR 2025 Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liang_Distilling_Monocular_Foundation_Model_for_Fine-grained_Depth_Completion_CVPR_2025_paper.html) — Liang et al., CVPR 2025, pp. 22254–22265
3. **IEEE Xplore**: [IEEE Conference Publication](https://ieeexplore.ieee.org/document/11093959/)
4. **공식 GitHub 코드**: [DMD³C GitHub](https://github.com/Sharpiless/DMD3C) — *1st Place on KITTI Depth Completion Leaderboard*
5. **CVPR 2025 포스터**: [CVPR Virtual Poster](https://cvpr.thecvf.com/virtual/2025/poster/33728)
6. **Quick Review (Liner)**: [Liner.com Review](https://liner.com/review/distilling-monocular-foundation-model-for-finegrained-depth-completion)
7. **OGNI-DC (ECCV 2024)**: *OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations* — [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00319.pdf)
8. **Flexible Depth Completion (CVPR 2024)**: *Flexible Depth Completion for Sparse and Varying Point Densities* — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_Flexible_Depth_Completion_for_Sparse_and_Varying_Point_Densities_CVPR_2024_paper.pdf)
9. **Depth Completion Survey (MDPI 2022)**: *A Comprehensive Survey of Depth Completion Approaches* — [MDPI Sensors](https://www.mdpi.com/1424-8220/22/18/6969)
10. **Distill Any Depth (arXiv 2025)**: [arXiv:2502.19204](https://arxiv.org/html/2502.19204v1)
