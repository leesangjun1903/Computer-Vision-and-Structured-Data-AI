
# DiffusionDet: Diffusion Model for Object Detection

---

# 1. 핵심 주장 및 주요 기여 요약

DiffusionDet은 객체 검출(Object Detection)을 **노이즈 박스에서 객체 박스로의 디노이징 확산 과정(denoising diffusion process)**으로 정식화하는 새로운 프레임워크입니다. 이 모델은 확산 모델(diffusion model)을 객체 검출에 적용한 최초의 신경망 모델로, 기존의 성숙한 검출 모델들과 비교하여 우수한 성능을 달성한 새로운 검출 패러다임입니다.

### 주요 기여:
1. **새로운 패러다임 제시**: 확산 모델을 객체 검출에 적용한 최초의 논문입니다.
2. **유연성(Flexibility)**: 동적 박스 개수(dynamic number of boxes)와 반복적 평가(iterative evaluation)를 가능하게 하는 유연성을 보유합니다.
3. **Zero-shot 전이 성능**: COCO에서 CrowdHuman으로의 zero-shot 전이 시 더 많은 박스와 반복 단계로 평가할 때 5.3 AP 및 4.8 AP 향상을 달성합니다.
4. ICCV 2023 Best Paper Finalist에 선정되었습니다.

---

# 2. 상세 분석

## 2.1 해결하고자 하는 문제

기존 객체 검출 방법들은 다음과 같은 한계를 가지고 있습니다:

- **Anchor 기반 방법** (Faster R-CNN 등): 사전 정의된 앵커 박스에 의존하여, 양/음성 샘플 균형 문제가 있음
- **Query 기반 방법** (DETR, Sparse R-CNN 등): 학습 가능한 쿼리를 사용하지만, 고정된 수의 쿼리로 학습/추론이 결합되어 유연성이 제한됨
- 세그멘테이션 분야에서는 생성적 확산 모델이 성공적으로 적용되었으나, 객체 검출에는 성공적으로 적응된 사례가 없었으며 그 발전이 크게 뒤처져 있었습니다.

이 연구는 사전 정의된 객체 후보에 의존하지 않고 무작위 박스를 직접 정제(refine)하는 새로운 객체 검출 프레임워크를 개발하는 것을 목표로 합니다.

## 2.2 제안하는 방법 (수식 포함)

### (A) Forward Diffusion Process (전방 확산 과정)

학습 단계에서 객체 박스가 ground-truth 박스에서 랜덤 분포로 확산됩니다. Ground-truth 박스 $z_0 = b$가 주어졌을 때, $T$ 스텝에 걸쳐 점진적으로 가우시안 노이즈를 추가합니다:

$$q(z_t | z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta_t}\, z_{t-1},\; \beta_t \mathbf{I})$$

여기서 $\beta_t$는 분산 스케줄(variance schedule)입니다. Reparameterization trick을 사용하면 임의의 타임스텝 $t$에서 직접 샘플링할 수 있습니다:

$$z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

여기서 $\alpha_t = 1 - \beta_t$이고 $\bar{\alpha}\_t = \prod_{s=1}^{t} \alpha_s$입니다. 노이즈 스케일은 $\alpha_t$에 의해 제어되며, $\alpha_t$에 대해 단조 감소하는 코사인 스케줄을 채택합니다.

### (B) 데이터 표현

데이터 샘플은 바운딩 박스의 집합 $z_0 = b$로 정의되며, $b \in \mathbb{R}^{N \times 4}$는 $N$개의 박스 집합입니다. 신경망 $f_\theta(z_t, t, x)$는 노이즈가 추가된 박스 $z_t$에서 $z_0$를 예측하도록 학습되며, 대응하는 이미지 $x$에 조건화됩니다.

### (C) Training (학습)

먼저 원본 ground-truth 박스에 추가 박스를 패딩하여 모든 박스가 고정된 수 $N_{\text{train}}$에 도달하도록 합니다. 패딩된 ground-truth 박스에 가우시안 노이즈를 추가합니다.

Signal scaling factor $d$를 도입하여 box 좌표에 적용합니다:

$$z_t = \sqrt{\bar{\alpha}_t}\, (d \cdot z_0) + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

객체 검출은 이미지 생성 작업보다 상대적으로 높은 signal scaling 값을 선호한다는 것을 발견했습니다.

$N_{\text{train}}$개의 예측에 대해 set prediction loss를 적용합니다. 최적 운송(optimal transport) 할당 방법을 통해 각 ground truth에 가장 낮은 비용을 가진 상위 $k$개의 예측을 선택하여 다수의 예측을 할당합니다.

학습 손실 함수는 다음과 같습니다:

$$\mathcal{L} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{L1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{giou}} \mathcal{L}_{\text{giou}}$$

여기서 $\mathcal{L}\_{\text{cls}}$는 분류 손실 (focal loss), $\mathcal{L}\_{\text{L1}}$은 L1 회귀 손실, $\mathcal{L}_{\text{giou}}$는 GIoU 손실입니다.

### (D) Inference (추론)

추론 시, 모델은 무작위로 생성된 박스 집합을 점진적으로 출력 결과로 정제합니다.

가우시안 분포에서 샘플링된 박스로부터 시작하여 모델이 예측을 점진적으로 정제합니다. 각 샘플링 단계에서 무작위 박스 또는 이전 단계의 추정 박스가 검출 디코더로 전송되어 카테고리 분류와 박스 좌표를 예측합니다. 현재 단계의 박스를 얻은 후 DDIM을 사용하여 다음 단계의 박스를 추정합니다.

DDIM에 의한 역방향 샘플링:

$$z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot f_\theta(z_t, t, x) + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \frac{z_t - \sqrt{\bar{\alpha}_t} \cdot f_\theta(z_t, t, x)}{\sqrt{1 - \bar{\alpha}_t}}$$

### (E) Box Renewal 전략

예측된 박스는 원하는 예측(적절한 위치의 박스)과 원하지 않는 예측(임의 분포)으로 분류됩니다. 원하지 않는 박스를 다음 반복에 직접 전달하면 이점이 없으므로, 추론을 학습과 정렬시키기 위해 box renewal 전략을 제안합니다. 특정 임계값보다 낮은 점수의 원하지 않는 박스를 필터링한 후, 남은 박스에 가우시안 분포에서 새로 샘플링된 무작위 박스를 연결합니다.

## 2.3 모델 구조

DiffusionDet의 아키텍처는 크게 **두 부분**으로 구성됩니다:

### (1) Image Encoder
이미지 인코더가 입력 이미지에서 특징 표현(feature representation)을 추출합니다. 기본 설정으로 백본은 ResNet-50에 FPN을 결합합니다.

### (2) Detection Decoder
검출 디코더는 노이즈 박스를 입력으로 받아 카테고리 분류와 박스 좌표를 예측합니다. 검출 디코더는 하나의 detection head에 6개의 스테이지를 가지며, DETR 및 Sparse R-CNN의 구조를 따릅니다. 또한 DiffusionDet은 이 detection head를 여러 번 재사용할 수 있으며, 이를 "반복적 평가(iterative evaluation)"라고 합니다.

### Sparse R-CNN과의 차이점:
DiffusionDet과 Sparse R-CNN의 주요 차이는 (1) DiffusionDet은 무작위 박스에서 시작하지만 Sparse R-CNN은 고정된 학습 박스를 사용하고, (2) Sparse R-CNN은 proposal 박스와 대응하는 proposal 특징의 쌍을 입력으로 받지만 DiffusionDet은 proposal 박스만 필요하며, (3) DiffusionDet은 타임스텝 임베딩으로 반복적으로 detector head를 재사용할 수 있지만 Sparse R-CNN은 한 번만 사용한다는 것입니다.

## 2.4 성능 비교

### COCO val2017 벤치마크:

| 모델 | Backbone | AP |
|------|----------|------|
| Faster R-CNN | ResNet-50 | 40.2 |
| DETR | ResNet-50 | 42.0 |
| Sparse R-CNN | ResNet-50 | 45.0 |
| **DiffusionDet** (1 step, 300 boxes) | ResNet-50 | **45.8** |
| **DiffusionDet** (4 steps, 500 boxes) | ResNet-50 | **46.8** |

ResNet-50 백본을 사용한 DiffusionDet은 단일 샘플링 단계와 300개 무작위 박스로 45.8 AP를 달성하여, Faster R-CNN(40.2 AP)과 DETR(42.0 AP)을 크게 초과하고 Sparse R-CNN(45.0 AP)과 대등합니다.

ImageNet-21k 사전 학습된 Swin-Base를 백본으로 사용할 때 DiffusionDet은 52.5 AP를 달성하여 강력한 기존 모델들을 능가합니다.

### 추론 속도:
DiffusionDet은 단일 반복 단계와 300개 평가 박스를 사용할 때 Sparse R-CNN과 유사한 속도(30 FPS vs 31 FPS)를 보여줍니다.

## 2.5 한계

1. 현재 모델은 DINO와 같이 deformable attention 및 더 넓은 detection head 등 고급 컴포넌트를 사용하는 잘 개발된 연구들에 비해 뒤처져 있습니다.
2. **추론 속도**: 반복 단계와 박스 수를 늘리면 성능이 향상되지만 추론 시간이 증가합니다.
3. 더 진보된 확산 전략이 DiffusionDet의 속도 성능 저하 문제를 잠재적으로 해결할 수 있으며, 향후 연구에서 탐구할 계획입니다.

---

# 3. 일반화 성능 향상 가능성 (중점 분석)

## 3.1 Zero-Shot 전이 (핵심 일반화 지표)

COCO에서 사전 학습된 모델을 CrowdHuman 데이터셋에서 fine-tuning 없이 평가한 결과, DiffusionDet은 각각 5.3 AP와 4.8 AP의 향상을 보였습니다. 반면 기존 방법들은 제한적인 이득 또는 심각한 성능 저하(최대 14.0 AP 감소)를 보였습니다. DiffusionDet의 뛰어난 유연성은 희소한 환경과 밀집된 환경 모두에서 추가적인 fine-tuning 없이도 객체 검출에 매우 유용한 자산임을 시사합니다.

## 3.2 동적 박스 수의 이점

DiffusionDet은 학습과 평가 단계를 분리하여 재학습 없이 동적 박스 수와 반복적 정제를 허용하며 유연성을 높입니다.

DiffusionDet이 학습에 사용한 무작위 박스의 수에 관계없이, 약 2000개 무작위 박스의 포화 지점까지 $N_{\text{eval}}$이 증가함에 따라 정확도가 꾸준히 향상됩니다. 또한 $N_{\text{train}}$과 $N_{\text{eval}}$이 서로 일치할 때 더 좋은 성능을 보이는 경향이 있습니다.

## 3.3 일반화의 원천

DiffusionDet의 강력한 일반화 성능은 다음 요인에서 기인합니다:

1. **노이즈-투-박스(Noise-to-Box) 파이프라인**: 무작위 분포에서 시작하므로 사전 정의된 앵커나 학습된 쿼리에 대한 편향이 없음
2. 무작위 박스 생성 메커니즘은 알려진 클래스에 대한 의존도를 줄여, 학습 중 모델 매개변수의 편향을 완화합니다.
3. DiffusionDet은 무작위 박스에 대해 강건하며 신뢰할 수 있는 결과를 생성합니다.

## 3.4 다양한 도메인으로의 확장

DiffusionDet의 아키텍처는 RGB, 깊이(depth), 편광(polarimetric), 적외선(infrared), 레이더, 2D/3D 공간 도메인, low-shot 적응 시나리오 전반에서 SOTA 성능을 뒷받침하며, proposal 유연성, 멀티모달 융합, 불확실성 정량화에서 독특한 강점을 보입니다.

## 3.5 Open-World Object Detection에서의 일반화

DiffusionDet 기반의 DDOWOD는 무작위 박스를 생성하고 GT의 특성을 재구성하는 능력 덕분에 배경에 숨겨진 미지의 객체를 더 잘 탐지하고, 학습 중 알려진 클래스 객체에 대한 모델의 편향을 줄일 수 있습니다.

---

# 4. 향후 연구에 미치는 영향 및 고려할 점

## 4.1 연구에 미치는 영향

1. **생성 모델의 인식(Perception) 영역 확장**: DiffusionDet은 생성 모델(diffusion model)을 판별 과제(discriminative task)에 성공적으로 적용한 선구적 연구로, 이후 instance segmentation, 3D 객체 검출, 객체 추적 등으로 확산되었습니다.
   - DiffusionInst: 인스턴스를 벡터로 표현하고 인스턴스 세그멘테이션을 noise-to-vector 디노이징 프로세스로 정식화합니다.
   - Diff3Det: 확산 모델을 3D 객체 검출의 proposal 생성에 적용합니다.
   - 확산 기반 방법론을 추적 작업에 적용하여 7개 벤치마크에서 우수한 성능을 달성합니다.

2. **멀티모달 융합 연구 촉진**: RGBX-DiffusionDet과 같이 DiffusionDet을 확장하여 이질적인 2D 모달리티(깊이, 적외선, 편광 데이터)와 RGB 이미지를 원활하게 융합하는 프레임워크가 제안되었습니다.

3. **샘플링 효율성 연구**: DPM-Det은 DiffusionDet과 비교하여 검출 정확도와 속도 모두에서 개선을 이루어 정확도와 속도의 균형을 달성합니다.

## 4.2 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|----------|----------|
| **추론 속도 개선** | Consistency models, DPM-Solver++ 등 효율적 샘플링 기법 적용 필요 |
| **고급 컴포넌트 통합** | Deformable attention, 더 넓은 detection head 등 기존 DETR 계열의 기술 융합 |
| **확장성(Scalability)** | 대규모 데이터셋(Objects365 등)에서의 사전 학습 효과 검증 |
| **경량화** | 실시간 응용을 위한 모델 경량화 및 edge 디바이스 배포 전략 |
| **불확실성 정량화** | 확산 과정의 확률적 특성을 활용한 검출 불확실성 추정 |

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 발표연도/학회 | 핵심 접근 | COCO AP (R50) | 속도 | 특징 |
|------|------------|----------|---------------|------|------|
| DETR | 2020 / ECCV | Transformer + 학습 쿼리 | 42.0 | ~28 FPS | End-to-end, NMS 제거 |
| Deformable DETR | 2021 / ICLR | Deformable attention | 43.8 | ~19 FPS | 수렴 속도 개선 |
| Sparse R-CNN | 2021 / CVPR | Learnable proposals | 45.0 | ~31 FPS | 희소 후보 사용 |
| **DINO** | 2022 / ICLR 2023 | Contrastive denoising + 개선된 앵커 | **49.4** (12ep) | ~5 FPS | DETR 계열 최강 정확도 |
| **DiffusionDet** | 2022 / ICCV 2023 | Diffusion으로 검출 | 45.8~46.8 | ~30 FPS (1 step) | 유연성, zero-shot 전이 우수 |
| **RT-DETR** | 2023 / CVPR 2024 | 효율적 하이브리드 인코더 | **53.1** | **108 FPS** | 최초 실시간 End-to-end DETR |
| DPM-Det | 2024 / MMM | DPM-Solver++ 가이드 샘플링 | DiffusionDet 대비 향상 | 속도 향상 | DiffusionDet 기반 개선 |
| DDOWOD | 2024 / Pattern Recognition Letters | DiffusionDet + Open-World | - | - | 미지 클래스 탐지 |
| RT-DETRv2 | 2024 | Bag-of-freebies | >55.0 | 실시간 | RT-DETR 개선 |

### 주요 비교 포인트:

**DINO vs DiffusionDet**: DINO는 ResNet-50으로 12 에폭에서 49.4AP, 24 에폭에서 51.3AP를 달성하며 이전 DETR 계열 모델을 크게 능가합니다. DiffusionDet(45.8~46.8 AP)보다 절대 정확도에서는 높지만, DiffusionDet의 zero-shot 전이 유연성에서는 뒤처집니다.

**RT-DETR vs DiffusionDet**: RT-DETR-R50은 DINO-R50보다 정확도에서 2.2% AP 향상, 속도에서 약 21배(108 FPS vs 5 FPS)를 달성합니다. RT-DETR-R50/R101은 COCO에서 53.1%/54.3% AP를 달성하며 T4 GPU에서 108/74 FPS입니다. RT-DETR은 속도·정확도 모두에서 DiffusionDet을 능가하지만, DiffusionDet만의 확률적 생성 기반 유연성과 불확실성 모델링 능력은 갖추지 못합니다.

**DPM-Det**: DiffusionDet은 한 번의 학습으로 모든 추론을 가능하게 하지만, 느린 샘플링 속도 문제가 있습니다. DPM-Det은 DPM-Solver++를 활용하여 이 문제를 완화합니다.

---

# 6. 결론

DiffusionDet은 **확산 모델을 객체 검출에 최초로 적용**하여 noise-to-box라는 새로운 검출 패러다임을 제시한 혁신적 연구입니다. 절대 정확도에서는 DINO, RT-DETR 등 후속 연구에 비해 낮지만, **동적 박스 수, 반복적 평가, zero-shot 전이**에서 탁월한 유연성과 일반화 능력을 보여줍니다. 이 논문은 생성 모델을 판별 과제에 적용하는 연구 방향을 개척하여, 멀티모달 검출, 3D 검출, 오픈 월드 검출 등 다양한 후속 연구에 영감을 주고 있습니다.

---

## 참고자료

1. Chen, S., Sun, P., Song, Y., & Luo, P. (2022). "DiffusionDet: Diffusion Model for Object Detection." *arXiv:2211.09788* → ICCV 2023. ([arxiv.org](https://arxiv.org/abs/2211.09788))
2. ICCV 2023 Open Access Paper: ([thecvf.com](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_DiffusionDet_Diffusion_Model_for_Object_Detection_ICCV_2023_paper.pdf))
3. GitHub Repository: ([github.com/ShoufaChen/DiffusionDet](https://github.com/ShoufaChen/DiffusionDet))
4. Zhang, H. et al. "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection." ICLR 2023. ([openreview.net](https://openreview.net/pdf?id=3mRwyG5one))
5. Zhao, Y. et al. "DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)." CVPR 2024. ([arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069))
6. DPM-Det: "Diffusion Model Object Detection Based on DPM-Solver++ Guided Sampling." MMM 2024. ([link.springer.com](https://link.springer.com/chapter/10.1007/978-3-031-53308-2_28))
7. DDOWOD: "DiffusionDet for Open-World Object Detection." Pattern Recognition Letters, 2024. ([sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0167865524002903))
8. RGBX-DiffusionDet: "A framework for multi-modal RGB-X object detection using DiffusionDet." Pattern Recognition, 2025. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0031320325011239))
9. DiffusionDet 관련 연구 종합: ([emergentmind.com/topics/diffusiondet](https://www.emergentmind.com/topics/diffusiondet))
10. Zhangtemplar Reading Note on DiffusionDet: ([zhangtemplar.github.io/diffusion-det](https://zhangtemplar.github.io/diffusion-det/))

# DiffusionDet: Diffusion Model for Object Detection

---

## 1. 핵심 주장과 주요 기여 요약

**DiffusionDet**은 객체 탐지(Object Detection)를 **노이즈 박스(noisy boxes)에서 객체 박스(object boxes)로의 디노이징 확산 과정(denoising diffusion process)**으로 정의한 최초의 프레임워크이다. 기존 탐지기들이 경험적 객체 프라이어(anchor, proposal) 또는 학습 가능한 쿼리(learnable query)에 의존하는 반면, DiffusionDet은 **순수한 랜덤 박스**에서 출발하여 점진적으로 객체 위치와 크기를 정제한다.

### 주요 기여
1. **패러다임 전환**: 객체 탐지를 생성적 디노이징 과정으로 재정의 — 확산 모델을 객체 탐지에 최초로 성공적으로 적용
2. **유연성(Flexibility)**: 학습과 평가 단계를 분리(decoupling)하여, 한 번 학습한 모델로 (1) **동적 박스 수**(Dynamic number of boxes)와 (2) **반복 평가**(Iterative evaluation)가 가능
3. **일반화 성능**: COCO에서 학습 후 CrowdHuman으로 제로샷 전이 시 박스 수 증가만으로 **+5.3 AP**, 반복 스텝 증가로 **+4.8 AP** 향상 — 기존 방법은 성능 저하 또는 미미한 개선에 그침

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 객체 탐지 파이프라인의 핵심 한계점:

| 패러다임 | 예시 | 한계 |
|--------|------|------|
| 경험적 프라이어 기반 | Anchor boxes, Sliding windows, Region proposals | 수작업 설계 필요, 시나리오별 최적화 필요 |
| 학습 가능한 쿼리 기반 | DETR, Sparse R-CNN | 고정된 수의 쿼리에 의존, 학습-평가 간 결합 |

**핵심 질문**: *학습 가능한 쿼리의 대리(surrogate) 없이도 더 단순한 접근법이 가능한가?*

DiffusionDet은 이에 대해 **랜덤 박스 → 객체 박스**라는 noise-to-box 패러다임으로 답한다. 이는 확산 모델의 **noise-to-image** 철학과 정확히 대응된다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) 확산 모델의 전방 과정 (Forward Diffusion Process)

데이터 샘플 $\boldsymbol{z}_0$에 점진적으로 가우시안 노이즈를 추가하는 마르코프 체인:

$$q(\boldsymbol{z}_t | \boldsymbol{z}_0) = \mathcal{N}(\boldsymbol{z}_t \mid \sqrt{\bar{\alpha}_t}\,\boldsymbol{z}_0,\;(1 - \bar{\alpha}_t)\,\boldsymbol{I}) $$

여기서 $\bar{\alpha}\_t := \prod_{s=0}^{t} \alpha_s = \prod_{s=0}^{t}(1 - \beta_s)$이고, $\beta_s$는 노이즈 분산 스케줄이다.

이를 재매개변수화(reparameterization)하면:

$$\boldsymbol{z}_t = \sqrt{\bar{\alpha}_t}\,\boldsymbol{z}_0 + \epsilon\sqrt{1 - \bar{\alpha}_t}, \quad \epsilon \sim \mathcal{N}(0, \boldsymbol{I}) $$

#### (B) 학습 목적 함수

신경망 $f_\theta(\boldsymbol{z}_t, t)$가 노이즈가 추가된 $\boldsymbol{z}_t$로부터 원본 $\boldsymbol{z}_0$를 예측하도록 $\ell_2$ 손실로 학습:

$$\mathcal{L}_{\text{train}} = \frac{1}{2}\|f_\theta(\boldsymbol{z}_t, t) - \boldsymbol{z}_0\|^2 $$

#### (C) DiffusionDet에서의 적용

- **데이터 샘플**: $\boldsymbol{z}_0 = \boldsymbol{b} \in \mathbb{R}^{N \times 4}$ (N개의 바운딩 박스, 각 박스는 $(c_x, c_y, w, h)$ )
- **조건부 신경망**: $f_\theta(\boldsymbol{z}_t, t, \boldsymbol{x})$ — 이미지 $\boldsymbol{x}$에 조건화되어 노이즈 박스로부터 GT 박스 예측
- **카테고리 레이블** $\boldsymbol{c}$는 박스 예측과 함께 동시에 생성

#### (D) 역과정 (Reverse Process)의 사후 분포

베이즈 정리에 의한 사후 분포:

$$q(\boldsymbol{z}_{t-1}|\boldsymbol{z}_t, \boldsymbol{z}_0) = \mathcal{N}(\boldsymbol{z}_{t-1};\;\tilde{\mu}_t(\boldsymbol{z}_t, \boldsymbol{z}_0),\;\tilde{\beta}_t\,\boldsymbol{I}) $$

여기서:

$$\tilde{\mu}_t(\boldsymbol{z}_t, \boldsymbol{z}_0) := \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,\boldsymbol{z}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,\boldsymbol{z}_t $$

$$\tilde{\beta}_t := \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t $$

신경망은 이를 근사:

$$p_\theta(\boldsymbol{z}_{t-1}|\boldsymbol{z}_t) := \mathcal{N}(\boldsymbol{z}_{t-1};\;\mu_\theta(\boldsymbol{z}_t, t),\;\Sigma_\theta(\boldsymbol{z}_t, t)) $$

#### (E) Set Prediction Loss

매칭 비용(Matching Cost):

$$\mathcal{C} = \lambda_{cls} \cdot \mathcal{C}_{cls} + \lambda_{L1} \cdot \mathcal{C}_{L1} + \lambda_{giou} \cdot \mathcal{C}_{giou} $$

학습 손실:

$$\mathcal{L} = \lambda_{cls} \cdot \mathcal{L}_{cls} + \lambda_{L1} \cdot \mathcal{L}_{L1} + \lambda_{giou} \cdot \mathcal{L}_{giou} $$

여기서 $\lambda_{cls}=2.0$, $\lambda_{L1}=5.0$, $\lambda_{giou}=2.0$이며, 최적 운송(Optimal Transport) 기반 할당으로 각 GT에 top- $k$ 예측을 매칭한다.

### 2.3 모델 구조

DiffusionDet의 아키텍처는 **두 부분**으로 분리된다:

```
┌─────────────────────────────────────────────────┐
│                DiffusionDet 전체 파이프라인            │
│                                                 │
│  Raw Image ──→ [Image Encoder] ──→ Feature Map  │
│                    (1회 실행)                      │
│                                                 │
│  Gaussian     ──→ [Detection Decoder] ──→ Class  │
│  Noise/Boxes      (반복 실행 가능)          + Box  │
│                                                 │
└─────────────────────────────────────────────────┘
```

#### Image Encoder
- **백본**: ResNet-50/101 또는 Swin Transformer
- **FPN** (Feature Pyramid Network)을 통한 다중 스케일 특징 맵 생성
- **1회만 실행** — 계산 효율성 확보

#### Detection Decoder
- Sparse R-CNN에서 차용한 구조
- **6개 캐스케이딩 스테이지**로 구성
- 노이즈 박스 → RoI Align → RoI Features → FC → Box + Class 예측
- **핵심 차별점**:

| 특성 | Sparse R-CNN | DiffusionDet |
|------|-------------|-------------|
| 입력 박스 | 학습된 고정 박스 | 랜덤 박스 (가우시안) |
| 추가 입력 | proposal feature 필요 | 박스만 필요 |
| 반복 사용 | 1회만 사용 | 여러 번 재사용 가능 (iterative evaluation) |
| timestep 구분 | 없음 | timestep embedding으로 구분 |

#### 학습 과정 (Algorithm 1 요약)
1. GT 박스를 $N_{train}$개로 패딩 (가우시안 랜덤 박스 연결)
2. Signal scaling: $\boldsymbol{b} \leftarrow (\boldsymbol{b} \times 2 - 1) \times \text{scale}$ (기본 scale = 2.0)
3. 랜덤 timestep $t$ 샘플링 후 가우시안 노이즈로 박스 오염
4. Detection decoder가 오염된 박스로부터 GT 예측
5. Set prediction loss로 학습

#### 추론 과정 (Algorithm 2 요약)
1. 가우시안 분포에서 랜덤 박스 샘플링
2. DDIM 기반 반복 디노이징
3. **Box Renewal**: 각 스텝 후 낮은 점수의 박스를 새로운 랜덤 박스로 교체
4. 모든 스텝의 예측을 NMS로 앙상블

### 2.4 성능 향상

#### COCO val2017 결과 (ResNet-50)

| Method | AP | AP₅₀ | AP₇₅ |
|--------|-----|------|------|
| Faster R-CNN | 40.2 | 61.0 | 43.8 |
| DETR | 42.0 | 62.4 | 44.2 |
| Sparse R-CNN | 45.0 | 63.4 | 48.2 |
| **DiffusionDet (1@300)** | **45.8** | **64.1** | **50.4** |
| **DiffusionDet (4@500)** | **46.8** | **65.3** | **51.8** |

#### Swin-Base 백본

| Method | AP |
|--------|-----|
| Cascade R-CNN | 51.9 |
| Sparse R-CNN | 52.0 |
| **DiffusionDet (4@300)** | **53.3** |

#### LVIS v1.0 (ResNet-50, 대규모 어휘)

DiffusionDet은 반복 평가에서 COCO보다 LVIS에서 더 큰 이득을 보임:
- COCO: 45.8 → 46.6 (**+0.8 AP**)
- LVIS: 29.4 → 31.5 (**+2.1 AP**)

→ **더 도전적인 벤치마크에서 더 유용**

#### CrowdHuman 풀 튜닝

| Method | AP₅₀ | mMR↓ | Recall |
|--------|------|------|--------|
| Sparse R-CNN (1000) | 89.7 | 49.1 | 97.5 |
| **DiffusionDet (3@1000)** | **91.4** | **45.7** | **98.4** |

### 2.5 한계

1. **추론 속도**: DDIM 기반 반복 디노이징으로 인해 단일 스텝 대비 다중 스텝 사용 시 속도 저하 (30 FPS → 20 FPS, 6×2 설정)
2. **DINO 등 최신 DETR 변종 대비 성능 격차**: Deformable Attention, 넓은 detection head 등 고급 컴포넌트 미사용. 논문에서도 "DINO [108] 등과 비교하면 아직 뒤처진다"고 명시적으로 언급
3. **Signal scaling 민감도**: 박스가 4개 파라미터만으로 표현되어 이미지 생성보다 SNR에 더 민감
4. **GT 패딩 전략 의존성**: 패딩 방식에 따라 성능이 달라짐 (Repeat: 44.2 vs Cat Gaussian: 45.8)
5. **학습 비용**: 450K iterations, 8 GPU 필요

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

DiffusionDet의 일반화 성능은 논문의 **가장 차별화된 강점**으로, 다음의 세 가지 메커니즘에 의해 달성된다:

### 3.1 학습-평가 분리 (Decoupling)

기존 방법들은 $N_{train} = N_{eval}$이 강제되지만, DiffusionDet은 랜덤 박스를 사용하므로:

$$N_{train} \neq N_{eval} \quad \text{가 가능}$$

**Table 5 분석**:

| $N_{train}$ \ $N_{eval}$ | 100 | 300 | 500 | 1000 | 2000 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 42.9 | 44.4 | 44.5 | 44.6 | 44.6 |
| 300 | 42.8 | 45.7 | 46.2 | 46.3 | 46.4 |
| 500 | 41.9 | 45.8 | 46.3 | 46.7 | 46.8 |

→ **어떤 $N_{train}$으로 학습하든 $N_{eval}$ 증가 시 일관된 성능 향상**
→ 포화점 ~2000에서 안정화

### 3.2 제로샷 도메인 전이 (Zero-shot Transfer)

**COCO → CrowdHuman** (추가 파인튜닝 없이):

| Method | 300 boxes | 2000 boxes | 변화 |
|--------|-----------|------------|------|
| DETR | 61.3 | 61.3 | **+0.0** |
| Sparse R-CNN | 66.6 | 66.5 | **-0.1** |
| **DiffusionDet** | **66.6** | **71.9** | **+5.3** |

| Method | 1 step | 4 steps | 변화 |
|--------|--------|---------|------|
| Sparse R-CNN | 66.6 | 52.6 | **-14.0** |
| **DiffusionDet** | **66.6** | **71.4** | **+4.8** |

**핵심 통찰**: 
- 기존 방법은 고정된 쿼리가 학습 데이터의 분포에 과적합되어, 다른 도메인(밀집 장면)에서 적응 불가
- DiffusionDet은 랜덤 박스에서 시작하므로 **데이터 분포에 대한 가정이 최소화**
- 밀집 장면에서는 더 많은 박스와 스텝을 사용하면 되므로 **시나리오 적응이 하이퍼파라미터 조절만으로 가능**

### 3.3 반복 정제의 일반화 효과

확산 모델의 반복 디노이징 특성 덕분에:
- 100개 박스: 41.9 AP (1 step) → 46.1 AP (8 steps) = **+4.2 AP**
- 500개 박스: 46.3 AP (1 step) → 46.9 AP (8 steps) = **+0.6 AP**

→ **적은 박스로 학습해도 반복 정제로 성능 회복 가능** — 자원 제약 환경에서의 일반화에 유리

### 3.4 랜덤 시드에 대한 안정성

5개 독립 학습 인스턴스 × 10개 평가 시드 실험:
- 대부분 결과가 **45.7 AP 부근에 밀집**
- 모델 인스턴스 간 차이도 미미 (45.66~45.77)

→ **랜덤 초기화에 강건한 일반화 성능**

### 3.5 일반화 성능 향상을 위한 추가 가능성

1. **Deformable Attention 통합**: DINO의 deformable attention을 detection decoder에 적용하면 큰 객체와 작은 객체 모두에서 성능 향상 기대
2. **고급 샘플링 전략**: Consistency Models, DPM-Solver 등으로 속도-성능 트레이드오프 개선
3. **다중 도메인 학습**: 다양한 장면 밀도의 데이터셋 혼합 학습으로 일반화 강화
4. **더 큰 백본**: 논문에서도 Swin-Base까지 일관된 향상을 보여, 더 큰 모델에서의 확장 가능성 시사

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

#### (1) 탐지 패러다임의 확장
DiffusionDet은 **객체 탐지를 생성 모델의 관점에서 바라보는 새로운 시각**을 제시하였다. 이는 탐지 파이프라인의 발전을 다음과 같이 정리할 수 있다:

```
Sliding Window → Region Proposal → Anchor → Learnable Query → Random Box (DiffusionDet)
```

이 패러다임은 instance segmentation, pose estimation, 3D object detection 등 다른 인식 과제에도 확장 가능하다.

#### (2) 생성 모델의 인식 과제 적용 가속
DiffusionDet의 성공은 segmentation (SegDiff, Pix2Seq v2), panoptic segmentation (Bit Diffusion) 등에 이어 **확산 모델이 인식 과제에서도 경쟁력 있음**을 입증하였다.

#### (3) 유연한 배포 모델
한 번 학습으로 다양한 시나리오(sparse/crowded)에 배포 가능한 특성은 **실제 산업 환경에서의 활용 가치**를 높인다.

### 4.2 향후 연구 시 고려할 점

| 연구 방향 | 구체적 고려사항 |
|---------|-----------|
| **추론 효율성** | Consistency Models, Progressive Distillation 등을 적용하여 1-step 추론 달성 |
| **고급 컴포넌트 통합** | Deformable attention, wider head, DN-DETR의 query denoising 등 |
| **대규모 어휘 탐지** | LVIS에서의 큰 이득이 시사하듯, 장기 꼬리(long-tail) 분포 대응 강화 |
| **3D/비디오 확장** | 3D 바운딩 박스나 비디오 프레임 간 시간적 일관성을 확산 과정에 통합 |
| **확산 전략 최적화** | Noise schedule, signal scaling, box renewal threshold의 자동 최적화 |
| **학습 효율** | 450K iterations은 DETR 계열 대비 길므로, 학습 가속 기법 필요 |
| **개방 어휘(Open-vocabulary) 탐지** | 텍스트 조건부 확산과 결합하여 novel class 탐지 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 아이디어 | DiffusionDet과의 관계 |
|------|------|-----------|-------------------|
| **DETR** (Carion et al.) | 2020 | Learnable query + Transformer decoder로 end-to-end 탐지 | DiffusionDet의 출발점; 고정 쿼리의 한계를 DiffusionDet이 해결 |
| **Deformable DETR** (Zhu et al.) | 2021 | Deformable attention으로 DETR 수렴 가속 | DiffusionDet에 통합 가능한 직교적 기법; DiffusionDet은 dynamic box 속성에서 우월 |
| **Sparse R-CNN** (Sun et al.) | 2021 | Learnable proposals + proposal features | DiffusionDet의 decoder 구조 기반; 그러나 고정 박스 한계 |
| **DN-DETR** (Li et al.) | 2022 | Query denoising으로 DETR 학습 가속 | DiffusionDet과 유사하게 노이즈를 활용하나, 학습 보조 기법 vs. 전체 패러다임 차이 |
| **DINO** (Zhang et al.) | 2022 | Improved denoising anchor boxes + contrastive denoising | 현재 DiffusionDet보다 높은 성능; deformable attention + wider head 등 고급 기법 사용 |
| **DAB-DETR** (Liu et al.) | 2022 | Dynamic anchor boxes as queries | DiffusionDet과 유사하게 박스를 직접 쿼리로 사용하나, 학습된 고정 앵커 |
| **Bit Diffusion / Pix2Seq v2** (Chen et al.) | 2022 | 확산 모델로 panoptic segmentation | DiffusionDet과 동일한 확산 철학; segmentation은 image-to-image으로 더 자연스러움 |
| **Consistency Models** (Song et al.) | 2023 | 확산 모델의 1-step 생성 | DiffusionDet의 추론 속도 문제 해결에 직접 적용 가능 |
| **Group DETR** (Chen et al.) | 2022 | Group-wise one-to-many assignment | DiffusionDet의 OT 기반 할당과 보완적 |
| **YOLOX** (Ge et al.) | 2021 | Anchor-free one-stage detector + OTA | 실시간성에서 우위이나 유연성 부족 |

### DiffusionDet vs. DINO 비교 (주요 차이점)

| 측면 | DiffusionDet | DINO |
|------|-------------|------|
| 패러다임 | 생성적 확산 모델 | 판별적 query-based |
| 입력 | 순수 랜덤 박스 | 학습된 앵커 + 노이즈 |
| 유연성 | 동적 박스 수 + 반복 평가 | 고정 쿼리 수 |
| 성능 (COCO R50) | 46.8 | ~49+ (deformable attn 포함) |
| 제로샷 전이 | 매우 우수 | 제한적 |
| Attention 메커니즘 | Standard (RoI-based) | Deformable attention |

→ DiffusionDet은 **유연성과 일반화**에서, DINO는 **절대 성능**에서 우위. 두 접근의 결합이 유망한 방향이다.

---

## 참고자료

1. **Shoufa Chen, Peize Sun, Yibing Song, Ping Luo, "DiffusionDet: Diffusion Model for Object Detection," arXiv:2211.09788v2, Aug 2023.** (본 논문 원문)
2. Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models," NeurIPS 2020.
3. Carion, N., et al. "End-to-End Object Detection with Transformers (DETR)," ECCV 2020.
4. Sun, P., et al. "Sparse R-CNN: End-to-End Object Detection with Learnable Proposals," CVPR 2021.
5. Zhu, X., et al. "Deformable DETR: Deformable Transformers for End-to-End Object Detection," ICLR 2021.
6. Zhang, H., et al. "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection," ICLR 2022.
7. Li, F., et al. "DN-DETR: Accelerate DETR Training by Introducing Query DeNoising," CVPR 2022.
8. Song, J., Meng, C., & Ermon, S. "Denoising Diffusion Implicit Models (DDIM)," ICLR 2021.
9. Nichol, A.Q. & Dhariwal, P. "Improved Denoising Diffusion Probabilistic Models," ICML 2021.
10. Song, Y., et al. "Consistency Models," arXiv:2303.01469, 2023.
11. Chen, T., et al. "A Generalist Framework for Panoptic Segmentation of Images and Videos (Bit Diffusion)," arXiv:2210.06366, 2022.
12. Liu, S., et al. "DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR," ICLR 2022.
13. GitHub Repository: https://github.com/ShoufaChen/DiffusionDet

<details>
   
# DiffusionDet: Diffusion Model for Object Detection
### 1. 핵심 주장과 주요 기여
**DiffusionDet**은 객체 탐지(object detection) 문제를 **노이즈에서 정제로의 확산 과정(denoising diffusion process)**으로 재정의한 획기적인 프레임워크이다. 이 논문의 핵심 주장은 기존의 고정된 학습 가능한 쿼리(learnable queries)나 휴리스틱 객체 후보(heuristic object priors)에 의존하지 않고, 완전히 무작위 박스에서 시작하여 점진적으로 정제하는 방식이 더 유연하고 효과적이라는 것이다.[1]

주요 기여는 다음과 같다:

- **첫 확산 모델 적용**: 객체 탐지에 확산 모델을 성공적으로 적용한 최초 연구[1]
- **유연한 아키텍처**: 학습 시와 평가 시의 박스 개수 분리, 반복적 평가 가능[1]
- **우수한 일반화 성능**: COCO에서 CROWDHUMAN으로의 제로샷 전이에서 5.3 AP 향상[1]
- **광범위한 벤치마크 검증**: COCO, LVIS, CrowdHuman 등 다중 데이터셋에서 우수한 성능 입증[1]

***

### 2. 해결 문제와 제안 방법
#### 2.1 문제 정의

기존의 DETR 이후 쿼리 기반 탐지 방법들은 다음의 한계를 지닌다:

- **고정 쿼리 의존성**: 학습 시 정해진 개수의 학습 가능한 쿼리에 의존
- **평가 유연성 부족**: 평가 시 다른 개수의 후보를 사용하면 성능 저하
- **복잡한 파이프라인**: 다양한 휴리스틱 요소(앵커, 프로포절 등)를 포함

#### 2.2 제안 방법

**확산 프로세스 공식화**:

DiffusionDet은 객체 탐지를 조건부 확산 모델로 정식화한다. 순방향 노이징 프로세스는:[1]

$$q(z_t | z_0) = \mathcal{N}(z_t | \sqrt{\bar{\alpha}_t} z_0, (1-\bar{\alpha}_t)I)$$

여기서 $$z_0 = b$$는 지정된 박스들의 집합이고, $$\bar{\alpha}\_t = \prod_{i=1}^t \alpha_i$$이며, $$\alpha_i$$는 분산 스케줄에 의해 제어된다.[1]

**역프로세스 학습**:

신경망 $$f_\theta(z_t, t, x)$$는 조건부 이미지 특성 $$x$$를 고려하여 $$z_0$$을 예측하도록 학습되며, 손실 함수는:

$$\mathcal{L} = \mathbb{E}_{t} \lVert f_\theta(z_t, t, x) - z_0 \rVert_2^2$$

**신호 스케일링**:

DiffusionDet은 이미지 생성 태스크와 달리, 박스 표현이 단 4개의 파라미터(중심 좌표 $$c_x, c_y$$, 너비 $$w$$, 높이 $$h$$)만 가지므로, 신호 대 노이즈 비(SNR)를 증가시켜 더 강한 학습 신호를 유지한다. 최적의 신호 스케일 값은 2.0으로 설정된다.[1]

***

### 3. 모델 구조
#### 3.1 아키텍처 구성

모델은 두 개의 주요 모듈로 구성된다:[1]

**이미지 인코더(Image Encoder)**:
- ResNet-50 또는 Swin Transformer 백본 사용
- Feature Pyramid Network(FPN)으로 다중 스케일 특성맵 생성
- 이미지 특성을 한 번만 추출하여 계산 효율성 극대화

**탐지 디코더(Detection Decoder)**:
- 6개의 캐스케이드 스테이지 구성
- 타임스텝 임베딩을 통해 확산 과정의 진행 단계 인코딩
- RoI Align을 사용하여 노이즈 박스로부터 RoI 특성 추출
- 반복적 평가 시 전체 디코더 헤드를 재사용 가능

#### 3.2 학습 프로세스

**알고리즘 1: DiffusionDet 학습**

입력: 이미지 배치 $$B \times H \times W \times 3$$, 지정 박스 $$B \times N \times 4$$

1. 이미지 인코더를 통해 특성 추출: $$\text{feats} = \text{ImageEncoder}(\text{images})$$

2. 지정 박스를 $$N$$개로 패딩(무작위 가우시안 박스 연결)

3. 신호 스케일링 적용: $$pb \leftarrow pb \cdot 2 - 1$$

4. 시간 스텝 $$t$$ 무작위 샘플링

5. 가우시안 노이즈 생성: $$\epsilon \sim \mathcal{N}(0, 1)$$

6. 노이징 박스 계산:
$$pb^{\text{crpt}} = \sqrt{\bar{\alpha}_t} pb + \sqrt{1-\bar{\alpha}_t} \epsilon$$

7. 탐지 디코더로 예측: $$pb^{\text{pred}} = \text{DetectionDecoder}(pb^{\text{crpt}}, \text{feats}, t)$$

8. 집합 예측 손실(Set Prediction Loss) 계산 및 역전파

#### 3.3 추론 프로세스

**알고리즘 2: DiffusionDet 샘플링**

입력: 이미지, 스테이 수 $$S$$, 총 시간 스텝 $$T$$

1. 이미지 특성 추출: $$\text{feats} = \text{ImageEncoder}(\text{images})$$

2. 초기 무작위 박스: $$pb_T \sim \mathcal{N}(0, 1)$$

3. 균등 샘플 스텝 크기 설정

4. 각 스텝에서:
   - 현재 박스에서 예측: $$pb_0 = \text{DetectionDecoder}(pb_t, \text{feats}, t_{\text{now}})$$
   - DDIM을 사용하여 다음 시간 스텝의 박스 추정
   - **박스 갱신(Box Renewal)**: 낮은 신뢰도의 박스를 새로운 무작위 박스로 교체

5. 최종 예측 반환

**박스 갱신 전략**:

박스 갱신은 학습 시 박스가 노이징 프로세스를 통해 생성되지만, 추론에서 원하지 않는 박스들은 임의로 분포하는 문제를 해결한다. 신뢰도 임계값 이하의 박스는 제거하고, 원하는 박스들과 새로운 무작위 박스를 연결한다.[1]

#### 3.4 손실 함수

집합 예측 손실(Set Prediction Loss)을 사용하며, 최적 수송(optimal transport)으로 매칭된 예측과 지정 객체 간의 손실을 계산한다:[1]

$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{L1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{giou}} \mathcal{L}_{\text{giou}}$$

여기서 $$\lambda_{\text{cls}} = 2.0, \lambda_{\text{L1}} = 5.0, \lambda_{\text{giou}} = 2.0$$[1]

***

### 4. 성능 향상 및 실험 결과
#### 4.1 COCO 데이터셋 성능

| Method | Backbone | AP | AP50 | AP75 |
|--------|----------|-----|------|------|
| RetinaNet | ResNet-50 | 38.7 | 58.0 | 41.5 |
| Faster R-CNN | ResNet-50 | 40.2 | 61.0 | 43.8 |
| Sparse R-CNN | ResNet-50 | 45.0 | 63.4 | 48.2 |
| **DiffusionDet (1 step, 300 boxes)** | **ResNet-50** | **45.8** | **64.1** | **50.4** |
| **DiffusionDet (4 steps, 500 boxes)** | **ResNet-50** | **46.8** | **65.3** | **51.8** |

ResNet-50 백본에서 DiffusionDet은 단일 스텝으로 45.8 AP를 달성하여 Sparse R-CNN(45.0 AP)을 능가하고, 4 스텝과 500개 박스로 46.8 AP를 달성한다.[1]

#### 4.2 동적 박스 개수 평가

DiffusionDet의 가장 눈에 띄는 특성 중 하나는 **평가 시 임의의 개수의 박스를 사용할 수 있다**는 점이다:[1]

| Number of Boxes | DETR (AP) | DiffusionDet (AP) |
|-----------------|-----------|-------------------|
| 50 | 31.0 | 38.4 |
| 100 | 34.9 | 38.4 |
| 300 | 38.8 | 45.8 |
| 500 | 36.5 | 46.3 |
| 1000 | 34.0 | 46.7 |
| 2000 | 30.2 | 46.8 |
| 4000 | 26.4 | 46.8 |

DETR은 300개의 쿼리에서 훈련되어 쿼리 개수가 변하면 성능이 저하되지만, DiffusionDet은 박스 개수가 증가해도 지속적으로 성능이 향상되거나 유지된다.[1]

#### 4.3 반복적 평가

| Iteration Steps | 100 boxes (AP) | 300 boxes (AP) | 500 boxes (AP) |
|-----------------|----------------|----------------|----------------|
| 1 | 41.9 | 45.8 | 46.3 |
| 2 | 44.5 | 46.5 | 46.8 |
| 3 | 45.2 | 46.6 | 46.9 |
| 4 | 45.8 | 46.6 | 47.0 |
| 8 | 46.1 | 46.8 | 46.9 |

반복 스텝이 증가함에 따라 성능이 개선되며, 100개 박스의 경우 1단계에서 4단계로 진행하면 4.2 AP 향상을 달성한다.[1]

#### 4.4 제로샷 전이 성능 - 일반화의 핵심

**COCO에서 CrowdHuman으로의 전이**:

| Method | COCO (AP) | CrowdHuman (300 boxes, AP) | CrowdHuman (2000 boxes/4 steps, AP) | Gain |
|--------|-----------|---------------------------|--------------------------------------|------|
| DETR | 61.3 | 61.3 | 61.3 | 0.0 |
| Sparse R-CNN | 66.6 | 66.5 | 66.5 | -0.1 |
| **DiffusionDet** | **66.6** | **69.0** | **71.9** | **5.3** |

이는 DiffusionDet의 가장 중요한 강점이다. CrowdHuman은 COCO보다 혼잡한 장면(평균 22.6명/이미지)을 포함하므로, DiffusionDet은 평가 시 박스 개수와 반복 스텝을 동적으로 조정하여 성능을 크게 향상시킨다.[1]

#### 4.5 LVIS 벤치마크 (장꼬리 분포)

| Method | Backbone | AP | AP_rare | AP_common | AP_freq |
|--------|----------|-----|---------|-----------|---------|
| Sparse R-CNN | ResNet-50 | 29.2 | 20.6 | 27.7 | 34.6 |
| **DiffusionDet (1 step, 1000 boxes)** | **ResNet-50** | **31.4** | **24.5** | **28.8** | **37.3** |
| **DiffusionDet (4 steps, 300 boxes)** | **ResNet-50** | **31.5** | **24.1** | **29.3** | **37.4** |

반복 평가는 COCO에서 0.8 AP 향상을 가져오지만, LVIS에서는 2.1 AP 향상을 달성한다. 이는 더 어려운 벤치마크일수록 DiffusionDet의 이점이 더 크다는 것을 시사한다.[1]

***

### 5. 일반화 성능 향상 분석 (중점)
#### 5.1 일반화 메커니즘

DiffusionDet의 뛰어난 일반화 성능은 다음 요인들에 의해 실현된다:

**1. 확률적 접근 방식**:
확산 모델의 확률적 특성은 다양한 박스 초기화와 점진적 정제를 통해 더 강건한 표현을 학습하게 한다. 고정된 학습 가능한 쿼리와 달리, 무작위 박스에서 시작하므로 훈련-평가 분포 불일치가 감소한다.[1]

**2. 유연한 박스 수 처리**:
동적 박스 개수 평가 메커니즘으로 인해 희소(sparse) 또는 혼잡(crowded) 장면에 자동으로 적응한다. 제로샷 전이에서 COCO(평균 7.7 객체/이미지)에서 CrowdHuman(평균 22.6 객체/이미지)로 전이할 때, 박스 개수를 동적으로 증가시켜 5.3 AP 향상을 달성한다.[1]

**3. 반복적 정제**:
다중 반복 스텝을 통한 점진적 정제는 모델이 예측 오류를 점차 수정하도록 한다. 이는 도메인 시프트 상황에서도 견고하게 작동한다.[1]

**4. 신호 스케일링 최적화**:
신호 스케일 2.0은 박스의 제한된 파라미터(4개)를 고려하여 더 강한 학습 신호를 유지한다. 이는 이미지 생성(신호 스케일 1.0) 또는 세그멘테이션(신호 스케일 0.1)과 다르며, 탐지 태스크에 맞춘 최적화이다.[1]

#### 5.2 통계적 안정성

무작위 박스 초기화로 인한 성능 변동성 분석:[1]
- 5개의 독립적 학습 인스턴스에서 45.7 AP 근처에 밀집된 분포
- 모델 인스턴스 간 성능 차이는 미미
- 10개의 다양한 무작위 시드로 평가해도 신뢰성 있는 결과

이는 DiffusionDet이 무작위 박스 초기화에 견고함을 입증한다.[1]

#### 5.3 도메인 특정 적응

LVIS의 장꼬리 분포에서 DiffusionDet의 우수한 성능:[1]
- 드물게 발생하는 클래스(rare class)에서 AP_rare 24.5 달성 (vs. Sparse R-CNN 20.6)
- 반복적 정제가 장꼬리 분포에 더 효과적
- 단일 모델로 다양한 클래스 분포에 자동 적응

***

### 6. 모델의 한계
#### 6.1 계산 효율성

**추론 속도 트레이드오프**:[1]
- 단일 스텝 (300 박스): 30 FPS - Sparse R-CNN과 유사
- 4 스텝 (300 박스): 20 FPS - 약 33% 속도 저하
- 4 스텝 (1000 박스): 24 FPS - 복잡한 장면에서 실시간성 감소

#### 6.2 기술적 한계

1. **DDIM 및 박스 갱신의 필요성**:
   - DDIM 없이는 반복 스텝이 증가해도 성능 개선 없음
   - 박스 갱신 전략이 추론 복잡도 증가

2. **고급 컴포넌트 부재**:
   - Deformable Attention 등 최신 기술 미적용
   - DINO와 같은 최고 성능 방법과 여전히 성능 차이

3. **신호 스케일 의존성**:
   - 신호 스케일 2.0이 최적값이지만, 다른 도메인에서는 재조정 필요 가능성

#### 6.3 학습 복잡도

- **박스 패딩 전략 필요**: 다양한 길이의 지정 박스 목록을 고정 크기로 패딩
- **최적 수송 할당**: 다대일 매칭으로 계산 복잡도 증가
- **하이퍼파라미터 민감성**: 신호 스케일, 타임스텝 스케줄 등 신중한 튜닝 필요

***

### 7. 최신 연구 기반 향후 전망 및 고려사항
#### 7.1 확산 모델 기반 탐지의 진화 방향

**최신 연구 추세**:[2][3][4][5][6][7][8]

1. **데이터 엔진으로서의 확산 모델**:[2]
   - DiffusionEngine은 확산 모델을 탐지용 합성 데이터 생성 엔진으로 활용
   - COCO에서 3.1% mAP, VOC에서 7.6% mAP 향상
   - **시사점**: DiffusionDet과 결합하면 데이터 부족 시나리오에서 큰 효과

2. **3D 탐지로의 확장**:[4][8][9]
   - 3DifFusionDet: LiDAR-Camera 퓨전의 강건한 확산 기반 3D 탐지
   - DiffRef3D: 포인트 클라우드 기반 3D 탐지에 확산 적용
   - Diff3Det: 무작위 3D 박스로부터 점진적 정제
   - **시사점**: 2D에서 3D로의 자연스러운 확장 가능

3. **특수 탐지 태스크로의 적용**:[3][10][11][12]
   - diffCOD: 위장된 객체 탐지(Camouflaged Object Detection)
   - DiffHOI: 인간-객체 상호작용 탐지
   - DiffusionTrack: 다중 객체 추적(MOT)
   - **시사점**: DiffusionDet 패러다임이 다양한 탐지 변종에 적용 가능

4. **도메인 일반화 강화**:[13][14][15]
   - 최신 논문(2025): "Mining Robust Features from Diffusion Models for Domain-Generalized Detection" - 14.0% mAP 향상[13]
   - 확산 모델의 멀티-스텝 중간 특성을 도메인 불변 표현으로 활용
   - 단일 도메인 일반화(SDG-DiffDet): 메모리 가이드 확산 모듈로 소스-타겟 분포 전이[15]
   - **시사점**: DiffusionDet의 일반화 성능을 더욱 극대화할 가능성

#### 7.2 향후 연구 시 고려할 점

**1. 속도 최적화**:
- Consistency Models나 다른 고속 샘플링 전략 적용[1]
- 적응형 스텝 개수 조정으로 정확도-속도 트레이드오프 최적화
- 고주파 정보 손실을 보완하는 경량화 방법

**2. 아키텍처 개선**:
- Deformable Attention, Wide Detection Head 등 최신 기법 통합
- 대규모 백본(예: Swin-Large)과의 결합
- 마일티-스케일 반복 정제 메커니즘

**3. 데이터 증강 및 합성**:
- DiffusionEngine과 DiffusionDet의 통합: 합성 데이터로 사전학습 후 탐지
- 도메인별 적응형 데이터 생성
- 라벨 부족 시나리오에서의 반자동 라벨링

**4. 이론적 심화**:
- 확산 모델이 객체 탐지 태스크에 왜 더 효과적인지에 대한 이론적 분석
- 신호 스케일과 성능의 관계에 대한 수학적 프레임워크
- 일반화 한계에 대한 엄밀한 분석

**5. 실제 응용 확대**:
- 자율주행, 감시, 의료 영상 등 특정 도메인 최적화
- 저전력 디바이스에서의 경량 버전 개발
- 실시간 성능 요구 시스템에 대한 적응형 스텝 조정

**6. 멀티모달 및 비디오 확장**:
- 텍스트-이미지 조건부 탐지
- 비디오 프레임 간 일관성 유지를 위한 확산 적용
- 시공간 정제 메커니즘

#### 7.3 실무적 함의

1. **제로샷 일반화 성능**: 새로운 도메인에 사전학습 모델을 직접 적용할 때, 기존 방법과 달리 박스 개수와 반복 스텝만 조정하면 추가 재학습 없이 성능 개선 가능[1]

2. **유연한 배포**: 동일한 모델 가중치로 다양한 속도-정확도 트레이드오프를 실현하므로, 단일 모델로 여러 응용에 대응 가능[1]

3. **혼잡 장면 강점**: CrowdHuman의 예시처럼 밀집된 객체 장면에서 성능 향상이 더 크므로, 군중 탐지, 교통 모니터링 등에 특히 유용[1]

***

### 8. 결론
DiffusionDet은 **객체 탐지 문제를 생성 모델 관점에서 재해석**한 혁신적 접근이다. 무작위 박스로부터 점진적 정제를 통해 학습과 평가의 유연성을 달성하고, 특히 **제로샷 도메인 전이에서 탁월한 일반화 성능**을 보여준다.[1]

최근 연구 동향을 보면, 확산 모델 기반 탐지는 단순한 한 가지 방법론을 넘어서 **데이터 생성, 특수 탐지 태스크, 3D 탐지, 도메인 일반화** 등으로 빠르게 확산되고 있다. 특히 2025년 발표된 최신 논문들은 확산 모델의 중간 특성 활용으로 도메인 일반화에서 14% 이상의 성능 향상을 달성하고 있다.[2][3][4][13]

향후 연구는 **계산 효율성 개선**, **최신 아키텍처 컴포넌트 통합**, **이론적 이해 심화**에 초점을 맞추면서, 동시에 **멀티모달 확장**과 **실제 응용 고려**를 통해 DiffusionDet 패러다임을 다양한 분야로 확산시킬 것으로 예상된다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/91a407ea-de7b-4717-82ae-e17d86caf86e/2211.09788v2.pdf)
[2](https://arxiv.org/abs/2309.03893)
[3](https://arxiv.org/abs/2308.00303)
[4](https://arxiv.org/abs/2312.02966)
[5](https://ieeexplore.ieee.org/document/10435420/)
[6](https://arxiv.org/abs/2307.02270)
[7](https://www.mdpi.com/2079-9292/12/24/4962)
[8](https://arxiv.org/abs/2311.03742)
[9](https://arxiv.org/abs/2309.02049)
[10](https://arxiv.org/abs/2305.12252)
[11](https://arxiv.org/abs/2305.17932)
[12](https://arxiv.org/abs/2308.09905)
[13](https://arxiv.org/abs/2503.02101)
[14](https://arxiv.org/abs/2412.13815)
[15](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Diffusion-based_Source-biased_Model_for_Single_Domain_Generalized_Object_Detection_ICCV_2025_paper.pdf)
[16](https://arxiv.org/pdf/2211.09788.pdf)
[17](http://arxiv.org/pdf/2312.11578.pdf)
[18](https://arxiv.org/pdf/2309.03893.pdf)
[19](http://arxiv.org/pdf/2310.16349.pdf)
[20](https://arxiv.org/html/2502.14891)
[21](https://arxiv.org/abs/2303.09813)
[22](https://arxiv.org/html/2408.12747v1)
[23](https://arxiv.org/abs/2211.09788)
[24](https://arxiv.org/html/2509.13214v1)
[25](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_DiffusionDet_Diffusion_Model_for_Object_Detection_ICCV_2023_paper.pdf)
[26](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Graham_Denoising_Diffusion_Models_for_Out-of-Distribution_Detection_CVPRW_2023_paper.pdf)
[27](https://viplab.snu.ac.kr/viplab/courses/mlvu_2023_1/projects/09.pdf)
[28](https://pmc.ncbi.nlm.nih.gov/articles/PMC11601717/)
[29](https://github.com/ShoufaChen/DiffusionDet)

</details>
