
# OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks

## 핵심 주장 및 주요 기여

**OverFeat의 기본 철학**은 단일 합성곱 신경망(ConvNet)으로 분류(Classification), 위치결정(Localization), 탐지(Detection) 세 가지 작업을 통합하여 수행할 수 있다는 것입니다. 논문은 이 세 작업이 점진적으로 증가하는 난이도의 관계를 가지며, 모든 작업이 공유된 특성 추출 기반을 통해 동시에 학습될 수 있음을 보여줍니다.[1]

**주요 기여는 다음과 같습니다:**[1]

첫째, 멀티스케일 슬라이딩 윈도우 접근 방식을 합성곱 신경망 내에 효율적으로 구현했습니다. 이는 전통적인 슬라이딩 윈도우 기반 탐지 방식과 달리, 합성곱의 계산을 공유하여 겹치는 영역의 중복 계산을 피할 수 있습니다.[2][1]

둘째, 경계 상자를 학습하여 예측하는 새로운 딥러닝 위치결정 방식을 제안했습니다. 기존의 비최대 억제(Non-Maximum Suppression, NMS)와 달리, 경계 상자를 누적(Accumulate)하여 탐지 신뢰도를 높였습니다.[1]

셋째, 회귀 네트워크를 통해 각 위치와 스케일에서 객체의 경계 상자 좌표를 예측했습니다. 이는 분류 신경망의 특성 추출 계층을 재사용하면서 최종 회귀 계층만 추가하는 방식으로 구현됩니다.[1]

---

## 문제 정의 및 해결 방식

**기본 문제:** 이미지 분류 데이터셋(ImageNet)은 대체로 객체가 이미지의 중앙에 배치되어 있으나, 실제 탐지 작업에서는 객체의 크기와 위치가 다양하게 변합니다. 따라서 슬라이딩 윈도우 접근 방식이 필요하지만, 컴퓨터 비전에서 객체의 일부만 보이는 상황에서도 분류 정확도가 높을 수 있어서 위치결정과 탐지 정확도는 떨어질 수 있습니다.[1]

**제안된 핵심 아이디어:**[1]

1. **멀티스케일 슬라이딩 윈도우**: 이미지를 다양한 스케일로 처리하여 여러 해상도의 객체를 탐지합니다.

2. **경계 상자 회귀**: 분류 확률뿐만 아니라 각 윈도우에서 객체의 경계 상자 좌표(x1, y1, x2, y2)를 예측합니다.

3. **경계 상자 누적 및 병합**: 여러 스케일과 위치에서 예측된 경계 상자들을 병합하여 최종 탐지 결과를 생성합니다.

---

## 모델 구조 및 핵심 수식

**분류 네트워크 아키텍처:** OverFeat의 기본 구조는 AlexNet을 개선한 형태로, 5개의 합성곱 계층(Layers 1-5)과 3개의 완전 연결 계층(Layers 6-8)으로 구성됩니다.[1]

| 계층 | 단계 | 필터 수 | 필터 크기 | 스트라이드 | 활성화 함수 |
|------|------|---------|-----------|----------|-----------|
| 1 | Conv + Max | 96 | 11×11 | 4×4 | ReLU |
| 2 | Conv + Max | 256 | 5×5 | 1×1 | ReLU |
| 3-5 | Conv (×3) | 512-1024 | 3×3 | 1×1 | ReLU |
| 6-7 | 완전 연결 | 3072-4096 | - | - | ReLU + Dropout |
| 8 | 완전 연결 | 1000 | - | - | Softmax |

**멀티스케일 분류의 핵심 메커니즘:**[3][1]

기본 정제 단계는 총 서브샘플링 비율이 36(4×1×1×1×3×2×3)이 되어 너무 낮은 해상도를 생성합니다. 이를 해결하기 위해 마지막 풀링 작업을 오프셋 $$\Delta x, \Delta y \in \{0, 1, 2\}$$에서 반복 수행합니다:[1]

$$
\text{총 서브샘플링 비율} = 12 \text{ (36에서 개선)}
$$

이를 통해 분류기의 필드 오브 뷰를 객체와 더 잘 정렬할 수 있습니다.[3][1]

**경계 상자 회귀 네트워크:** 회귀 네트워크는 분류 네트워크의 계층 5(Layer 5)로부터의 풀링된 특성 맵을 입력받으며, 다음과 같은 구조를 가집니다:[1]

- 입력: 계층 5 풀링 특성 맵 (5×5 × 256 채널)
- 숨겨진 계층 1: 4096개의 뉴런
- 숨겨진 계층 2: 1024개의 뉴런
- 출력: 4개의 값 (경계 상자 좌표: x1, y1, x2, y2)

**경계 상자 예측의 손실함수:** 회귀 네트워크는 L2 손실 함수를 사용합니다:[1]

$$
L_{\text{reg}} = \|(\hat{x}_i, \hat{y}_i) - (x_i^g, y_i^g)\|_2^2
$$

여기서 $$\hat{x}_i$$는 예측된 경계 상자 좌표이고, $$x_i^g$$는 실제(ground truth) 경계 상자 좌표입니다.[1]

**경계 상자 병합 알고리즘:** 여러 스케일과 위치에서 예측된 경계 상자들을 다음과 같이 병합합니다:[1]

```math
\text{match\_score}(b_1, b_2) = \text{distance}(\text{center}_1, \text{center}_2) + \text{intersection\_area}(b_1, b_2)
```

병합 기준: 

```math
\text{match\_score} < t
```
 인 경우 두 경계 상자를 병합합니다.[1]

***

## 성능 향상 및 실험 결과

**분류 성능:**[3][1]

| 접근 방식 | 톱-1 오류 (%) | 톱-5 오류 (%) |
|-----------|--------------|--------------|
| Krizhevsky et al. (AlexNet) | 40.7 | 18.2 |
| OverFeat - 1개 모델, 4 스케일 | 38.57 | 16.39 |
| OverFeat - 1개 정확 모델, 4 스케일 | 35.74 | 14.18 |
| OverFeat - 7개 정확 모델 앙상블 | 33.96 | 13.24 |

스케일 수의 영향을 명확히 보여주며, 단일 스케일에서 16.97% 오류에서 6개 스케일을 사용하면 16.27% 오류로 개선됩니다.[1]

**위치결정 성능 (ILSVRC 2013):**[1]

- OverFeat은 ILSVRC 2013 위치결정 작업의 우승자로, 29.9% 오류 달성
- 멀티스케일 및 멀티뷰 접근 방식이 결정적 역할: 단일 중앙 자르기(40% 오류) → 2개 스케일(31.5% 오류) → 4개 스케일(30.0% 오류)
- 클래스별 회귀(Per-Class Regression)는 예상과 달리 성능 향상을 보이지 않음 (44.1% vs. 31.3%)[1]

**탐지 성능 (ILSVRC 2013):**[1]

- 경쟁 기간: 19.4% 평균 정확도(mAP)로 3위
- 경쟁 후 개선: 24.3% mAP로 최첨단 달성 (당시 기준)
- R-CNN의 31.4% mAP에 비해 낮지만, 진정한 슬라이딩 윈도우 기반 방식 중 유일한 효과적 구현[4]

***

## 일반화 성능과 그 한계

**일반화 성능 향상의 메커니즘:**[5][1]

OverFeat의 일반화 성능 향상은 여러 요소에서 비롯됩니다:

1. **특성 공유 기반 구축**: 분류, 위치결정, 탐지 작업이 공유된 특성 추출 계층(1-5)을 사용하므로, 분류 작업의 학습 신호가 위치결정 작업을 정규화합니다.

2. **멀티태스크 학습의 장점**: 최신 연구에 따르면, 공유 기반을 통한 멀티태스크 학습은 작업 간 관련성이 있을 때 일반화 성능을 향상시킵니다. 특히 낮은 계층의 특성(엣지, 텍스처)은 모든 작업에 유용합니다.[6][5]

3. **멀티스케일 처리**: 6개 스케일에서 처리하면 다양한 객체 크기에 대한 견고성을 제공합니다.

**하지만 일반화 성능의 한계도 명확합니다:**[1]

1. **분리된 학습**: 분류와 회귀 네트워크가 순차적으로 학습되며, 회귀 네트워크는 고정된 특성 추출 계층을 사용합니다. 논문에서 "분류된 네트워크 전체를 역전파하지 않고 있다"고 명시했습니다.[1]

2. **L2 손실 함수의 한계**: 경계 상자 회귀에 L2 손실을 사용하지만, 평가 지표인 교집합 대비 합집합(IoU) 기준을 직접 최적화하지 않습니다.[1]

3. **경계 상자 매개변수화의 문제**: 예측되는 좌표가 상호 연관되어 네트워크 학습을 어렵게 합니다.[1]

4. **배경 클래스 학습 문제**: 탐지 작업에서 배경 샘플을 명시적으로 학습해야 하며, 전통적인 부트스트래핑 방식의 복잡성이 있습니다.[1]

---

## 연구에 미치는 영향

**OverFeat의 역사적 중요성:**[2][4]

OverFeat은 AlexNet 이후 ImageNet 대회에서 CNN으로 객체 탐지에 우수한 성과를 거둔 첫 사례입니다. 이는 후속 연구에 기초가 되었습니다.[2]

**초기 원스테이지 탐지 아키텍처의 선구자:** OverFeat은 R-CNN(두 단계 탐지)과 달리 밀집된 슬라이딩 윈도우 방식을 사용한 초기 원스테이지(One-stage) 탐지 방식으로, YOLO, SSD 등의 이후 방법들에 영향을 미쳤습니다.[2]

**최신 연구 관점에서의 영향:**[7][8]

더 빠른 R-CNN(Faster R-CNN, 2015)은 OverFeat의 아이디어를 기반으로 발전했습니다. Faster R-CNN의 영역 제안 네트워크(RPN)는 OverFeat의 경계 상자 회귀와 경계 상자 예측 메커니즘을 계승했으며, 수동 영역 제안 알고리즘(Selective Search)을 완전히 제거함으로써 진정한 엔드-투-엔드 학습을 구현했습니다.[8][7]

---

## 현대적 관점에서의 한계 및 고려사항

**구조적 한계:**[5][1]

1. **분리된 학습의 문제**: 분류와 회귀의 분리 학습은 최적화되지 않은 특성 표현을 초래합니다. 현대의 엔드-투-엔드 학습 방식이 더 효과적임이 증명되었습니다.[9][10]

2. **멀티태스크 학습에서의 음의 전이**: 작업 간의 부적절한 특성 공유는 개별 작업의 성능을 저해할 수 있습니다. OverFeat에서 클래스별 회귀가 일반 회귀보다 성능이 나쁜 것이 이를 시사합니다.[11][1]

3. **앙커 박스 전략의 부재**: 현대 방법들(Faster R-CNN, YOLO)의 앙커 박스와 달리, OverFeat은 정규화된 스케일 비율만 사용합니다.[1]

**최신 손실 함수와의 비교:**[12]

최근 연구에서는 경계 상자 회귀에 불확실성을 도입한 KL 발산 손실이 제안되었습니다:[12]

$$
L_{\text{reg}} = \frac{(x_g - x_e)^2}{2\sigma^2} + \frac{\log(\sigma^2)}{2}
$$

이는 위치 추정의 신뢰도를 동시에 모델링하여, 단순 L2 손실보다 우월합니다.[12]

**현대 객체 탐지 파이프라인과의 비교:**

현재의 최첨단 방법들(YOLO v8, Faster R-CNN 변형들)은:[13][14][15]

- 가중 교차 엔트로피 손실을 통한 포그라운드-배경 불균형 해결
- 다중 스케일 특성 피라미드를 통한 멀티스케일 표현
- 고급 비최대 억제(Soft-NMS) 또는 최적 수송 할당(OTA)
- 더 정교한 회귀 매개변수화

등을 통해 OverFeat의 기본 개념을 크게 발전시켰습니다.[15][13]

***

## 향후 연구 시 고려사항

**아키텍처 설계:**

1. **진정한 엔드-투-엔드 학습**: 분류와 회귀를 통합된 손실 함수로 동시에 최적화하고, 백프로퍼게이션이 전체 네트워크를 통과하도록 설계합니다.[1]

2. **작업 간 가중치 조정**: 멀티태스크 학습에서 음의 전이를 피하기 위해 작업별 손실에 적응적 가중치를 부여합니다.[9][6]

3. **특성 공유 구조의 최적화**: 모든 작업에 동등하게 공유할 계층과 작업별 분화된 계층을 식별하기 위해 데이터 중심 접근 방식을 사용합니다.[6]

**손실 함수 및 훈련 전략:**

1. **IoU 기반 손실**: L2 손실 대신 미분 가능한 IoU 손실(GIoU, DIoU, CIoU 등)을 사용합니다.[1]

2. **불확실성 모델링**: 경계 상자 좌표의 불확실성을 추정하여 탐지 신뢰도를 향상시킵니다.[12]

3. **클래스 불균형 처리**: 포컬 손실(Focal Loss)이나 가중 교차 엔트로피를 통해 배경 샘플 주도를 방지합니다.[13]

**일반화 성능 개선:**

1. **정규화 기법**: 드롭아웃, 배치 정규화, 조기 종료 등을 통한 과적합 방지.[1]

2. **데이터 증대**: 회전, 왜곡, 밝기 조정 등의 고급 증대 전략을 통해 모델의 견고성을 향상시킵니다.

3. **영역 제안 최적화**: Faster R-CNN의 RPN 개념을 통합하여 계산 효율성과 탐지 품질의 균형을 맞춥니다.[7]

---

## 결론

OverFeat은 단일 CNN 네트워크로 분류, 위치결정, 탐지를 통합하는 획기적인 프레임워크를 제시했으며, 특히 멀티스케일 처리와 경계 상자 회귀의 누적 방식은 현대의 객체 탐지 방법론의 기초가 되었습니다. 그러나 분리된 학습, L2 손실 함수, 그리고 진정한 엔드-투-엔드 최적화의 부재는 이후 Faster R-CNN과 같은 방법들에 의해 극복되었습니다. 

향후 연구에서는 OverFeat의 기본 개념을 유지하면서 최신의 손실 함수, 멀티태스크 최적화 기법, 그리고 주의 메커니즘을 통합하여 일반화 성능을 극대화해야 할합하여 일반화 성능을 극대화해야 할 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/54a0d694-97be-4e1f-b438-4f0fa122f085/1312.6229v4.pdf)
[2](https://wikidocs.net/167509)
[3](http://arxiv.org/pdf/2409.16073.pdf)
[4](http://fcv2011.ulsan.ac.kr/files/announcement/513/rcnn_pami.pdf)
[5](https://aclanthology.org/2025.coling-main.200.pdf)
[6](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lu_Fully-Adaptive_Feature_Sharing_CVPR_2017_paper.pdf)
[7](https://arxiv.org/abs/1506.01497)
[8](https://deep-math.tistory.com/26)
[9](https://openaccess.thecvf.com/content/ICCV2025/papers/Phan_Beyond_Losses_Reweighting_Empowering_Multi-Task_Learning_via_the_Generalization_Perspective_ICCV_2025_paper.pdf)
[10](https://www.openu.ac.il/Lists/MediaServer_Documents/Academic/CS/Guy%20Golan%20-%20End-to-end%20object%20detection%202.pdf)
[11](https://www.geeksforgeeks.org/deep-learning/introduction-to-multi-task-learningmtl-for-deep-learning/)
[12](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bounding_Box_Regression_With_Uncertainty_for_Accurate_Object_Detection_CVPR_2019_paper.pdf)
[13](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12456/2659662/An-object-classification-and-detection-method-with-faster-R-CNN/10.1117/12.2659662.full)
[14](https://arxiv.org/abs/2507.18967)
[15](https://www.mdpi.com/1424-8220/20/19/5490/pdf)
[16](https://www.semanticscholar.org/paper/9025f4fac1a007bbd60e95bc77001c86d3310634)
[17](http://arxiv.org/pdf/1312.6229v2.pdf)
[18](https://arxiv.org/pdf/2304.08876.pdf)
[19](http://arxiv.org/pdf/2410.05869.pdf)
[20](https://arxiv.org/pdf/2112.02814.pdf)
[21](http://arxiv.org/pdf/2103.05137.pdf)
[22](https://arxiv.org/pdf/1905.05055.pdf)
[23](https://arxiv.org/pdf/2104.06401.pdf)
[24](https://arxiv.org/abs/1312.6229)
[25](https://fiveable.me/images-as-data/unit-9/bounding-box-regression/study-guide/fOsrufrtx1skSwHh)
[26](https://academic.oup.com/nsr/article/5/1/30/4101432)
[27](https://journal.hep.com.cn/foe/EN/10.1007/s12200-019-0853-1)
[28](https://skyil.tistory.com/195)
[29](https://ijircst.org/view_abstract.php?title=A-Comparative-Analysis-of-CNN,-RCNN-&-Faster-RCNN-Object-Detection-Algorithm-for-CAPTCHA-Breaking&year=2024&vol=12&primary=QVJULTEyMzA=)
[30](https://link.springer.com/10.1007/s00371-023-02789-y)
[31](https://ieeexplore.ieee.org/document/10525276/)
[32](https://www.pnrjournal.com/index.php/home/article/view/1386/1157)
[33](https://link.springer.com/10.1007/s12204-023-2667-y)
[34](https://journals.sagepub.com/doi/full/10.3233/JIFS-212740)
[35](https://ieeexplore.ieee.org/document/9847725/)
[36](http://ieeexplore.ieee.org/document/7986913/)
[37](https://arxiv.org/pdf/1504.08083.pdf)
[38](https://pmc.ncbi.nlm.nih.gov/articles/PMC7582940/)
[39](https://downloads.hindawi.com/journals/mpe/2019/3808064.pdf)
[40](https://www.mdpi.com/1424-8220/24/11/3529)
[41](http://arxiv.org/pdf/1604.08893.pdf)
[42](https://pmc.ncbi.nlm.nih.gov/articles/PMC11355408/)
[43](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175249/)
[44](https://arxiv.org/pdf/2206.03064.pdf)
[45](https://proceedings.mlr.press/v195/bairaktari23a/bairaktari23a.pdf)
[46](https://openaccess.thecvf.com/content/WACV2023/papers/Sui_A_Simple_and_Efficient_Pipeline_To_Build_an_End-to-End_Spatial-Temporal_WACV_2023_paper.pdf)
[47](https://proceedings.neurips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)
[48](https://herbwood.tistory.com/10)
