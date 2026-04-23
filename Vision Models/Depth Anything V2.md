
# Depth Anything V2

---

## 📌 참고 자료 (출처)

1. **[논문 원문]** Yang et al., "Depth Anything V2," *arXiv:2406.09414*, NeurIPS 2024. — https://arxiv.org/abs/2406.09414
2. **[논문 HTML 전문]** https://arxiv.org/html/2406.09414v1
3. **[공식 프로젝트 페이지]** https://depth-anything-v2.github.io/
4. **[GitHub 공식 저장소]** https://github.com/DepthAnything/Depth-Anything-V2
5. **[Semantic Scholar]** https://www.semanticscholar.org/paper/Depth-Anything-V2
6. **[OpenReview (NeurIPS 2024)]** https://openreview.net/forum?id=cFTi3gLJ1X
7. **[NeurIPS 2024 공식 Proceedings]** https://proceedings.neurips.cc/paper_files/paper/2024/file/26cfdcd8fe6fd75cc53e92963a656c58-Paper-Conference.pdf
8. **[AI Models FYI 분석]** https://www.aimodels.fyi/papers/arxiv/depth-anything-v2
9. **[Emergent Mind 토픽]** https://www.emergentmind.com/topics/depth-anything-v2
10. **[Roboflow Blog]** https://blog.roboflow.com/depth-anything/
11. **[Claru AI Glossary]** https://claru.ai/glossary/depth-anything-v2
12. **[HuggingFace Blog: Fine-Tuning Depth Anything V2]** https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide
13. **[Intel Embodied SDK Docs]** https://eci.intel.com/embodied-sdk-docs/content/developer_tools_tutorials/model_tutorials/model_depthanythingv2.html
14. **[Survey on Monocular Metric Depth Estimation]** arXiv:2501.11841 — https://arxiv.org/pdf/2501.11841
15. **[UniDepth: Universal Monocular Metric Depth Estimation]** CVPR 2024 — https://arxiv.org/html/2403.18913v1
16. **[Video Depth Anything]** arXiv:2501.12375 — https://arxiv.org/html/2501.12375v2
17. **[Benchmark on MDE in Wildlife Setting]** arXiv:2510.04723 — https://arxiv.org/html/2510.04723v1
18. **[Depth Pro 분석 블로그]** https://learnopencv.com/depth-pro-monocular-metric-depth/
19. **[Awesome-Monocular-Depth GitHub]** https://github.com/choyingw/Awesome-Monocular-Depth

---

## 1. 📝 핵심 주장 및 주요 기여 요약

Depth Anything V2는 복잡한 기법을 추구하지 않고, 강력한 단안 깊이 추정(Monocular Depth Estimation, MDE) 모델 구축을 위한 핵심 발견들을 제시한다. V1 대비 세 가지 핵심 실천 방식으로 훨씬 더 정밀하고 강건한 깊이 예측을 달성한다:
1. 라벨링된 실제 이미지를 합성 이미지로 대체
2. Teacher 모델의 용량 확장
3. 대규모 의사 라벨(pseudo-labeled) 실제 이미지를 매개로 Student 모델 학습

Stable Diffusion 기반의 최신 모델들과 비교하여 10배 이상 빠르고 더 정확하며, 25M에서 1.3B 파라미터 규모의 다양한 모델을 제공하여 다양한 응용 시나리오를 지원한다.

또한 현재 테스트 세트의 다양성 부족과 잦은 노이즈 문제를 고려하여, 정밀한 어노테이션과 다양한 씬으로 구성된 종합 평가 벤치마크(DA-2K)를 새롭게 구축하여 향후 연구를 촉진한다.

---

## 2. 🔬 상세 분석

### 2-1. 해결하고자 하는 문제

단안 깊이 추정(MDE)은 단일 2D 이미지에서 깊이 정보를 추론하는 어려운 컴퓨터 비전 과제이다.

V1 모델의 설계를 재검토한 결과, 라벨링된 깊이 정보를 가진 실제 이미지에 크게 의존하면서, 합성 데이터와 실제 이미지 간의 도메인 갭이 중요한 한계로 식별되었다.

실제 이미지의 거친(coarse) 깊이 맵은 세밀한 예측에 해롭고, 더 많은 합성 이미지를 수집하는 방법은 모든 실제 시나리오를 모방하는 그래픽 엔진을 만드는 것이 불가능하기 때문에 지속 불가능하다. 따라서 합성 데이터로 MDE 모델을 구축하기 위한 신뢰할 수 있는 해결책이 절실하다.

구체적으로 논문은 다음 세 가지 핵심 질문을 제기한다:
- **Q1**: MiDaS나 Depth Anything의 거친 깊이가 discriminative 모델링 자체에서 오는가?
- **Q2**: 세밀한 디테일을 위해 heavy diffusion 기반 모델링을 반드시 채택해야 하는가?
- **Q3**: 합성 이미지에서 정밀함과 실제 이미지에서의 강건함을 어떻게 동시에 달성할 수 있는가?

### 2-2. 제안하는 방법 및 수식

#### ① 학습 파이프라인 (Teacher-Student 구조)

모델 아키텍처는 다음 3단계로 구성된다:
1. DINOv2-G 기반의 신뢰할 수 있는 Teacher 모델을 고품질 합성 이미지로만 학습
2. 대규모 라벨 없는 실제 이미지에 대한 정밀 pseudo depth 생성
3. pseudo 라벨링된 실제 이미지로 최종 Student 모델 학습 (강건한 일반화)

595K개의 합성 이미지로 초기 최대 Teacher 모델을 훈련하고, 62M개 이상의 실제 pseudo-labeled 이미지로 최종 Student 모델을 훈련한다.

#### ② 손실 함수 (Loss Functions)

**Scale-and-Shift-Invariant Loss (SSI Loss)**:
서로 다른 데이터셋 간의 스케일과 이동 차이를 무시하는 affine-invariant 손실 함수이다.

$$\mathcal{L}_{ssi} = \frac{1}{HW} \sum_{i=1}^{HW} \rho\left(\hat{d}_i^*, \hat{d}_i\right)$$

여기서 $\rho(\hat{d}_i^\*, \hat{d}_i) = |\hat{d}_i^* - \hat{d}_i|$ 이며, 예측값과 GT를 정규화하기 위한 scale $s(d)$와 translation $t(d)$는:

$$t(d) = \mathrm{median}(d), \quad s(d) = \frac{1}{HW}\sum_{i=1}^{HW}|d_i - t(d)|$$

이를 통해 다중 데이터셋 공동 학습 시 각 샘플의 알 수 없는 스케일과 이동을 무시하는 affine-invariant 학습이 가능하다.

**Gradient Matching Loss (GM Loss)**:
Gradient Matching Loss($\mathcal{L}_{gm}$)는 엣지를 선명하게 하고 객체 경계를 정확하게 보존하는 데 사용된다.

**최종 손실 함수**:

$$\mathcal{L}_{total} = \mathcal{L}_{ssi} + \lambda \cdot \mathcal{L}_{gm}$$

Gradient Matching Loss의 가중치는 affine-invariant 손실 대비 2:1 비율로 세심하게 조정되어, 세밀한 경계와 얇은 구조 재건을 최적화한다.

**Metric Depth를 위한 SiLog Loss**:
Metric depth 학습 시 SiLog 손실 함수를 사용한다. 절대 깊이 학습 시에도 일종의 "스케일 정규화"를 활용하며, 파라미터 $\lambda = 0.5$가 전역 일관성과 지역 정확도 사이의 균형을 조절한다.

**Noisy Pixel Filtering**:
샘플별로 손실이 가장 높은 상위 10% 픽셀을 무시하여 불확실한 teacher pseudo-label의 영향을 완화한다.

#### ③ Knowledge Distillation at Label Level

소규모 모델은 대규모 라벨링되지 않은 실제 이미지를 활용하여 가장 유능한 Teacher 모델의 고품질 예측을 모방하는 방법을 학습하는데, 이는 knowledge distillation과 유사하다. 그러나 기존과 달리 이 distillation은 원본 라벨 데이터의 feature 또는 logit 수준이 아닌, 추가적인 unlabeled 실제 데이터를 통해 label 수준에서 강제된다. 이는 feature 수준 distillation이 특히 teacher-student 간 규모 차이가 클 때 항상 유익하지는 않다는 증거가 있기 때문에 더 안전한 접근 방식이다.

### 2-3. 모델 구조

아키텍처는 특징 추출을 위한 DINOv2 Vision Transformer(ViT)와 깊이 회귀를 위한 Dense Prediction Transformer(DPT) 디코더를 결합한다. 모델 확장성이 내재되어 있어 지연 시간에 민감한 실시간 작업부터 고정밀 오프라인 처리까지 다양한 시나리오를 지원한다.

구체적인 모델 구성:

| 모델 | 인코더 | 파라미터 수 | 특징 채널 | 출력 채널 |
|------|--------|------------|----------|---------|
| ViT-Small | vits | 25M | 64 | [48, 96, 192, 384] |
| ViT-Base | vitb | ~100M | 128 | [96, 192, 384, 768] |
| ViT-Large | vitl | ~300M | 256 | [256, 512, 1024, 1024] |
| ViT-Giant | vitg | ~1.3B | 384 | [1536, 1536, 1536, 1536] |

V1 대비 DINOv2-DPT 아키텍처에 소규모 수정이 있었는데, V1에서는 디코딩에 DINOv2의 마지막 4개 레이어 피처를 사용했으나, V2에서는 중간 피처를 사용한다.

### 2-4. 성능 향상

Depth Anything V2는 DINOv2 백본의 우수한 시각적 표현과 합성 데이터를 활용하여 zero-shot 벤치마크에서 두 모델을 능가한다. NYUv2 벤치마크에서 Depth Anything V2 ViT-L은 delta-1 정확도 0.982를 달성했으며, 이는 ZoeDepth의 0.955와 MiDaS v3.1의 0.918에 비해 크게 향상된 수치이다.

DA-2K 벤치마크에서 ViT-S가 95.3%의 정확도를 달성하여 Stable Diffusion 기반 최상위 모델인 Marigold(86.8%)를 상회한다.

야생 동물 모니터링 환경에서의 벤치마크에서 Depth Anything V2는 평균 절대 오차 0.454m, 상관계수 0.962로 최고 성능을 달성했으며, ZoeDepth의 MAE(3.087m)를 크게 능가한다.

### 2-5. 한계점

현재 62M개의 unlabeled 이미지를 학습에 사용하며 이로 인한 계산 부담이 매우 크다. 향후에는 이러한 대규모 시각 데이터를 더 효율적으로 활용하는 방법을 연구할 계획이다.

또한 현재의 합성 학습 세트가 충분히 다양하지 않다는 한계가 있다. 보다 유능한 Teacher 모델을 위해 더 많은 소스로부터 합성 이미지를 수집하려는 계획이 있다.

논문에서 Depth Anything V2 모델의 잠재적 편향이나 실패 모드를 깊이 다루지 않으며, 가려진 객체, 비정상적인 조명 조건, 또는 비일반적인 카메라 각도를 포함한 다양한 실제 장면에서의 성능을 이해하는 것이 중요하다.

---

## 3. 🌐 일반화 성능 향상 관련 핵심 내용

### 3-1. 합성 데이터의 전략적 활용

Depth Anything V2의 핵심 혁신은 supervised 학습 구성 요소에서 노이즈가 많은 실제 세계 라벨 데이터로부터 고품질 합성 데이터로의 전환이다.

초기 Teacher 학습에는 어떠한 라벨링된 실제 이미지도 사용되지 않고, 합성 데이터셋(BlendedMVS, Hypersim, IRS, TartanAir, vKITTI2; 총 595K 이미지)이 세밀한 구조에도 정확한 레이블을 제공한다.

### 3-2. 합성 데이터별 일반화 기여도

5개의 합성 데이터셋을 Teacher 모델 학습에 사용하며, 각각의 일반화 능력 기여도를 검토했다. 그 결과, 순수 실내 데이터셋인 Hypersim과 IRS가 놀랍게도 가장 많은 일반화 능력을 제공하는 것으로 나타났다.

### 3-3. Pseudo-Label 전략의 일반화 효과

합성 이미지로만 학습한 것과 비교하여, pseudo-labeled 실제 이미지를 통합하면 모델이 크게 향상된다. V1과 달리, Student 모델 학습 시 합성 이미지를 제거하면 소규모 모델(ViT-S, ViT-B)에서 약간 더 좋은 결과로 이어지는 것을 발견하였다. 따라서 최종적으로 Student 모델은 순수하게 pseudo-labeled 이미지로만 학습한다.

### 3-4. 다운스트림 태스크로의 전이 (Semantic Segmentation)

사전 학습된 인코더를 하위 semantic segmentation 태스크에 fine-tune하면, 다양한 스케일의 모델들이 다른 방법들을 크게 능가하는 최고 성능을 달성한다. 이는 해당 모델이 다양한 하위 의미 관련 태스크의 초기화로 활용될 잠재력을 보여준다.

### 3-5. Metric Depth로의 Fine-tuning

강한 일반화 능력을 바탕으로, Metric depth 레이블로 fine-tune하여 Metric depth 모델을 획득한다.

---

## 4. 🔮 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구에 미치는 영향

**① Teacher-Student + Pseudo-Labeling 패러다임의 확산**

Depth Anything V2는 합성 이미지 학습과 pseudo-labeled 실제 이미지를 통해 도메인 갭을 해소하는 설득력 있는 해결책을 제시한다. 이 접근법은 깊이 정확도와 효율성을 모두 높이고, 다양한 시나리오에 걸친 MDE 모델의 적용 가능성을 넓힌다.

**② 비디오 깊이 추정으로의 확장**

Video Depth Anything은 Depth Anything V2 기반으로 개발되어, 일반화 능력, 세부 표현의 풍부함, 계산 효율성을 희생하지 않고 시간적으로 일관된 비디오 깊이 추정이 가능함을 보인다.

**③ 다양한 특수 도메인 적용 가능성**

특수 도메인으로의 파인튜닝된 버전들이 이벤트 기반 깊이 추정, zero-shot 원격 감지 캐노피 높이 매핑, LoRA 기반 적응을 통한 강건한 단안 수술 내비게이션에서 최고 성능을 제공하고 있다.

**④ 범용 Foundation Model로서의 지위**

증강 현실, 로봇공학, 계산 사진학, 자율주행 등 광범위한 응용 분야에 중요한 시사점을 가진다. 단일 이미지에서 정확한 깊이 추정을 가능하게 함으로써 이러한 도메인의 새로운 능력을 열 수 있다.

### 4-2. 향후 연구 시 고려할 점

**① 합성 데이터 다양성 확대**

향후 연구는 합성 데이터의 다양성을 더욱 확장하고 pseudo-labeling 기법을 개선하여 깊이 추정 기술의 한계를 극복하는 데 집중해야 한다.

**② 계산 효율성 문제**

앞으로의 우선순위에는 계산 효율성 향상, 다중 뷰 설정에서의 기하학적 일관성 강화, 도메인 적응 발전이 포함된다.

**③ 실패 모드(Failure Mode) 분석**

모델의 해석 가능성과 설명 가능성을 높이는 추가 연구가 필요하며, 이를 통해 모델이 깊이 예측을 어떻게 수행하고 어디에서 오류가 발생하기 쉬운지에 대한 인사이트를 제공해야 한다. 이는 개발자와 사용자가 모델의 강점과 한계를 더 잘 이해할 수 있도록 도와준다.

**④ 대규모 데이터의 효율적 활용**

현재 62M개의 unlabeled 이미지를 학습에 활용하는데 계산 부담이 매우 크므로, 이러한 대규모 시각 데이터를 보다 효율적으로 활용하는 방법을 연구할 필요가 있다.

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

**MiDaS (Ranftl et al., 2020~)**는 12개의 깊이 데이터셋 혼합으로 훈련된 선구적인 강건한 단안 깊이 모델로서, affine-invariant 손실로 훈련되었으며 zero-shot 전이를 위한 다양한 데이터에 대한 훈련 패러다임("train on everything, test on anything")을 확립하였다.

**ZoeDepth (Bhat et al., 2023)**는 MiDaS를 확장하여 상대 깊이와 메트릭 예측을 결합한 메트릭 깊이 헤드를 추가함으로써 강력한 메트릭 깊이 정확도를 달성하였다.

**UniDepth (Piccinelli et al., 2024)**는 깊이와 카메라 내부 파라미터를 함께 예측하여 크로스 도메인 메트릭 깊이를 더욱 개선하였다.

**Marigold (Ke et al., 2024)**와 같은 생성형 확산 모델들은 고주파 세부 정보와 복잡한 기하학 복구에 강한 잠재력을 보이지만, 확산 방법은 계산 비용이 높다는 단점이 있다.

**ZoeDepth**는 zero-shot MMDE의 돌파구로서, MiDaS를 적응형 메트릭 빈닝과 씬 인식 라우팅으로 확장하여 실내외 데이터에 걸친 강력한 크로스 도메인 일반화를 달성하였다.

아래 표는 주요 연구들을 비교한 것이다:

| 모델 | 연도 | 방법론 | 강점 | 약점 |
|------|------|--------|------|------|
| **MiDaS** | 2020 | Affine-invariant, 다중 데이터셋 | Zero-shot 패러다임 정립 | 세부 표현 부족 |
| **ZoeDepth** | 2023 | 상대+메트릭 결합 | 메트릭 정확도 | 특정 도메인 한계 |
| **Metric3D** | 2023 | 정규 공간 변환 | 카메라 파라미터 활용 | 카메라 정보 의존성 |
| **Marigold** | 2024 | Stable Diffusion 재활용 | 고주파 디테일 | 속도 느림, 계산 부담 |
| **UniDepth** | 2024 | 깊이+카메라 공동 예측 | 범용 메트릭 깊이 | 복잡한 아키텍처 |
| **Depth Anything V2** | 2024 | Teacher-Student + 합성→실제 | 속도·정확도·범용성 균형 | 합성 데이터 다양성 한계 |
| **Depth Pro (Apple)** | 2024 | 멀티스케일 ViT | 고해상도, 날카로운 경계 | 별도의 초점 거리 추정 필요 |

모노큘러 깊이 추정 모델들은 크게 affine-invariant 상대 깊이 모델(DPT, DepthAnything, Marigold)과 절대 스케일로 깊이를 추정하는 메트릭 깊이 모델(ZoeDepth, Metric3D, UniDepth)로 나뉜다. 메트릭 깊이 모델은 카메라 파라미터를 포함한 메트릭 깊이 데이터로 학습해야 하므로 사용 가능한 학습 데이터가 더 제한적이고 결과적으로 일반화 성능이 낮다.

명확한 연구 트렌드는 미지의 도메인에 대한 zero-shot 일반화이다. ZoeDepth와 UniDepth 같은 접근법들은 아키텍처 혁신과 대규모 학습을 통해 유망한 전이 가능성을 보여준다.

---

> **⚠️ 주의**: 본 답변의 일부 수식(특히 손실 함수 세부 계수)은 공개된 논문 HTML 및 관련 기술 블로그에서 수집한 정보를 기반으로 하며, 논문 전문의 부록까지 완전히 검증되지 않은 세부 사항이 포함될 수 있습니다. 가장 정확한 수식 확인을 위해서는 반드시 **arXiv:2406.09414** 원문을 직접 참조하시기 바랍니다.
