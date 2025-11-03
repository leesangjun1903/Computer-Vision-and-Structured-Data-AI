# Large-scale Video Classification with Convolutional Neural Networks

### 1. 핵심 주장과 주요 기여

**핵심 주장**[1]

Karpathy 등이 2014년 발표한 논문의 핵심은 **Convolutional Neural Networks(CNNs)가 이미지 분류에서의 성공을 바탕으로 비디오 분류에 효과적으로 적용될 수 있는가**라는 질문이었습니다. 저자들은 CNNs를 시간 영역으로 확장하여 **시공간(spatio-temporal) 정보를 학습할 수 있다**고 주장했습니다.

**주요 기여**[1]

- **Sports-1M 데이터셋 공개**: 487개 클래스의 YouTube 비디오 100만 개로 구성된 대규모 비디오 분류 벤치마크를 최초로 제공했습니다.
- **시공간 정보 융합 전략**: Early Fusion, Late Fusion, Slow Fusion 등 세 가지 CNN 아키텍처를 제안하여 시간 정보를 신경망에 통합하는 방법을 체계적으로 평가했습니다.
- **다중해상도 아키텍처**: 컨텍스트 스트림과 포비아 스트림의 이중 경로 구조로 2-4배 속도 향상을 달성하면서도 정확도를 유지했습니다.
- **전이 학습 효과 검증**: Sports-1M에서 학습한 특징이 UCF-101 데이터셋으로 전이될 때 63.3%의 성능을 달성하여, 사전학습된 특징의 범용성을 입증했습니다.

---

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

**해결하고자 하는 문제**[1]

당시 비디오 분류는 세 가지 주요 도전을 직면했습니다:

1. **데이터 부족**: 이미지 데이터셋(ImageNet, CIFAR)에 비해 대규모 비디오 벤치마크가 없었습니다.
2. **시간 정보 모델링의 불확실성**: CNN이 이미지에서 성공했으나, 비디오의 시간적 역학을 효과적으로 포착하는 방법이 명확하지 않았습니다.
3. **계산 복잡성**: 비디오 처리는 다중 프레임을 동시에 처리해야 하므로 이미지 처리보다 훨씬 더 비용이 높습니다.

**제안하는 방법: 시공간 정보 융합**[1]

**Single-frame 기준 모델**

기본 구조는 다음과 같습니다:

$$
C(96, 11, 3)-N-P-C(256, 5, 1)-N-P-C(384, 3, 1)-C(384, 3, 1)-C(256, 3, 1)-P-FC(4096)-FC(4096)
$$

여기서 $$C(d, f, s)$$는 $$d$$개의 $$f \times f$$ 필터를 가진 합성곱 계층, $$N$$은 정규화 계층, $$P$$는 풀링 계층, $$FC(n)$$은 $$n$$개의 노드를 가진 완전연결 계층입니다.

**Three Fusion Architectures**[1]

| 방법 | 구현 | 특징 |
|------|------|------|
| **Early Fusion** | 첫 번째 합성곱 필터를 시간 차원으로 확장 (11×11×3×T, T=10) | 픽셀 수준의 모션 감지 가능 |
| **Late Fusion** | 2개의 단일프레임 네트워크를 15프레임 간격으로 배치 후 첫 완전연결 계층에서 병합 | 전역 모션 특성 학습 |
| **Slow Fusion** | 모든 합성곱 계층을 시간 차원으로 확장 (T=4, 2, 1 감소 추세) | 진점적 시간 정보 융합 |

**다중해상도 아키텍처**[1]

계산 효율성을 위해 제안된 아키텍처:

- **입력**: 178×178 프레임
- **컨텍스트 스트림**: 다운샘플된 89×89 저해상도 프레임
- **포비아 스트림**: 중앙 89×89 고해상도 영역
- **통합**: 두 스트림의 활성화를 연결하여 첫 완전연결 계층에 입력

이 구조는 카메라 바이어스(객체가 중앙에 위치하는 경향)를 활용하여 입력 차원을 절반으로 줄일 수 있습니다.

**최적화 방법**[1]

- Downpour Stochastic Gradient Descent (분산 학습)
- 미니배치: 32개
- 모멘텀: 0.9
- 가중치 감쇠: 0.0005
- 초기 학습률: 1e-3 (검증 오류 정체 시 감소)

**성능 결과**[1]

| 모델 | Clip Hit@1 | Video Hit@1 | Video Hit@5 |
|------|-----------|-----------|-----------|
| 특징 기반 베이스라인 | - | 55.3% | - |
| Single-Frame | 41.1% | 59.3% | 77.7% |
| Single-Frame + 다중해상도 | 42.4% | 60.0% | 78.5% |
| Early Fusion | 38.9% | 57.7% | 76.8% |
| Late Fusion | 40.7% | 59.3% | 78.7% |
| **Slow Fusion** | **41.9%** | **60.9%** | **80.2%** |
| CNN 앙상블 (4개 모델 평균) | 41.4% | **63.9%** | **82.4%** |

**한계 분석**[1]

논문에서 발견한 중요한 한계:

1. **모션의 제한적 기여**: Slow Fusion이 Single-frame 모델(59.3%)보다 겨우 1.6%만 개선되었습니다. 이는 **정적 외관 정보가 동작 정보만큼 중요할 수 있음**을 시사합니다.

2. **카메라 모션의 부작용**: 모션 인식 네트워크는 카메라 움직임이 있는 경우 성능 저하가 발생했습니다. 예를 들어, Cricket과 Wrestling에서 단일프레임 모델이 더 우수한 성능을 보였습니다.

3. **라벨 노이즈의 영향**: 자동 주석으로 인한 약한 라벨이 학습에 영향을 미쳤습니다.

---

### 3. 일반화 성능 향상 가능성

**전이 학습 성능**[1]

UCF-101 데이터셋에서의 전이 학습 실험은 매우 유망한 결과를 보여주었습니다:

| 접근 방식 | 3-Fold 정확도 |
|----------|-----------|
| 기존 기준선 (Soomro et al) | 43.9% |
| 특징 기반 접근 (Hand-crafted) | 59.0% |
| 처음부터 학습 | 41.3% |
| 상위 계층 미세조정 | 64.1% |
| **상위 3계층 미세조정** | **65.4%** |
| 모든 계층 미세조정 | 62.2% |

이 결과는 **Slow Fusion 모델**을 사용한 것입니다.

**클래스별 전이 학습 효과**[1]

| 카테고리 | 처음부터 학습 | 상위 3계층 미세조정 | 상위 계층만 미세조정 |
|---------|------------|-----------|-----------|
| Human-Object Interaction | 0.26 | 0.55 | 0.52 |
| Body-Motion Only | 0.32 | 0.57 | 0.52 |
| Human-Human Interaction | 0.40 | 0.68 | 0.65 |
| Playing Musical Instruments | 0.42 | 0.65 | 0.46 |
| Sports | 0.57 | 0.79 | 0.80 |
| **전체 평균** | **0.44** | **0.68** | **0.66** |

**핵심 통찰**[1]

1. **최적 미세조정 전략**: 상위 3계층만 재학습하는 것이 가장 좋은 성능(65.4%)을 제공했습니다.
2. **과적합 위험**: 모든 계층을 재학습(62.2%)하는 것이 상위 3계층만 미세조정하는 것보다 성능이 낮았습니다. 이는 **스포츠 도메인에서 배운 고수준 특징이 과도하게 특화되어 있음**을 나타냅니다.
3. **비스포츠 카테고리 개선**: 상위 계층을 미세조정하면 스포츠 성능은 거의 변하지 않지만(0.80 → 0.79), 비스포츠 카테고리의 평균 정확도는 0.66에서 0.68로 향상되었습니다.

**일반화 메커니즘**[1]

논문의 전이 학습 결과는 다음을 시사합니다:

- **저수준 특징(엣지, 기하학적 패턴)**: 일반적이고 다양한 도메인으로 전이 가능
- **고수준 특징(동작, 스포츠 특화)**: 도메인 특화적이지만, 부분적 미세조정으로 새로운 작업에 적응 가능

***

### 4. 논문의 영향과 미래 연구 방향

**논문의 역사적 영향**[2][3]

이 논문은 비디오 이해 분야의 획기적인 작업이었습니다:

1. **Deep Learning 기반 비디오 분류의 선두주자**: 이전까지 대부분의 연구가 손으로 만든 특징(HOG, SIFT 등)에 의존했지만, 이 논문은 **엔드-투-엔드 딥러닝의 가능성**을 보였습니다.

2. **벤치마크 영향**: Sports-1M 데이터셋은 후속 연구를 촉발했으며, 3D CNN, Two-stream 네트워크, RNN 기반 방법들이 이를 기반으로 개발되었습니다.

3. **인용도**: 약 9,000회 이상 인용되어 컴퓨터 비전 분야의 가장 영향력 있는 논문 중 하나가 되었습니다.

**최근 연구 발전 방향 (2024-2025)**[4][5][6][7][8][9]

| 연구 영역 | 주요 발전 | 문제점 해결 |
|---------|--------|---------|
| **시공간 모델링** | Vision Transformer(ViT), 3D CNN, CNN-LSTM 융합 | 더 정교한 시간 정보 모델링 |
| **도메인 일반화** | Spatial-Temporal Diversification Network (STDN), VideoDG | 카메라 모션, 도메인 시프트 문제 해결 |
| **전이 학습** | 대규모 사전학습(CLIP, 기초 모델) | 범용적 특징 학습 |
| **효율성** | 경량 모델(MobileNetV2), 프루닝 기법 | 실시간 처리 가능성 |

**현재의 주요 과제와 솔루션**[7][10][9]

**1. 도메인 시프트 문제**

최근 연구(VideoDG, STDN)는 **도메인 일반화**를 중점적으로 다루고 있습니다:

- 문제: 스포츠-1M에서 학습한 모델도 다른 데이터셋에서는 성능 저하가 발생
- 원인: 카메라 관점, 시각적 스타일, 환경 조건의 변화
- 솔루션: 다중 도메인 특징 학습, 적대적 학습, 불변 특징 추출

**2. 카메라 모션 처리**

Karpathy 논문이 지적한 "카메라 모션에 약함"은 여전히 과제입니다:

- 최신 접근: Optical flow 기반 방법으로 카메라 움직임과 객체 움직임 분리[11][12]
- 방향: 상대 좌표계에서의 특징 추출

**3. 장기 시간 의존성**

- 문제: Karpathy의 접근은 고정 크기 클립만 처리
- 솔루션: RNN/Transformer 기반 시퀀스 모델링으로 전체 비디오 맥락 이해

**앞으로의 연구 고려사항**[13][10][9]

1. **기초 모델 활용**: CLIP, 대규모 사전학습 모델의 활용으로 범용적 특징 학습

2. **멀티모달 학습**: 오디오, 텍스트와의 결합으로 더 풍부한 맥락 이해

3. **약한 감독 학습 개선**: 자동 주석 데이터 품질 개선으로 노이즈 문제 완화

4. **계산 효율성**: 모바일 및 엣지 기기 배포를 위한 압축 기술

5. **설명 가능성**: 모델이 어떤 시공간 특징을 활용하는지 이해하는 연구

6. **실시간 처리**: 스트리밍 비디오 분류를 위한 온라인 학습 방법

---

### 결론

Karpathy 등의 "Large-scale Video Classification with Convolutional Neural Networks"는 **딥러닝이 비디오 이해에 적용될 수 있음을 실증한 획기적 논문**입니다. 이 논문은 시공간 정보를 학습하는 여러 CNN 아키텍처를 비교하고, Sports-1M이라는 대규모 벤치마크를 제시함으로써 후속 연구를 크게 촉발했습니다.

특히 **전이 학습 효과**(63.3%)는 사전학습된 특징의 범용성을 보여주어, 오늘날 기초 모델 시대의 선행 연구가 되었습니다. 다만 카메라 모션, 약한 라벨 문제 등의 한계는 여전히 현대 연구의 과제로 남아 있으며, 이를 해결하기 위해 도메인 일반화, 멀티모달 학습, 기초 모델 활용 등이 활발히 진행 중입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4434195a-669e-42a2-be52-dfa1e7751fe8/Karpathy_Large-scale_Video_Classification_2014_CVPR_paper.pdf)
[2](https://research.google.com/pubs/archive/42455.pdf)
[3](https://alinlab.kaist.ac.kr/resource/2022_AI602_Lec04.pdf)
[4](https://ieeexplore.ieee.org/document/11171939/)
[5](https://www.researchsquare.com/article/rs-148673/v1)
[6](http://arxiv.org/pdf/2310.17942.pdf)
[7](https://ieeexplore.ieee.org/document/9556560/)
[8](https://openreview.net/pdf?id=YsZTDcIQwQ)
[9](https://arxiv.org/html/2505.24346v1)
[10](http://arxiv.org/pdf/2211.10412.pdf)
[11](https://arxiv.org/html/2411.10501v1)
[12](https://arxiv.org/html/2411.10501v2)
[13](https://arxiv.org/html/2209.01610v3)
[14](https://ieeexplore.ieee.org/document/10418588/)
[15](https://f1000research.com/articles/10-1010/v1)
[16](https://ieeexplore.ieee.org/document/11159037/)
[17](https://ieeexplore.ieee.org/document/11076686/)
[18](https://ieeexplore.ieee.org/document/10937029/)
[19](https://ieeexplore.ieee.org/document/8424735/)
[20](https://link.springer.com/10.1007/s11042-022-12856-6)
[21](https://ieeexplore.ieee.org/document/9761454/)
[22](https://arxiv.org/abs/1504.01561)
[23](https://arxiv.org/pdf/1711.08200.pdf)
[24](https://arxiv.org/pdf/1503.08909.pdf)
[25](http://arxiv.org/pdf/2111.13813.pdf)
[26](https://arxiv.org/pdf/1711.01201.pdf)
[27](https://arxiv.org/pdf/2211.17042.pdf)
[28](https://arxiv.org/pdf/2502.07277.pdf)
[29](https://keras.io/examples/vision/video_classification/)
[30](https://www.ijcai.org/proceedings/2021/0178.pdf)
[31](https://onlinelibrary.wiley.com/doi/10.1155/2021/5865200)
[32](https://cs231n.stanford.edu/slides/2018/cs231n_2018_ds08.pdf)
[33](http://proceedings.mlr.press/v48/fernando16.pdf)
[34](https://arxiv.org/abs/1810.06807)
[35](https://fvl.fudan.edu.cn/_upload/article/files/47/d2/b538f59340d78dbe9908f8b113b3/06ffc8b2-9c9c-4399-8292-72d8edea5dfb.pdf)
[36](https://www.mdpi.com/1424-8220/23/23/9380)
[37](https://arxiv.org/abs/2302.14309)
[38](https://ieeexplore.ieee.org/document/8954791/)
[39](https://link.springer.com/10.1007/s42979-024-03126-3)
[40](https://ieeexplore.ieee.org/document/10849412/)
[41](https://ieeexplore.ieee.org/document/9202063/)
[42](https://link.springer.com/10.1007/s11760-022-02410-0)
[43](http://ieeexplore.ieee.org/document/7779129/)
[44](https://202.138.229.150/index.php/telematika/article/view/730)
[45](https://arxiv.org/pdf/1912.03716.pdf)
[46](https://arxiv.org/html/2405.19525)
[47](https://arxiv.org/pdf/2212.07101.pdf)
[48](https://arxiv.org/html/2312.02021v4)
[49](http://arxiv.org/pdf/2404.18758.pdf)
[50](http://arxiv.org/pdf/2404.00710.pdf)
[51](https://arxiv.org/abs/1901.09819)
[52](https://openreview.net/pdf?id=WVgk0adrw9)
[53](https://openaccess.thecvf.com/content/CVPR2025/papers/Wen_Domain_Generalization_in_CLIP_via_Learning_with_Diverse_Text_Prompts_CVPR_2025_paper.pdf)
[54](https://openaccess.thecvf.com/content/WACV2023/papers/Aich_Cross-Domain_Video_Anomaly_Detection_Without_Target_Domain_Adaptation_WACV_2023_paper.pdf)
[55](https://openaccess.thecvf.com/content/CVPR2025/papers/Nam_Optical-Flow_Guided_Prompt_Optimization_for_Coherent_Video_Generation_CVPR_2025_paper.pdf)
[56](https://thesai.org/Downloads/Volume15No3/Paper_129-Video-based%20Domain%20Generalization%20for%20Abnormal%20Event.pdf)
[57](https://www.sciencedirect.com/science/article/abs/pii/S1047320319300926)
