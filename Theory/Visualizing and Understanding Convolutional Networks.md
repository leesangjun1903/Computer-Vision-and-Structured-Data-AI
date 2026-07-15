# Visualizing and Understanding Convolutional Networks

---

## 1. 핵심 주장과 주요 기여 요약

이 논문(흔히 **ZFNet** 논문으로 불림)은 2012년 AlexNet이 ImageNet 대회에서 압도적 성능을 보였음에도 "왜, 어떻게" 그런 성능이 나오는지 설명할 수 없었던 문제의식에서 출발한다. 이 논문은 대형 컨볼루션 네트워크 모델에서 중간 특징 층의 기능과 분류기의 동작에 대한 통찰을 제공하는 새로운 시각화 기법을 소개하며, 이를 진단 도구로 사용하여 Krizhevsky et al.의 성능을 능가하는 모델 구조를 찾아낸다.

주요 기여는 다음 세 가지로 요약된다.

- **Deconvnet 기반 시각화 기법 제안**: 입력 픽셀 공간으로 특징 활동을 역으로 매핑하여, 특징 맵에서 특정 활성화를 원래 유발한 입력 패턴이 무엇인지 보여주는 새로운 방법을 제시하며, 이 매핑은 Deconvolutional Network(deconvnet)로 수행된다. 이전 연구에서 deconvnet은 비지도 학습 수행 방법으로 제안되었으나, 이 논문에서는 학습 능력 없이 이미 학습된 convnet을 조사하는 프로브(probe)로만 사용된다.
- **아키텍처 개선(ZFNet)**: 시각화를 통해 발견한 문제(1층 필터의 앨리어싱, 죽은 뉴런 등)를 바탕으로 AlexNet의 구조를 개선하여 ImageNet 분류 성능을 향상시켰다.
- **일반화(전이 학습) 실증**: ImageNet으로 학습한 모델이 다른 데이터셋에도 잘 일반화됨을 보였으며, softmax 분류기만 재학습했을 때 Caltech-101과 Caltech-256 데이터셋에서 당시 최신 결과를 확실히 능가했다. DeCAF 논문과 동시에 발표되면서, 전이 학습(transfer learning)의 최초 사례 중 하나를 실증적으로 보여주었다.

---

## 2. 해결 문제, 방법론(수식), 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

2013년 이전까지 CNN의 학습은 사실상 "시행착오(hit-and-trial)" 방식에 의존했다. 2013년 이전 CNN의 학습 메커니즘은 주로 시행착오에 기반했으며, 성능 향상의 정확한 원인을 알지 못했고, 이러한 이해 부족은 복잡한 이미지에 대한 심층 CNN 성능을 제한했다. 또한 첫 번째 층 이후의 CNN을 시각화하는 것은 특징 맵을 픽셀 공간으로 다시 매핑할 수 없기 때문에 어려웠고, 기존 시각화 기법들은 입력 이미지의 어느 부분이 어느 특징 맵에 직접 영향을 주는지 보여주지 못했다.

### 2.2 제안 방법 (수식 포함)

**(1) Deconvnet 구조**

Deconvnet 층은 convnet 층에 부착되며, convnet의 층 아래에서 온 특징의 근사적 재구성 버전을 만든다. 언풀링(unpooling) 연산은 convnet에서 풀링을 수행하는 동안 각 풀링 영역에서 로컬 최댓값의 위치를 기록하는 스위치(switches)를 이용한다.

이를 수식으로 표현하면, 컨볼루션 층의 순전파가 필터 $W_l$, 편향 $b_l$, 비선형함수 ReLU에 대해

$$
h_l = \text{ReLU}(W_l * h_{l-1} + b_l), \qquad p_l = \text{pool}(h_l)
$$

로 주어질 때, deconvnet의 역과정은 세 단계로 구성된다.

1. **Unpooling**: 풀링 과정에서 기록한 switch 위치 $s_l$을 이용해 활성값을 원래 위치에만 복원(나머지는 0)

$$
u_l = \text{unpool}(a_l, s_l)
$$

2. **Rectification**: 재구성된 신호를 다시 ReLU에 통과

$$
r_l = \max(0, u_l)
$$

3. **Filtering(전치 컨볼루션)**: 학습된 필터를 수평·수직으로 뒤집은 전치 필터 $W_l^{T}$를 이용해 역컨볼루션 수행

$$
d_{l-1} = W_l^{T} * r_l
$$

이 과정을 최하위 층까지 반복하면 특정 뉴런의 활성화를 유발한 입력 패턴이 픽셀 공간에 재구성된다. 이 재구성물은 모델에서 샘플링된 것이 아니라, 주어진 특징 맵에서 높은 활성화를 유발하는 검증 세트의 재구성된 패턴이다.

**(2) Occlusion Sensitivity(가림 민감도) 분석**

모델이 실제로 객체 위치를 근거로 분류하는지 검증하기 위해, 입력 이미지의 특정 영역을 회색 패치로 가리면서 정답 클래스에 대한 확률 $p(c \mid x)$의 변화를 측정한다.

$$
S(i,j) = p\big(c \mid x\big) - p\big(c \mid x \odot M_{i,j}\big)
$$

여기서 $M_{i,j}$는 위치 $(i,j)$를 중심으로 한 가림 마스크이다. 이 실험은 convnet이 실제로 원하는 객체를 배경의 다른 패턴이 아니라 이미지 내에서 위치시키고 탐지하는지를 테스트하기 위해 수행된다.

### 2.3 모델 구조

모델은 8층 convnet 구조이다. 224×224 크기로 자른 입력 이미지(3개 색상 평면 포함)가 입력되며, 이는 11×11 크기의 96개의 서로 다른 1층 필터와 컨볼루션된다.(원 AlexNet 기준) ZFNet은 이를 개선하여, 1층 필터 크기를 11×11 대신 7×7로 줄이고, 1층과 2층 모두에서 stride 2 컨볼루션 층을 사용하여 해당 층의 특징에서 더 많은 정보를 보존했다. 최상위 컨볼루션 층은 벡터 형태(6×6×256=9216 차원)로 입력되며, 마지막 층은 클래스 수 C에 대한 C-way softmax 함수이다.

### 2.4 성능 향상

- 이 아키텍처는 2013년 ImageNet 대회에서 우승했으며, 14.8%의 오류율을 달성했다(전년도 15.4% 대비 개선).
- 단일 ZFNet 모델은 top-1, top-5 테스트 오류율 각각 38.4%, 16.5%를 달성하여 AlexNet보다 1.7% 낮은 결과를 보였다.
- 이 아키텍처가 대회에서 직접 우승한 것은 아니지만, 그 추론 방식은 그해 우승자(Zeiler가 설립한 Clarifai)에 의해 구현되었다.
- 아키텍처 변경 근거가 된 실험: 완전연결층(6, 7)을 제거해도 오류율은 약간만 증가했는데, 이는 해당 층들이 모델 파라미터의 대부분을 차지한다는 점에서 예상 밖의 결과였다. 중간 컨볼루션 층 두 개를 제거해도 오류율은 별로 변하지 않았다. 하지만 특정 영역이 아니라 네트워크에 최소한의 깊이를 갖추는 것이 모델 성능에 결정적임을 어블레이션 연구를 통해 보였다.

### 2.5 한계

- Deconvnet 재구성은 근사적(approximate) 산물이며 원인-결과를 엄밀하게 규명하는 정량적 지표가 아니다.
- 시각화 결과, AlexNet의 1·2층에서는 일부 뉴런만 활성화되고 다른 뉴런은 죽어(dead) 있었으며, 2층 특징에는 앨리어싱 아티팩트가 나타났다. 이는 개선의 단서였지만 근본적 원인 분석(왜 그런 현상이 생기는지)에는 한계가 있었다.
- 시각화는 최대 활성화 예제(top-9)에 의존하므로 뉴런의 전체 반응 분포를 대표하지 못할 수 있다.
- 방법론이 max-pooling·ReLU·컨볼루션이 명확히 계층화된 순차적 CNN 구조에 최적화되어 있어, 이후 등장한 skip-connection(ResNet), attention 기반(Transformer) 구조에는 직접 적용이 어렵다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문은 시각화 자체보다 **"일반화 가능성 실증"**이라는 측면에서 이후 연구에 더 큰 영향을 주었다.

**(1) 불변성(invariance) 분석**: 모델 내의 수직 이동, 스케일, 회전 불변성 분석을 수행하였으며, 변환을 겪는 예시 이미지들을 통해 계층별 불변성 정도를 조사했다. 이는 상위 층으로 갈수록 각 특징 맵 내의 강한 그룹화, 상위 층에서의 더 큰 불변성, 이미지의 판별적 부분(예: 개의 눈과 코)의 강조가 나타남을 보여주었으며, 이러한 불변성 획득이 일반화 성능의 근원임을 시사한다.

**(2) 깊이(depth)의 중요성**: 위에서 언급했듯 특정 영역보다 네트워크의 최소 깊이가 모델 성능에 결정적이라는 발견은, 얕은 특징이 아니라 깊은 계층적 특징 추출이 일반화에 핵심적임을 시사하며, 이후 VGG·ResNet 등 "더 깊게(deeper)" 가는 연구 흐름의 실증적 근거가 되었다.

**(3) 전이 학습(Transfer Learning) 실증**: 가장 직접적인 일반화 성능 증거다. Caltech-256에서 60개 학습 이미지/클래스 기준 Bo et al.의 55.2% 대비 74.2%로 상당한 격차의 성능을 보였다. 다만 Caltech-101과 마찬가지로 처음부터(scratch) 학습한 모델은 성능이 나빴으며, 사전학습된 모델을 사용하면 단 6개의 Caltech-256 학습 이미지만으로 10배 많은 이미지를 사용한 기존 최고 방법을 능가했다. 이는 "one-shot/few-shot" 일반화 가능성을 최초로 시사한 실험 중 하나다.

또한 층별로 재학습 범위를 달리하며 선형 SVM을 얹는 실험에서, Caltech-101과 Caltech-256 모두 계층을 올라갈수록 성능이 꾸준히 개선되어, 특징 계층이 깊어질수록 더 강력한 특징을 학습한다는 것을 보였다.

**(4) 국소 구조를 넘어선 문맥(context) 민감도**: 가림(occlusion) 실험을 통해 모델이 이미지 내 국소 구조뿐 아니라 넓은 장면 맥락(broad scene context)에도 민감함을 보였다. 이는 CNN이 단순 패턴 매칭이 아니라 문맥 정보까지 활용하는 표현을 학습함으로써 일반화 능력을 확보할 수 있음을 시사한다.

**(5) 후속 검증**: 이후 연구들은 이 논문의 프로토콜을 그대로 따라 일반화 가능성을 재확인했다. 예를 들어 Caltech-101에서 (Zeiler & Fergus, 2014)의 40±1.7% 결과를 기준으로, 동일한 실험 프로토콜(클래스당 30개 이미지 무작위 선택, 나머지는 테스트)을 따라 여러 후속 연구가 재현·비교를 수행했다.

종합하면, 이 논문은 "CNN이 학습한 특징은 학습 데이터셋에 국한되지 않고 다른 도메인·태스크에도 재사용 가능하다"는 것을 최초로 체계적인 실험을 통해 보여준 논문 중 하나이며, 이는 오늘날 사전학습-미세조정(pretrain-finetune), 나아가 파운데이션 모델 패러다임의 초기 실증적 토대가 되었다.

---

## 4. 후속 연구에 대한 영향과 향후 고려사항

### 4.1 미친 영향

- **설명가능 AI(XAI) 분야의 초석**: Zeiler and Fergus가 제안한 DeConvNets 방법은 역컨볼루션과 언풀링을 통해 CNN의 순전파 과정을 재구성하여 고차원 추상 특징을 픽셀 공간으로 매핑했으며, CNN 특징의 픽셀 수준 시각화를 최초로 달성했다. 이는 CAM(2016), Grad-CAM(2017), LRP 등 후속 해석가능성 연구의 직접적 출발점이 되었다.
- **아키텍처 설계 관행 확립**: 시각화를 통한 "진단 후 개선" 방법론은 이후 신경망 아키텍처 탐색 연구의 방법론적 선례가 되었다.
- **전이학습 패러다임 확산**: DeCAF과 함께 ImageNet 사전학습 특징의 범용성을 입증하여, 컴퓨터 비전 전반에서 "사전학습된 CNN을 특징 추출기로 사용"하는 관행을 정착시켰다.

### 4.2 향후 연구 시 고려할 점

1. **설명의 신뢰성(faithfulness) 검증 필요**: Deconvnet이나 occlusion 기반 시각화가 실제 모델의 의사결정 근거와 일치하는지에 대한 정량적 검증(what you see is what the network gets) 없이 사용될 위험이 있다. 이후 OpenXAI(2022) 같은 벤치마크가 이 문제를 다룬다(§5 참고).
2. **모델 구조 변화에 따른 적용 한계**: ResNet의 skip-connection, Transformer의 self-attention 구조에서는 deconvnet의 switch 기반 언풀링 개념이 그대로 적용되지 않으므로, 구조에 특화된 새로운 해석 기법(예: attention rollout, relevance propagation)이 필요하다.
3. **일반화 평가의 데이터 오염(data leakage) 문제**: 전이 학습 성능 평가 시 Caltech 데이터셋과 ImageNet 간 이미지 중복 가능성을 반드시 검증해야 한다. 실제로 후속 연구에서도 Zeiler & Fergus(2014)가 한 것처럼 정규화된 상관관계를 이용해 중복 이미지를 식별·제거하는 절차를 따랐다.
4. **정성적 시각화에서 정량적 평가로의 전환 필요**: 시각화만으로는 재현성과 객관적 비교가 어렵기 때문에, 최근 연구는 ADCC(Average Drop-Coherence-Complexity) 등 정량 지표를 사용한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

Zeiler & Fergus(2014)의 deconvnet/occlusion 접근은 "**재구성(reconstruction)/섭동(perturbation) 기반**" 해석 방법의 원류로 자리매김했고, 2020년 이후 연구는 크게 세 방향으로 발전했다.

**(1) CAM 계열의 정교화(그래디언트/섭동 기반)**
- Ablation-CAM(WACV 2020)은 그래디언트 없이(gradient-free) 국소화를 수행하는 시각적 설명 기법이다.
- Axiom-based Grad-CAM(BMVC 2020)과 Score-CAM(CVPRW 2020)은 Grad-CAM의 그래디언트 불안정성 문제를 개선했다.
- LayerCAM(IEEE TIP, 2021)은 국소화를 위한 계층적 클래스 활성화 맵을 탐구한다.
- Relevance-CAM(CVPR 2021)은 "모델은 이미 어디를 봐야 하는지 알고 있다"는 관점에서 관련성 기반 설명을 제안한다.

이들은 여전히 Zeiler & Fergus의 "입력 영역을 체계적으로 가려서(masking) 모델 출력의 변화를 직접 측정하는" 섭동 기반 아이디어를 계승하되, 그래디언트 정보를 결합해 계산 효율성과 국소화 정밀도를 높였다는 차이가 있다.

**(2) 트랜스포머 구조로의 확장**

ZFNet 시대의 deconvnet은 CNN 고유의 풀링·컨볼루션 구조에 의존하므로 Vision Transformer(ViT)에는 그대로 적용할 수 없다. 이에 따라:
- Chefer, Gur, Wolf의 "Transformer interpretability beyond attention visualization"(CVPR 2021)은 어텐션 시각화를 넘어선 트랜스포머 해석 기법을 제안했다.
- ViT-ReciproCAM(2023)은 공간 마스크 인코딩 특징과 네트워크 예측 결과 간의 상호적 관계를 활용하는 그래디언트·어텐션 프리(free) XAI 기법이다.
- ICLR 2022 전후로 등장한 여러 ViT 해석 연구들은 CNN의 deconvnet적 접근과 달리 self-attention map을 직접적인 해석 신호로 활용한다.

**(3) 정량적 벤치마크 및 개념 기반(concept-based) 해석으로의 전환**
- OpenXAI(2022)는 모델 설명의 투명한 평가를 지향하는 벤치마크이다.
- B-cos Networks(CVPR 2022)는 정렬(alignment)이 해석가능성에 핵심이라는 관점에서 아예 네트워크 자체를 해석 가능하도록 설계한다.
- Concept Whitening(Nature Machine Intelligence, 2020)은 특징 맵 자체를 사람이 이해 가능한 개념 축에 정렬시키는 방식으로, Zeiler & Fergus식 사후(post-hoc) 시각화에서 "설계 단계의 해석가능성 내재화"로 패러다임이 이동했음을 보여준다.

**비교 요약**: Zeiler & Fergus(2014)는 (i) 단일 뉴런의 최대 활성화 예제를 역투영하는 재구성 기반 시각화와 (ii) 입력 가림을 통한 섭동 기반 민감도 분석이라는 두 축을 제시했다. 2020년 이후 연구는 이 두 축을 계승하면서도, ① 그래디언트·관련성 전파 결합으로 계산 효율과 정밀도 향상, ② CNN 전용 구조(풀링/컨볼루션)에서 벗어나 트랜스포머의 어텐션 구조에 맞춘 새로운 해석 프레임워크 개발, ③ 정성적 시각화에서 ADCC·OpenXAI 등 정량적·재현 가능한 평가 체계로의 전환이라는 세 가지 뚜렷한 방향으로 발전했다.

---

## 참고 문헌 (출처)

1. Zeiler, M.D. & Fergus, R., *Visualizing and Understanding Convolutional Networks*, ECCV 2014 — https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
2. Zeiler, M.D. & Fergus, R., arXiv:1311.2901 — https://arxiv.org/pdf/1311.2901
3. Springer Link, *Visualizing and Understanding Convolutional Networks* — https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53
4. Semantic Scholar 논문 페이지 — https://www.semanticscholar.org/paper/Visualizing-and-Understanding-Convolutional-Zeiler-Fergus/1a2a770d23b4a171fa81de62a78a3deb0588f238
5. Medium, "Paper Summary: Visualizing and Understanding Convolutional Networks" — https://karan3-zoh.medium.com/paper-summary-visualizing-and-understanding-convolutional-networks-aaa4a87a35f9
6. GitHub, saketd403 구현 — https://github.com/saketd403/Visualizing-and-Understanding-Convolutional-neural-networks
7. dev.to, "A Decade of Deep CNN Archs. - ZFNet" — https://dev.to/zohebabai/zfnet-ilsvrc-runner-up-2013-4hnj
8. Pechyonkin, "Key Deep Learning Architectures - ZFNet" — https://pechyonkin.me/architectures/zfnet/
9. Medium (Sik-Ho Tsang), "Review: ZFNet" — https://medium.com/coinmonks/paper-review-of-zfnet-the-winner-of-ilsvlc-2013-image-classification-d1a5a0c45103
10. 구조 조사 논문, arXiv:1901.06032, "A Survey of the Recent Architectures of Deep Convolutional Neural Networks"
11. Achlioptas 외, "Greedy Layerwise Learning Can Scale to ImageNet", arXiv:1812.11446
12. ResearchGate PDF, "Visualizing and Understanding Convolutional Networks" — https://www.researchgate.net/publication/364640176
13. "Basic Level Categorization Facilitates Visual Object Recognition", arXiv:1511.04103
14. ResearchGate 그림, "Transfer learning results on Caltech-101 and Caltech-256" — https://www.researchgate.net/publication/283986647
15. "A Generative Model for Deep Convolutional Learning", arXiv:1504.04054
16. ResearchGate, DeConvNets 관련 RequestPDF — https://www.researchgate.net/publication/258424423
17. arXiv:2405.12175, "Enhancing Explainable AI: Hybrid GradCAM/LRP"
18. arXiv:2503.14640, "Dynamic Accumulated Attention Map for Interpreting Vision Transformer"
19. arXiv:2310.02588, "ViT-ReciproCAM"
20. arXiv:2309.08035, "Interpretability-Aware Vision Transformer"
21. arXiv:2404.02388, "CAPE: CAM as a Probabilistic Ensemble"
22. arXiv:2312.05975, "FM-G-CAM: A Holistic Approach for Explainable AI"
23. arXiv:2509.16745, "CAMBench-QR"
24. NetVLAD, arXiv:1511.07247
25. USPTO 특허 문서, "Structure defect detection using machine learning algorithms"
