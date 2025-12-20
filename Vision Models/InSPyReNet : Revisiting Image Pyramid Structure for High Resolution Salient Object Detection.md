
# InSPyReNet : Revisiting Image Pyramid Structure for High Resolution Salient Object Detection

## 1. 논문 핵심 요약

**"Revisiting Image Pyramid Structure for High Resolution Salient Object Detection"**은 Inverse Saliency Pyramid Reconstruction Network(InSPyReNet)를 제안하여, 저해상도(LR) 이미지로만 학습하면서도 고해상도(HR) 이미지에서 최고 수준의 사리언트 객체 감지(SOD) 성능을 달성한다. 이 논문의 핵심 기여는 **엄격한 이미지 피라미드 구조 설계**, **스케일 불변 어텐션 모듈(SICA)**, **효과적 수용장(ERF) 불일치 해결**이다.[1]

## 2. 해결하고자 하는 문제

### 2.1 주요 문제점

**고해상도 이미지의 어노테이션 비용 문제**: HR 이미지와 픽셀 수준 주석은 LR 이미지에 비해 훨씬 더 많은 노동력과 시간이 필요하다.[1]

**기존 방법들의 한계**:
1. **단순 크기 조정 방식**: LR 방법으로 학습한 모델에 HR 입력을 하면 고주파 디테일이 손실된다.[1]
2. **복잡한 아키텍처**: 기존 HR SOD 방법들은 여러 학습 단계와 복잡한 구조를 필요로 한다.[1]
3. **이미지 피라미드 구조의 부정확성**: 이전 연구들이 이미지 피라미드를 예측했지만 실제 구조를 엄격히 따르지 않아 블렌딩 기법을 적용할 수 없었다.[1]

**효과적 수용장(ERF) 불일치**: 같은 픽셀이라도 원본 이미지와 크기 조정된 이미지에서 다른 수용장을 가지므로, 네트워크는 다른 문맥 정보를 바탕으로 예측한다.[1]

## 3. 제안 방법: InSPyReNet

### 3.1 전체 구조

InSPyReNet의 아키텍처는 다음 네 가지 핵심 요소로 구성된다:

**1) Backbone 네트워크**: Res2Net50 또는 Swin Transformer를 사용하여 다중 스케일 특징맵을 추출한다.[1]

**2) 이미지 피라미드 기반 사리언시 맵 예측**: 네트워크는 Stage-3(최소 크기)부터 시작하여 라플라시안 사리언시 맵을 각 스테이지에서 예측한다.[1]

**3) 스케일 불변 문맥 어텐션(SICA)**: 다양한 입력 크기에 강건하게 작동하는 어텐션 모듈이다.[1]

**4) 피라미드 블렌딩**: LR과 HR 피라미드를 합성하여 고품질 HR 예측을 생성한다.[1]

### 3.2 핵심 수식

**EXPAND 연산(상향 샘플링)**:
$$S_j^e(x, y) = 4 \sum_{m=-3}^{3} \sum_{n=-3}^{3} g(m, n) \cdot S_{j+1}\left(\frac{x-m}{2}, \frac{y-n}{2}\right)$$

여기서 $g(m,n)$은 표준편차 1인 커널 크기 7의 가우스 필터이고, $S_{j+1}$은 상위 스테이지의 사리언시 맵이다.[1]

**사리언시 맵 재구성**:
$$S_j = S_j^e + U_j$$

여기서 $S_j$는 j 스테이지의 최종 사리언시 맵이고, $U_j$는 라플라시안 사리언시 맵(고주파 디테일)이다.[1]

**REDUCE 연산(하향 샘플링)**:
$$G_j(x, y) = \sum_{m=-3}^{3} \sum_{n=-3}^{3} g(m, n) \cdot G_{j-1}(2x+m, 2y+n)$$

이 연산으로 각 스테이지에 맞춤형 Ground Truth를 생성한다.[1]

**종합 손실 함수**:
$$L(S, G) = \sum_{j=0}^{3} \lambda_j L_{wbce}(S_j, G_j) + \eta \sum_{j=0}^{2} \lambda_j L_{pc}(S_j, \tilde{S}_j)$$

여기서:
- $L_{wbce}$는 픽셀 위치 가중 이진 교차 엔트로피 손실[1]
- $L_{pc}$는 피라미드 일관성 손실:

$$L_{pc}(S_j, \tilde{S}_j) = \sum_{(x,y) \in I_j} ||S_j(x,y) - \tilde{S}_j(x,y)||_1$$

- $\eta = 10^{-4}$, $\lambda_j = 4^j$ (스테이지별 손실 균형)[1]

### 3.3 SICA(Scale Invariant Context Attention) 모듈

**문제**: 기존 어텐션 기반 디코더는 훈련 시 고정 크기 입력으로 학습되므로, 훈련보다 큰 입력에서 문제가 발생한다. 비로컬 연산의 결과 크기가 입력 이미지의 공간 차원에 따라 달라지기 때문이다.[1]

**해결책**: 입력 특징맵 $x$와 문맥맵 $c$를 훈련 시 크기(h, w)로 리사이즈한다:

$$x' \in \mathbb{R}^{\frac{h}{s} \times \frac{w}{s} \times C}, \quad c' \in \mathbb{R}^{\frac{h}{s} \times \frac{w}{s} \times N}$$

**Object Region 표현**:
$$f_k = \sum_{(x,y) \in I} c_k(x,y) x(x,y)$$

**어텐션 점수**:
$$w_k(x,y) = \frac{\exp(T_x(x(x,y))^T T_f(f_k))}{\sum_{l=1}^K \exp(T_x(x(x,y))^T T_f(f_l))}$$

**문맥 강화 특징**:
$$y(x,y) = T_y\left(\sum_{l=1}^K w_l(x,y) T_f'(f_l)\right)$$

이를 통해 HR 이미지에서도 훈련 시점의 스케일 특성을 유지하면서 정확한 라플라시안 사리언시 맵을 예측할 수 있다.[1]

### 3.4 피라미드 블렌딩 기법

**핵심 아이디어**: LR 피라미드(4 스테이지: Stage 3~0)와 HR 피라미드(3 스테이지: Stage 2~0)를 조합하여 7 스테이지 피라미드를 구성한다.[1]

**절차**:
1. LR 이미지로부터 4 스테이지 피라미드 생성
2. HR 이미지로부터 3 스테이지 피라미드 생성
3. LR 피라미드의 Stage 0부터 시작
4. HR 피라미드의 라플라시안 맵을 전이 영역(Transition Area)으로 필터링하여 추가
5. 최종 Stage 0에서 고품질 HR 예측 획득[1]

**전이 영역 계산**:
$$\text{Transition Area} = \text{Dilation}(S) - \text{Erosion}(S)$$

이를 통해 경계 부근의 고주파 디테일만 HR 피라미드에서 추출하고, 잘못된 탐지를 필터링한다.[1]

### 3.5 감독 전략

**각 스테이지별 맞춤형 Ground Truth**: Stage-3에서 생성된 초기 사리언시 맵은 물리적으로 Stage-2보다 더 많은 디테일을 가질 수 없으므로, REDUCE 연산으로 각 스테이지에 적절한 GT를 제공한다.[1]

**Stop-Gradient 전략**: 상위 스테이지의 디코더로 흐르는 그래디언트를 차단하여, 각 스테이지가 독립적으로 자신의 스케일에 집중하도록 강제한다.[1]

## 4. 성능 향상 결과

### 4.1 저해상도(LR) 벤치마크 성능

**DUTS-TE (Standard 384×384)**:
- Res2Net50 백본: $S_\alpha = 0.904$, $F_{\max} = 0.892$, $MAE = 0.035$
- Swin Transformer 백본: $S_\alpha = 0.931$, $F_{\max} = 0.927$, $MAE = 0.024$[1]

이는 PA-KRN(이전 SOTA)의 $S_\alpha = 0.898$보다 3.3% 향상된 결과이다.[1]

**DUT-OMRON**: $S_\alpha = 0.875$, $F_{\max} = 0.832$[1]

**다중 데이터셋 성능**:
| 데이터셋 | $S_\alpha$ | $F_{\max}$ | $MAE$ |
|---------|----------|----------|-------|
| DUTS-TE | 0.931 | 0.927 | 0.024 |
| DUT-OMRON | 0.875 | 0.832 | 0.045 |
| ECSSD | 0.949 | 0.960 | 0.023 |
| HKU-IS | 0.944 | 0.955 | 0.021 |
| PASCAL-S | 0.893 | 0.893 | 0.048 |

### 4.2 고해상도(HR) 벤치마크 성능

**DAVIS-S** (HD 해상도, 평균 728×576):
- $S_\alpha = 0.962$, $F_{\max} = 0.959$, $MAE = 0.009$, **mBA = 0.743**[1]

**HRSOD-TE** (고해상도, 평균 1073×848):
- $S_\alpha = 0.952$, $F_{\max} = 0.949$, $MAE = 0.016$, **mBA = 0.738**[1]

**UHRSD-TE** (초고해상도, 평균 2389×1699):
- $S_\alpha = 0.932$, $F_{\max} = 0.938$, $MAE = 0.029$, **mBA = 0.741**[1]

### 4.3 기존 방법과의 비교

**PGNet(2022) vs InSPyReNet**:
- PGNet은 HR 데이터(H, U)로 학습했을 때 DAVIS-S mBA 0.730을 달성
- InSPyReNet은 LR 데이터(D)만으로 학습하여 DAVIS-S mBA 0.743 달성[1]

**중요한 발견**: PGNet은 HR 벤치마크에서 우수하지만 LR 벤치마크(DUTS-TE $S_\alpha = 0.911$)에서는 성능 저하. 반면 InSPyReNet은 LR과 HR 모두에서 강건함.[1]

### 4.4 경계 정확도(mBA) 분석

InSPyReNet의 특장점은 경계 정확도에 있다:
- DAVIS-S: 0.743 (vs Zeng et al. 0.618, Tang et al. 0.716)
- 이는 라플라시안 피라미드 구조가 경계 정보를 효과적으로 보존하기 때문[1]

## 5. 일반화 성능 향상 가능성

### 5.1 강점: 우수한 일반화 능력

**LR과 HR의 동시 우수 성능**: 대부분의 HR 메서드는 HR에서 좋은 성능을 보이지만 LR에서 떨어지거나 그 반대이다. 그러나 InSPyReNet은:[1]
- LR에서 $S_\alpha = 0.931$ (SOTA 수준)
- HR에서 $S_\alpha = 0.962$ (SOTA 수준)

이는 **scale-agnostic design**의 성공을 의미한다.[1]

**HR 데이터셋 미필요**: 논문의 실험(보충 자료 B.3)에 따르면, InSPyReNet을 HR 데이터셋으로 추가 학습하면 성능이 더욱 향상되지만, HR 데이터 없이도 최고 성능을 달성한다. 이는:[1]
- 어노테이션 비용 대폭 감소
- 데이터 희귀 도메인에서의 적용 가능성

**이미지 피라미드 구조의 보편성**: 라플라시안 피라미드는 $60+$ 년 역사의 수학적으로 검증된 방법이므로, HR 이미지뿐 아니라 다양한 해상도에 일반화 가능하다.[1]

### 5.2 약점: 제약 조건

**CNN Backbone의 한계**: Res2Net50을 사용한 HR 예측은 많은 아티팩트를 생성한다. 이유:[1]
- CNN의 ERF 크기가 훈련 데이터셋의 특성에 의존
- 고정된 수용장으로 매우 큰 입력에 적응 불가[1]

따라서 HR 예측에는 Vision Transformer(Swin)만 사용 권장.[1]

**메모리 제약**: 현재 구현은 $L=1280$(이미지의 최단변)으로 제한되어 4K 이미지 처리 불가.[1]

**Backbone 선택 의존성**: Swin Transformer는 매우 효과적이지만, 일반화 관점에서 다른 Vision Transformer(DeiT, T2T-ViT)와의 비교가 부족하다.[1]

### 5.3 도메인별 일반화 가능성

**의료 이미지**: 보충 자료 B.4에서 DIS5K(이분할 분할) 데이터셋에 적용하면 IS-Net(SOTA)을 상당히 초과한다.[1]

| 데이터셋 | IS-Net | InSPyReNet (LR 학습) | 개선 |
|---------|--------|------------------|------|
| DIS-VD | 0.813 | 0.887 | +7.4% |
| DIS-TE1 | 0.787 | 0.862 | +7.5% |

이는 방법이 도메인에 특화되지 않았음을 시사한다.[1]

**주의사항**: 현재 HR 일반화 연구는 자연 이미지 도메인에 집중되어 있으며, 의료, 위성, 산업 영상 등 다양한 도메인에서의 검증 필요.[1]

## 6. 한계와 실패 케이스

### 6.1 두 가지 실패 패턴

**1) Global Context Failure**: LR 브랜치가 객체를 감지하지 못하면, HR 블렌딩도 실패한다. 이는 방법의 근본적인 한계로, HR 피라미드만으로는 전역 문맥을 복구할 수 없다.[1]

**2) Local Detail Failure**: LR 브랜치는 객체를 정확히 감지하지만, HR 브랜치가 고주파 디테일을 생성하지 못하는 경우. 예: 자전거 바퀴의 스포크, 바구니 디테일 등.[1]

### 6.2 기술적 한계

**HR 예측 한계**: 현재 L=1280 제약으로 매우 큰 이미지 처리 불가. 메모리 효율적 알고리즘 개발 필요.[1]

**Pyramid Blending 적용 조건**: 입력 최단변 < 512이면 적용 안 함.[1]

## 7. 최신 관련 연구 비교 분석 (2020년 이후)

### 7.1 핵심 HR SOD 방법 진화

| 연도 | 저자 | 방법 | HR 데이터 | 복잡도 | 성능 |
|------|------|------|---------|-------|------|
| 2019 | Zeng et al. | GSN+LRN+GLFN | ✓ | 높음 | DAVIS-S $S_\alpha$=0.876 |
| 2021 | Tang et al. | LRSCN+HRRN | 동적 | 중간 | DAVIS-S $S_\alpha$=0.920 |
| 2022 | Xie et al. (PGNet) | 단일 네트워크 | ✓ | 중간 | DAVIS-S $S_\alpha$=0.947 |
| **2022** | **Kim et al. (InSPyReNet)** | **피라미드 블렌딩** | **✗** | **낮음** | **DAVIS-S $S_\alpha$=0.962** |

**InSPyReNet의 우월성**:
1. **HR 데이터 미필요**: 유일하게 LR만으로 SOTA 달성[1]
2. **아키텍처 단순성**: 3개 브랜치(Zeng), 2개 네트워크(Tang) vs 1개 네트워크[1]
3. **LR 성능 유지**: PGNet(DUTS-TE $S_\alpha$=0.911)과 달리 0.931 달성[1]

### 7.2 Vision Transformer 도입 영향 (2021-2022)

**2021 VST**: Vision Saliency Transformer가 제안되어 SOD에서 ViT 효과성 입증[1]

**2021 Swin Transformer**: InSPyReNet이 채택한 계층적 ViT로, 다중 스케일 학습에 최적[1]

**기여**: Vision Transformer의 비로컬 성질과 큰 ERF가 HR에서의 강건성 제공[1]

### 7.3 2023-2025년 최신 경향

**S3OD (2025)**: 
- 139,000+ 대규모 합성 데이터셋
- 크로스 데이터셋 일반화에서 InSPyReNet 초과
- DIS (Dichotomous Image Segmentation) 데이터셋에서 우수[1]

**EFCRFNet (2024)**:
- 효율적 특징 융합 프레임워크
- MAE, F-measure 개선[1]

**BiRefNet**:
- 양방향 개선 네트워크
- 경계 개선과 배경 억제 강화[1]

### 7.4 연구 방향의 시사점

**데이터 효율성**: InSPyReNet 이후 합성 데이터 활용(S3OD), Few-shot 학습 등 데이터 효율성이 주요 연구 방향이 됨.[1]

**일반화 성능**: 크로스 도메인, 크로스 데이터셋 일반화가 점점 중요해지고 있음.[1]

**경계 품질**: mBA 메트릭의 도입으로 경계 정확도가 평가의 중요 요소로 인정됨.[1]

## 8. 앞으로의 연구에 미치는 영향

### 8.1 이론적 영향

**1) 고전적 기법의 재평가**
라플라시안 피라미드는 1983년부터 알려진 기법이지만, InSPyReNet의 성공은 **엄격한 구조 설계와 현대적 감독 전략**의 중요성을 증명했다. 이는 향후:[1]
- 다른 고전적 이미지 처리 기법의 재검토 필요
- 구조적 제약이 신경망 학습에 미치는 영향 연구[1]

**2) Scale-Invariant Design의 중요성**
SICA 모듈의 성공은 **훈련 시 스케일을 명시적으로 고려하는 설계**의 가치를 보여준다. 이는:[1]
- 다중 해상도 작업의 표준 설계 원칙 제시
- 다른 밀도 예측 작업(의미론적 분할, 깊이 추정)에 적용 가능[1]

### 8.2 실무적 영향

**1) 어노테이션 비용 절감**
HR 데이터 미필요로 새로운 도메인 진입 장벽 대폭 감소[1]

**2) 실시간 응용 가능성**
단일 네트워크로 다양한 해상도 처리 가능하여, 모바일/엣지 컴퓨팅 적용 용이[1]

**3) 의료 이미지 분석**
DIS5K 실험에서 보듯 의료 도메인으로의 자연스러운 확장 가능[1]

### 8.3 미래 연구 방향

**1) 멀티모달 확장**:
- RGB-D SOD (깊이 정보)
- RGB-Thermal SOD (열화상)
- 비디오 SOD (시간 정보)

논문에서 명시적으로 언급: _"멀티모달 입력(RGB-D SOD, 비디오 SOD)으로의 확장을 기대한다."_[1]

**2) 아키텍처 진화**:
- ConvNeXt, FastFourierConv 등 현대적 CNN 백본
- 동적 해상도 적응 메커니즘
- 하이브리드 CNN-Transformer 구조[1]

**3) 손실 함수 혁신**:
- 경계별 가중 손실
- 불확실성 기반 손실
- 대조 학습 통합

**4) 실시간 성능**:
- 경량 모델 설계
- 메모리 효율적 피라미드 블렌딩
- 추론 가속화[1]

## 9. 연구 시 고려해야 할 점

### 9.1 방법론적 고려사항

**1) Stage 설계의 일반성**
논문에서 Stage-3부터 시작하는 설정은 DUTS의 평균 해상도(378×469) 기반이다. 다른 데이터셋에서:[1]
- 더 큰 해상도 → Stage-4, Stage-5부터 시작 고려
- 더 작은 해상도 → Stage-2부터 시작 가능성 검토[1]

**2) 손실 함수 가중치의 적응성**
$\lambda_j = 4^j$는 경험적 설정이다. 다양한 데이터셋에서:[1]
- $\lambda_j = 2^j$, $3^j$, $5^j$ 비교 필요
- 데이터셋 특성별 최적 가중치 탐색[1]

**3) SICA의 재설계 가능성**
현재 SICA는 고정 크기로 리사이즈하는 방식이다. 다음 개선 가능:[1]
- 동적 리사이징으로 더 나은 특징 보존
- 다른 어텐션 메커니즘(self-attention, cross-attention) 통합

### 9.2 실험적 고려사항

**1) 데이터 불균형 영향**
- DUTS-TR: 중앙 객체 편향
- HRSOD-TE: 제한된 HR 데이터
- UHRSD-TE: 4K 해상도, 복잡한 장면

다양한 조합으로 학습할 때의 성능 영향 분석 필요.[1]

**2) Backbone 의존성**
논문에서 Swin은 필수, Res2Net은 HR에 부적합하다고 결론. 그러나:[1]
- ConvNeXt, EfficientNet 등 최신 CNN의 성능 평가 필요
- ViT 변형(DINO, DINOv2) 활용 가능성[1]

**3) 일반화 성능의 정량화**
현재 연구는 표준 SOD 데이터셋에 제한됨. 필요한 검증:[1]
- 도메인 시프트(의료, 산업, 위성)에서의 성능 저하율 정량화
- Few-shot 적응 능력 평가
- Zero-shot 전이 학습 가능성[1]

### 9.3 구현적 고려사항

**1) 메모리 효율성**
$L=1280$ 제약은 8V100 GPU 기준. 개선 방향:[1]
- Gradient checkpointing으로 메모리 절감
- 효율적 어텐션 메커니즘(linear attention) 도입
- 타일 기반 처리로 4K 지원[1]

**2) 추론 속도**
LR과 HR 이중 전진이 오버헤드를 야기. 최적화 방안:[1]
- 한 번의 전진으로 LR/HR 피라미드 동시 생성
- 조건부 계산(early exit)으로 불필요한 스테이지 스킵
- 모바일 최적화된 경량 버전[1]

**3) 재현성 관리**
논문의 강점인 공개 코드 기반으로:[1]
- 다양한 프레임워크(PyTorch, TensorFlow) 구현
- 다양한 하드웨어에서의 성능 보장
- 하이퍼파라미터 민감도 분석 자동화[1]

### 9.4 평가 메트릭 선택

**현재 메트릭의 문제점**:
- $S_\alpha$(Structure-Measure): 구조 유사성 중심
- $F_{\max}$(Maximum F-measure): 이진 맵 기반
- $MAE$(Mean Absolute Error): 픽셀 수준 오차
- **mBA**(Mean Boundary Accuracy): 경계 품질[1]

**주의**: 논문에서 BDE(Boundary Displacement Error)를 배제한 이유는 2002년부터의 오래된 메트릭이며, 구현이 불일치하기 때문. 향후:[1]
- mBA와 BIoU(Boundary IoU) 표준화 필요
- 도메인별 맞춤형 메트릭 개발[1]

## 결론

InSPyReNet은 **데이터 효율성, 아키텍처 단순성, 강건한 일반화**의 세 가지 측면에서 획기적인 기여를 했다. LR 데이터만으로 HR 성능을 달성하는 방식은 향후 밀도 예측 작업의 새로운 표준이 될 가능성이 높다. 다만 CNN 백본의 한계, 메모리 제약, 도메인별 일반화 검증 등의 과제가 남아있으며, 이들이 극복될 때 이 방법은 실제 응용 시스템에 광범위하게 채택될 것으로 예상된다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba7cba1e-6533-476d-a7f0-bb1a5c7d41ea/2209.09475v3.pdf)
[2](https://www.phwr.org/journal/view.html?doi=10.56786/PHWR.2023.16.3.1)
[3](https://ieeexplore.ieee.org/document/10448084/)
[4](https://journal.microbe.ru/jour/article/view/1819)
[5](https://www.canada.ca/content/dam/phac-aspc/documents/services/reports-publications/canada-communicable-disease-report-ccdr/monthly-issue/2023-49/issue-10-october-2023/ccdrv49i10a02-eng.pdf)
[6](https://link.springer.com/10.1007/s00247-023-05808-1)
[7](https://onlinelibrary.wiley.com/doi/10.1111/irv.13036)
[8](http://www.cdc.gov/mmwr/volumes/72/wr/mm7238a1.htm?s_cid=mm7238a1_w)
[9](https://aacrjournals.org/cancerres/article/83/7_Supplement/787/719268/Abstract-787-Lung-cancer-screening-use-before-amp)
[10](https://ieeexplore.ieee.org/document/10208745/)
[11](https://www.semanticscholar.org/paper/6b0df129ed1930525c5ebe57ccabfb3d23f89be8)
[12](https://rgu-repository.worktribe.com/preview/1694900/REN%202022%20PS-net%20(AAM).pdf)
[13](https://arxiv.org/ftp/arxiv/papers/2211/2211.06697.pdf)
[14](https://downloads.hindawi.com/journals/cin/2022/7780756.pdf)
[15](http://arxiv.org/pdf/2405.02906.pdf)
[16](https://arxiv.org/pdf/1501.02741.pdf)
[17](https://dx.plos.org/10.1371/journal.pone.0323757)
[18](https://arxiv.org/pdf/1908.07274.pdf)
[19](https://www.tandfonline.com/doi/pdf/10.1080/08839514.2022.2094408?needAccess=true)
[20](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Pixels_Regions_and_Objects_Multiple_Enhancement_for_Salient_Object_Detection_CVPR_2023_paper.pdf)
[21](https://pmc.ncbi.nlm.nih.gov/articles/PMC11119044/)
[22](https://peerj.com/articles/cs-2623/)
[23](https://arxiv.org/abs/1411.5878)
[24](https://www.nature.com/articles/s41598-024-61105-3)
[25](https://openaccess.thecvf.com/content/ICCV2021/papers/Tang_Disentangled_High_Quality_Salient_Object_Detection_ICCV_2021_paper.pdf)
[26](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2024.1356793/pdf)
[27](https://openaccess.thecvf.com/content/ACCV2022/papers/Kim_Revisiting_Image_Pyramid_Structure_for_High_Resolution_Salient_Object_Detection_ACCV_2022_paper.pdf)
[28](https://arxiv.org/abs/2108.03551)
[29](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2024.1356793/full)
[30](https://arxiv.org/html/2510.21605v1)
[31](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Pyramid_Grafting_Network_for_One-Stage_High_Resolution_Saliency_Detection_CVPR_2022_paper.pdf)
[32](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Pyramid_Feature_Attention_Network_for_Saliency_Detection_CVPR_2019_paper.pdf)
[33](https://arxiv.org/html/2412.14576v1)
[34](https://arxiv.org/abs/2209.09475)
[35](https://arxiv.org/html/2412.16609v2)
[36](http://arxiv.org/pdf/1903.00179.pdf)
[37](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510698.pdf)
[38](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516841/)
[39](https://pmc.ncbi.nlm.nih.gov/articles/PMC8514339/)
[40](http://arxiv.org/pdf/2309.16645.pdf)
[41](http://arxiv.org/pdf/2408.12605.pdf)
[42](http://arxiv.org/pdf/2411.09166.pdf)
[43](https://linkinghub.elsevier.com/retrieve/pii/S0140673624009334)
[44](http://arxiv.org/pdf/2403.05818.pdf)
[45](https://arxiv.org/html/2411.07463v1)
[46](https://www.paperdigest.org/wp-content/uploads/2021/10/ICCV-2021-Paper-Digests.pdf)
[47](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zeng_Towards_High-Resolution_Salient_Object_Detection_ICCV_2019_paper.pdf)
[48](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/papers/Zhang_ViT-YOLOTransformer-Based_YOLO_for_Object_Detection_ICCVW_2021_paper.pdf)
[49](https://en.wikipedia.org/wiki/Vision_transformer)
[50](https://arxiv.org/pdf/2408.01137.pdf)
[51](https://openaccess.thecvf.com/content_ICCV_2019/html/Zeng_Towards_High-Resolution_Salient_Object_Detection_ICCV_2019_paper.html)
[52](https://arxiv.org/abs/2104.10127v1)
[53](https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24l/liu24l.pdf)
[54](https://arxiv.org/html/2311.14746v1)
[55](https://arxiv.org/html/2509.19687v1)
[56](https://arxiv.org/abs/1908.07274)
[57](https://arxiv.org/abs/2108.07851)
[58](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Scaling_up_Image_Segmentation_across_Data_and_Tasks_CVPR_2025_paper.pdf)
