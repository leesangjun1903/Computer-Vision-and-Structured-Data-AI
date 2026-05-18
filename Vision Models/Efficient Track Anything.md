
# Efficient Track Anything (EfficientTAM)

> **논문 정보**
> - 제목: *Efficient Track Anything*
> - 저자: Yunyang Xiong et al. (Meta AI)
> - arXiv: [2411.18933](https://arxiv.org/abs/2411.18933) (2024년 11월 28일)
> - 발표: **ICCV 2025** (CVF Open Access)
> - 코드: [GitHub - yformer/EfficientTAM](https://github.com/yformer/EfficientTAM)

---

## 1. 핵심 주장 및 주요 기여 요약

SAM 2의 핵심 성능 요소는 대규모 다단계(multi-stage) 이미지 인코더와 과거 프레임의 메모리 컨텍스트를 저장하는 메모리 메커니즘인데, 이 고연산 복잡도가 모바일 기기 등 실세계 응용에서의 활용을 제한한다.

이를 해결하기 위해 EfficientTAM은 다음 두 가지 핵심 주장을 제시합니다:

| 핵심 기여 | 설명 |
|---|---|
| ① 경량 ViT 인코더 재활용 | 비계층적(non-hierarchical) 바닐라 ViT를 이미지 인코더로 사용 |
| ② 효율적 메모리 모듈 | 공간 메모리 토큰의 지역성을 활용한 풀링 기반 크로스-어텐션 |

그 결과, 제안된 EfficientTAM은 SAM 2 (HieraB+SAM 2)와 비슷한 성능을 보이면서 A100 기준 약 2배 속도 향상과 약 2.4배 파라미터 감소를 달성한다.

이미지 세그멘테이션 태스크에서도 원래의 SAM 대비 A100 기준 약 20배 속도 향상 및 약 20배 파라미터 감소를 달성한다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

SAM 2의 핵심 성능 요소인 대규모 다단계 이미지 인코더와 메모리 메커니즘은 높은 연산 복잡도를 초래하며, 모바일 기기에서의 비디오 객체 분할 등 실세계 작업에의 응용을 제한한다.

특히 메모리 토큰(공간 메모리 토큰과 객체 포인터 토큰의 결합)이 약 30K로 매우 길어, 메모리 모듈의 효율성 병목이 발생한다.

### 2-2. 제안 방법 및 핵심 수식

#### ① 효율적 이미지 인코더

경량 바닐라 ViT 이미지 인코더(예: ViT-Tiny/Small)를 사용하여 SAM 2의 복잡도를 줄이면서 적절한 성능을 유지한다.

ViT-Small 및 ViT-Tiny를 $16 \times 16$ 패치 크기로 채택하며, $14 \times 14$ 비겹침 윈도우드 어텐션과 4개의 균등 간격 전역 어텐션 블록을 사용하여 고해상도 프레임 특징을 효율적으로 추출한다.

SAM 2의 이미지 인코더와 달리, EfficientTAM의 이미지 인코더는 단일 스케일 피처 맵만 제공하며, 분할 마스크 생성을 위한 디코딩 시 업샘플링 레이어에 다른 피처가 추가되지 않는다.

#### ② 효율적 메모리 크로스-어텐션 (핵심 수식)

기존 SAM 2의 표준 크로스-어텐션은 다음과 같이 정의됩니다.

현재 프레임 피처 $X \in \mathbb{R}^{L \times d_q}$에서 쿼리를 생성하고, 메모리 토큰 $M_b \in \mathbb{R}^{(n+P) \times d_m}$에서 키와 값을 생성합니다:

$$Q = X W_Q, \quad K = M_b W_K, \quad V = M_b W_V$$

표준 스케일드 닷-프로덕트 크로스-어텐션:

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

이때 복잡도는 $O(L \cdot (n + P))$로, $n \approx 30\text{K}$이면 매우 비효율적입니다.

**EfficientTAM의 효율적 크로스-어텐션:**

메모리 공간 토큰의 내재적 구조를 활용하며, 메모리 공간 토큰이 강한 지역성(locality)을 가지고 있어 더 거친(coarser) 표현이 크로스-어텐션의 좋은 대리(proxy)가 됨을 관찰했다.

모든 메모리 토큰을 사용하는 대신 인접 토큰을 그룹화하여 평균하는 풀링 연산을 제안한다. 이 대리 표현은 크로스-어텐션에 관여하는 토큰 수를 줄여 처리 시간을 최적화하고 전통적으로 이차(quadratic) 복잡도를 갖는 행렬 곱셈 비용을 줄인다. 평균 풀링 전략은 입력 공간 토큰을 직사각형 영역으로 분할하고 평균을 계산하여 더 적은 토큰으로 영역을 표현한다.

풀링된 키와 값을 $\bar{K}, \bar{V}$로 표기하면:

$$\bar{K} = \text{AvgPool}(K_{\text{spatial}}), \quad \bar{V} = \text{AvgPool}(V_{\text{spatial}})$$

효율적 메모리 크로스-어텐션은 세 단계로 구성된다: (a) 공간 키와 값에 대한 평균 풀링, (b) 평탄화(flatten) 후 객체 포인터와 연결(concatenate), (c) 크로스-어텐션 연산 수행.

구체적으로, 메모리 공간 토큰의 지역성을 활용한 더 거친 표현을 생성하고, 이 거친 임베딩을 객체 포인터 토큰과 연결한다.

따라서 최종 효율적 크로스-어텐션은:

$$\tilde{K} = \text{Concat}(\bar{K}_{\text{spatial}},\, K_{\text{ptr}}), \quad \tilde{V} = \text{Concat}(\bar{V}_{\text{spatial}},\, V_{\text{ptr}})$$

$$\text{EfficientCrossAttn}(Q, \tilde{K}, \tilde{V}) = \text{softmax}\!\left(\frac{Q\tilde{K}^\top}{\sqrt{d}}\right)\tilde{V}$$

풀링 시 공간 토큰만을 사용하는 것이 전체 메모리 토큰을 풀링하는 것에 비해 상당한 성능 향상을 가져온다는 것이 실험으로 확인되었다.

#### ③ 학습 전략

SA-1B 데이터셋으로 메모리 컴포넌트 없이 90k 스텝 사전학습하며, ViT 이미지 인코더는 사전학습된 ViT로 초기화된다. AdamW 옵티마이저를 사용하고 ($\beta_2 = 0.999$), 글로벌 배치 크기 256, 초기 학습률 $4 \times 10^{-4}$을 적용한다. 학습률은 역 제곱근 스케줄로 감소시키며, 1k 이터레이션 선형 웜업과 5k 이터레이션 선형 쿨다운을 사용한다.

### 2-3. 모델 구조 상세

EfficientTAM은 SAM 2의 전체 구조를 대체로 따르되, 핵심 컴포넌트를 경량화합니다.

```
[Input Frame]
     │
     ▼
[Vanilla ViT Encoder (ViT-Tiny/Small)]  ← 단일 스케일 피처 맵
     │
     ▼
[Memory Attention Module]
  - Query: 현재 프레임 피처
  - Key/Value: 풀링된 공간 메모리 토큰 + 객체 포인터 토큰
     │
     ▼
[Mask Decoder]
     │
     ▼
[Output Segmentation Mask]
```

EfficientTAM은 경량 바닐라 ViT를 이미지 인코더로 사용하며, SAM 2의 계층적 아키텍처와 대조적이다. 구체적으로 ViT-Tiny와 ViT-Small을 활용하여 파라미터와 연산을 줄인다.

EfficientTAM은 통합 이미지 및 비디오 분할을 위해 SA-1B(이미지)와 SA-V(비디오) 데이터셋에서 완전히 학습된다.

### 2-4. 성능 향상

MOSE, DAVIS, LVOS, SA-V(비디오 분할), SA-23(이미지 분할) 등의 광범위한 벤치마크에서 평가를 수행하며, EfficientTAM은 Cutie-base, XMem, DEVA 등의 강력한 준지도 비디오 객체 분할 방법들을 효율성 측면에서 능가한다.

특히 EfficientTAM-S/2의 2×2 윈도우 풀링은 iPhone 15에서의 지연 시간을 프레임당 1010.8ms에서 450ms로 2배 이상 줄이면서 정확도 손실은 미미하다(SA-V 테스트 기준 74.5 J&F → 74.0 J&F). 이는 공간 토큰 지역성을 활용한 효율적 크로스-어텐션이 성능을 심각하게 희생하지 않고도 효과적임을 입증한다.

EfficientTAM-Mobile은 SAM 2 대비 추론 시간을 약 4.6배, 파라미터 크기를 약 4.5배 줄이며, EdgeTAM과 비교하면 A100에서 약 2배 효율적이면서 더 정확하다.

프롬프터블 비디오 분할에서도 EfficientTAM은 SAM+XMem++와 SAM+Cutie를 오프라인 및 온라인 평가 모두에서 능가하며, 8개 주석 프레임 기준 EfficientTAM-S와 EfficientTAM-S/2는 약 82 J&F(오프라인), 81 J&F(온라인)를 달성한다.

### 2-5. 한계점

해상도 변화 실험에서 낮은 해상도는 속도를 향상시키지만 정확도를 낮추는 트레이드오프가 존재함이 확인되었다.

또한 다음과 같은 구조적 한계가 존재합니다:

- **단일 스케일 피처 맵**: SAM 2와 달리 단일 스케일 피처 맵만 제공하여, 세밀한 객체 경계 처리에서 다중 스케일 정보의 부재로 인한 한계가 있을 수 있다.
- **복잡한 환경에서의 일반화 한계**: SAM2조차 MOSEv1 기준 76.4%에서 MOSEv2에서 50.9%로 급격히 성능이 하락하며, 경량 모델인 EfficientTAM은 이러한 복잡 환경에서 추가적인 성능 저하가 우려됩니다.
- **선형 어텐션과의 비교**: 선형 어텐션 등의 기존 효율적 어텐션 방법들이 메모리 크로스-어텐션 모듈에 적용될 경우 성능이 저하되는 것을 예비 실험에서 확인하였다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. SA-1B + SA-V 통합 학습을 통한 범용성

EfficientTAM 모델은 SA-1B(이미지)와 SA-V(비디오) 데이터셋에서 완전히 학습되어 통합 이미지 및 비디오 분할 능력을 제공한다.

이러한 대규모 다양한 데이터셋에서의 사전 학습은 다양한 객체 범주와 장면에 대한 일반화를 가능하게 합니다.

### 3-2. 플러그-앤-플레이 모듈 호환성을 통한 일반화 확장

DAM(Distractor-Aware Memory) 모듈이 EfficientTAM에 통합되었을 때 EfficientTAM의 성능을 최대 11%까지 향상시켰으며, 이는 DAM의 일반화 능력을 입증한다.

DAM이 통합된 EfficientTAM은 여러 추적 및 분할 벤치마크에서 비실시간 SAM 2.1-L과 동등한 추적 품질을 달성하며, EdgeTAM과의 통합에서도 4% 성능 향상을 보여 아키텍처 간 우수한 일반화를 입증한다.

### 3-3. 제로샷 일반화 가능성

플레인(비계층적) 이미지 인코더를 비디오 객체 분할에 재적용한 EfficientTAM은, 바닐라 경량 ViT 이미지 인코더로 계층적 이미지 인코더와 경쟁적인 이미지/비디오 분할 능력을 보이면서도 모바일 기기에 보다 효율적으로 배포 가능함을 입증했다.

### 3-4. 일반화의 한계와 극복 방향

MOSEv2와 같은 복잡한 벤치마크 평가에서 현재 최첨단 방법들도 상당한 성능 저하를 겪으며, SAM2는 DAVIS 2017에서 90.7%에서 MOSEv2에서 50.9%로 하락한다. 이러한 결과는 기존 알고리즘과 실세계 배포 요구사항 간의 격차를 보여준다.

희귀 카테고리에 대한 일반화도 중요한 과제이며, VOS 방법들이 일반적으로 클래스 비가지적(class-agnostic)으로 설계되지만 희귀하거나 보지 못한 카테고리에 대한 강건한 일반화는 여전히 중요한 도전 과제이다.

이를 해결하기 위해 첫 프레임 단서를 더 효과적으로 활용하는 테스트 타임 적응 기법, 또는 희귀하고 시각적으로 모호한 객체에 더 잘 일반화하는 강건한 인스턴스 수준 표현 개발이 유망한 방향이다.

---

## 4. 앞으로의 연구에 미치는 영향 및 연구 시 고려할 점

### 4-1. 연구에 미치는 영향

**① 경량화 패러다임 전환:**
비계층적 이미지 인코더를 비디오 객체 분할에 활용하는 패러다임이 계층적 인코더에 필적하는 성능을 낼 수 있음을 증명하여, 대형 모델 설계 가정에 재고를 촉구한다.

**② 온디바이스 AI 연구 활성화:**
EfficientTAM은 온디바이스 트랙 애니씽 응용에 많은 잠재적 활용 가능성을 가진다.
이 연구는 온디바이스 능력에 맞춘 효율적인 AI 모델 설계의 추가 탐구를 위한 길을 열었다.

**③ 모듈화된 효율성 개선 기반 제공:**
EfficientTAM 및 EdgeTAM에 DAM이 통합되어 일관되게 기준선을 능가하는 결과를 보여, SAM 계열 경량 모델이 향후 다양한 메모리 개선 연구의 테스트베드가 됨을 보여준다.

### 4-2. 앞으로 연구 시 고려할 점

**① 복잡 환경 일반화 강화:**
효율적인 고해상도 처리 전략, 소형 객체에 특화된 향상된 피처 학습, 다중 스케일 아키텍처, 혼잡 환경에서 시각적으로 유사한 방해물로부터 객체를 분리하는 대조 학습 기법 등이 유망한 연구 방향이다.

**② 장기 비디오에서의 오류 누적 대응:**
SAM2Long은 SAM 2를 기반으로 다중 경로 메모리 트리 구조와 불확실성 메커니즘을 제안하여 장기 비디오에서의 오류 누적을 완화하는데, EfficientTAM도 유사한 장기 추적 안정성 연구가 필요합니다.

**③ 악천후·저조도 환경 강건성:**
비, 눈, 안개, 야간 및 수중 등 다양한 환경이 기존 VOS 방법의 성능을 크게 저하시키며, 이러한 조건에서 객체 외관이 불안정해지고 시간적 일관성이 깨질 수 있다.

**④ 메모리 구조 고도화:**
메모리 공간 토큰의 강한 지역성을 활용한 거친 표현이 크로스-어텐션에 좋은 대리가 됨을 보였으나, 이를 더욱 정교하게 제어하는 적응형 풀링 전략이나 계층적 메모리 구조 연구가 필요합니다.

**⑤ 테스트 타임 적응(Test-Time Adaptation) 연구:**
분포 이동 하에서 매칭 기반 VOS 방법의 일반화를 테스트 시 단일 비디오를 통한 미세조정으로 개선하는 방법으로, 이는 분류 작업에서 최근 주목받는 테스트 타임 학습(TTT)의 유망한 적용 방향이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 특징 | DAVIS2017 J&F | 모바일 FPS |
|---|---|---|---|---|
| XMem | 2022 | 통합 메모리 공간 기반 매칭 | 86.2% | ❌ |
| SAM (원본) | 2023 | 이미지 세그멘테이션 파운데이션 모델 | N/A | ❌ |
| SAM-Track | 2023 | SAM+DeAOT+Grounding-DINO 결합 | ~79.2% | ❌ |
| SAM 2 (HieraB+) | 2024 | 계층적 인코더+메모리 모듈 | 90.7% | ~1 FPS |
| **EfficientTAM-S** | **2024** | **바닐라 ViT+풀링 크로스-어텐션** | **SAM2 수준** | **~10 FPS** |
| EdgeTAM | 2024 | 2D Spatial Perceiver 기반 | 87.7% | ~16 FPS |
| DAM4SAM+EfficientTAM | 2025 | DAM 통합 EfficientTAM | +11% 향상 | ~10 FPS |

TAM(Track Anything Model)은 SAM과 XMem을 결합하여 대화형 비디오 객체 추적 및 분할을 수행한다.

SAM-Track은 SAM, DeAOT, Grounding-DINO를 결합하여 비디오에서의 객체 추적 및 분할을 수행한다.

EdgeTAM은 2D Spatial Perceiver를 활용하여 연산 비용을 줄이며, 이 퍼시버는 고정된 학습 가능한 쿼리 집합을 가진 경량 트랜스포머로 밀집 저장된 프레임 레벨 메모리를 인코딩한다. 비디오 분할이 밀집 예측 태스크이므로, 쿼리를 전역 수준과 패치 수준 그룹으로 분리하여 메모리의 공간 구조를 보존하는 것이 필수적임을 발견했다.

EdgeTAM은 DAVIS 2017에서 87.7 J&F, MOSE에서 70.0 J&F, SA-V val에서 72.3, SA-V test에서 71.7 J&F를 달성하며 iPhone 15 Pro Max에서 16 FPS로 동작한다.

---

## 📚 참고 자료 및 출처

1. **Efficient Track Anything (EfficientTAM)** - arXiv:2411.18933, Yunyang Xiong et al., 2024
   - arXiv 원문: https://arxiv.org/abs/2411.18933
   - ICCV 2025 논문: https://openaccess.thecvf.com/content/ICCV2025/papers/Xiong_Efficient_Track_Anything_ICCV_2025_paper.pdf
   - 공식 GitHub: https://github.com/yformer/EfficientTAM
   - 프로젝트 페이지: https://yformer.github.io/efficient-track-anything/
   - Hugging Face: https://huggingface.co/papers/2411.18933

2. **SAM 2: Segment Anything in Images and Videos** - arXiv:2408.00714, Ravi et al., 2024
   - https://arxiv.org/html/2408.00714v1

3. **Segment and Track Anything (SAM-Track)** - arXiv:2305.06558, Cheng et al., 2023
   - https://arxiv.org/abs/2305.06558

4. **Distractor-Aware Memory-Based Visual Object Tracking (DAM4SAM)** - arXiv:2509.13864, 2025
   - https://arxiv.org/html/2509.13864v1

5. **MOSEv2: A More Challenging Dataset for VOS** - arXiv:2508.05630, 2025
   - https://arxiv.org/html/2508.05630v1

6. **EdgeTAM** - LVOS Benchmark 관련 연구
   - https://www.researchgate.net/publication/395588995_LVOS

7. **[Literature Review] Efficient Track Anything** - Moonlight AI
   - https://www.themoonlight.io/en/review/efficient-track-anything

8. **[Quick Review] Efficient Track Anything** - Liner
   - https://liner.com/review/efficient-track-anything

9. **EfficientSAM GitHub (EfficientTAM 코드 포함)** - yformer/EfficientSAM
   - https://github.com/yformer/EfficientSAM
