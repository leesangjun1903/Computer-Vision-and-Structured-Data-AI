# SAM Fails to Segment Anything? – SAM-Adapter: Adapting SAM in Underperformed Scenes: Camouflage, Shadow, Medical Image Segmentation, and More

---

## 1. Executive Summary (10문장 이내)

SAM(Segment Anything Model)은 대규모 시각 데이터로 학습된 파운데이션 모델이지만, 위장 객체 감지(Camouflaged Object Detection)나 그림자 감지(Shadow Detection) 같은 특수 도메인에서 심각한 성능 저하를 보인다.  
본 논문은 SAM의 이러한 한계를 체계적으로 분석하고, 이를 해결하기 위한 **SAM-Adapter**를 제안한다.  
SAM-Adapter는 SAM의 사전학습 가중치를 동결(freeze)한 채로, 경량 어댑터 모듈을 통해 도메인별 태스크 특화 지식을 주입하는 방식이다.  
어댑터는 두 개의 MLP와 GELU 활성화 함수로 구성된 단순하면서도 효과적인 구조를 채택하였다.  
실험은 위장 객체 감지(COD10K, CAMO, CHAMELEON), 그림자 감지(ISTD), 용종 분할(kvasir-SEG) 데이터셋에서 수행되었다.  
SAM-Adapter는 위장 객체 감지에서 기존 SOTA 대비 Sα 기준 최대 +17.9% 향상을 달성하였다.  
그림자 감지에서는 SAM의 BER 40.51에서 1.43으로 급격히 개선하여 SOTA를 달성하였다.  
의료 영상 분할에서도 UNet, UNet++ 대비 높은 mDice(0.850)를 기록하였다.  
본 연구는 대형 파운데이션 모델을 특수 도메인에 효율적으로 적응시키는 방법론적 선례를 제시한다.  
SAM-Adapter는 의료, 농업, 원격 탐사 등 다양한 분야로의 확장 가능성을 열어놓았다.

---

### 1-1. 연구의 목적과 필요성

**목적:** SAM이 취약한 특수 시각 도메인(위장, 그림자, 의료 영상)에서 성능을 획기적으로 향상시키면서도, SAM의 대규모 사전학습 지식을 최대한 보존하는 경량 적응(adaptation) 방법론 개발.

**필요성:**
- SAM은 범용 이미지 분할 모델로 설계되었으나, 저수준 구조적 특징(low-level structural features)이 중요한 태스크에서 심각한 성능 저하 발생 (p.2, Introduction)
- 전체 파인튜닝(full fine-tuning)은 대형 모델에 과도한 계산 자원을 요구하며, 사전학습 지식을 손상시킬 위험이 있음
- 파운데이션 모델의 능력을 하위 도메인 태스크에 효과적으로 이전하는 방법론이 부재했음

> 💡 **파운데이션 모델(Foundation Model):** BERT, GPT-3, SAM처럼 대규모 데이터로 사전학습된 후 다양한 하위 태스크에 적용 가능한 대형 AI 모델. "기반 모델"이라고도 함.

> 💡 **파인튜닝(Fine-tuning):** 사전학습된 모델의 가중치를 특정 태스크의 데이터로 추가 학습하는 과정.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|-------|
| SAM은 특수 도메인에서 성능이 저하된다 | 위장 감지에서 SAM Sα=0.727(CHAMELEON), 그림자 감지 BER=40.51로 기존 SOTA 대비 현저히 낮음 | Table 1, 2 (p.5) |
| SAM-Adapter는 태스크 특화 지식을 효과적으로 주입한다 | 어댑터를 통한 패치 임베딩+고주파 성분 주입으로 위장 감지 Sα +17.9% 향상 | p.5, Section 4.3 |
| SAM의 인코더를 동결해도 어댑터만으로 성능 향상 가능 | 동결된 ViT-H 인코더 위에 경량 MLP 어댑터만으로 SOTA 달성 | Figure 1 (p.4), Section 3 |
| SAM-Adapter는 다중 태스크에 범용 적용 가능 | 위장/그림자/의료 영상 세 가지 이질적 태스크에서 모두 성능 향상 | Table 1, 2, 3 (p.5, 7) |
| 기존 태스크 특화 모델을 능가하는 SOTA 달성 | 위장 감지에서 FBNet, PFNet 등 태스크 특화 모델 대비 우월한 성능 | Table 1 (p.5) |

---

### 2-1. 상세 설명

#### ① 해결하고자 하는 문제 (p.1-2)

SAM은 대규모 범용 이미지로 학습되었기 때문에:
- **위장 객체**: 배경과 시각적으로 유사한 객체를 구별하지 못함
- **그림자**: 그림자 영역과 일반 어두운 배경을 구별하지 못함 (BER=40.51)
- **의료 영상**: 도메인 특화 시각 패턴(용종 등)을 인식하지 못함

#### ② 제안하는 방법 (p.3-4, Section 3)

**핵심 수식:**

$$P^i = \text{MLP}_{up}\left(\text{GELU}\left(\text{MLP}^i_{tune}(F_i)\right)\right) \quad \cdots (1)$$

**기호 설명:**
- $P^i$: $i$번째 어댑터가 생성하는 출력 프롬프트 (SAM 트랜스포머 레이어에 주입됨)
- $\text{MLP}^i_{tune}$: 각 어댑터마다 독립적인 선형 레이어 (layer-unshared), 태스크 특화 프롬프트 생성용
- $\text{MLP}_{up}$: 모든 어댑터가 공유하는 업프로젝션 선형 레이어 (layer-shared), 트랜스포머 입력 차원에 맞게 조정
- $F_i$: $i$번째 어댑터에 입력되는 태스크 특화 정보
- $\text{GELU}$: Gaussian Error Linear Unit 활성화 함수

$$F_i = \sum_{1}^{N} w_j F_j \quad \cdots (2)$$

**기호 설명:**
- $F_i$: 최종 합성된 태스크 특화 입력 정보
- $F_j$: $j$번째 유형의 특화 지식/특징 (예: 패치 임베딩 $F_{pe}$, 고주파 성분 $F_{hfc}$)
- $w_j$: 각 지식 유형의 혼합 강도를 조절하는 가중치 (실험에서 $w_j=1$로 설정)
- $N$: 사용하는 지식 유형의 수

**실험 설정에서의 구체적 적용:**

$$F_i = F_{hfc} + F_{pe}$$

- $F_{hfc}$: 고주파 성분 (High-Frequency Components) — 경계선, 텍스처 등 세밀한 구조 정보
- $F_{pe}$: 패치 임베딩 (Patch Embedding) — 위치 및 전반적 시각 정보

> 💡 **GELU (Gaussian Error Linear Unit):** ReLU의 개선된 버전으로, 음수 입력에도 작은 기울기를 허용하는 활성화 함수. Transformer 모델에서 널리 사용됨.

> 💡 **고주파 성분(High-Frequency Components):** 이미지에서 급격히 변하는 부분(경계, 텍스처)을 나타내는 신호. 위장 객체처럼 경계가 모호한 경우 중요한 단서가 됨.

#### ③ 모델 구조 (Figure 1, p.4)

```
[입력 이미지]
      ↓
[Patch Embedding] ← Adaptor 1 ← Task-Specific F_i
      ↓                              ↑
[SAM Encoder Layer 1] ← Adaptor 2   MLP_tune (layer-unshared)
      ↓                              MLP_up   (layer-shared)
[SAM Encoder Layer 2] ← Adaptor 3
      ↓  (동결된 가중치)
      ...
[SAM Encoder Layer N] ← Adaptor N+1
      ↓
[SAM Mask Decoder] (파인튜닝 가능)
      ↓
[분할 결과]
```

- **동결(Frozen):** SAM 이미지 인코더(ViT-H/16, 14×14 windowed attention + 4개 global attention blocks)
- **학습 가능(Tunable):** 어댑터 모듈 전체 + SAM 마스크 디코더

> 💡 **ViT-H (Vision Transformer Huge):** 이미지를 패치 단위로 분할하여 트랜스포머로 처리하는 모델의 초대형 버전. SAM의 이미지 인코더로 사용됨.

> 💡 **Windowed Attention:** 메모리 효율을 위해 전체 이미지 대신 국소 윈도우 내에서만 자기 어텐션(self-attention)을 계산하는 방식.

#### ④ 성능 향상 (Table 1, 2, 3)

| 태스크 | 지표 | SAM | SAM-Adapter | 향상 |
|--------|------|-----|-------------|------|
| 위장(COD10K) | $S_\alpha$ ↑ | 0.783 | **0.883** | +10.0% |
| 위장(CHAMELEON) | $S_\alpha$ ↑ | 0.727 | **0.896** | +23.2% |
| 위장(CAMO) | $S_\alpha$ ↑ | 0.684 | **0.847** | +23.8% |
| 그림자(ISTD) | BER ↓ | 40.51 | **1.43** | -96.5% |
| 용종 분할(kvasir) | mDice ↑ | 0.778 | **0.850** | +9.3% |

> 💡 **$S_\alpha$ (S-measure):** 구조적 유사도를 측정하는 지표. 높을수록 예측 분할이 실제와 구조적으로 유사함. 0~1 범위.

> 💡 **BER (Balance Error Rate):** 그림자 감지의 균형 오차율. 낮을수록 좋음. 양성/음성 클래스 오류를 균등하게 고려함.

> 💡 **mDice:** 두 집합의 중첩 정도를 측정하는 Dice 계수의 평균. 의료 영상 분할에서 자주 사용. 높을수록 좋음.

#### ⑤ 한계

- 실험 도메인이 3가지 태스크로 제한적 (위성 이미지, 농업 등 미검증)
- 태스크 특화 정보($F_i$)의 설계가 수동(hand-crafted)에 의존
- 어블레이션 스터디(ablation study) 부재 — 어댑터 구성 요소별 기여도 불명확
- 논문이 프리프린트(arXiv) 상태로, 동료 심사(peer review) 미완료

> 💡 **어블레이션 스터디(Ablation Study):** 모델의 각 구성 요소를 하나씩 제거하거나 변경하여, 각 요소의 기여도를 측정하는 실험.

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|-----------|
| SAM은 위장 감지에서 실패한다 | p.5, Section 4.3, Figure 2, Table 1 |
| SAM은 그림자 감지에서 실패한다 | p.5-6, Section 4.4, Figure 4, 5, Table 2 |
| SAM-Adapter 구조 설명 | p.3-4, Section 3.1-3.3, Figure 1 |
| 어댑터 수식 | p.4, Equation (1), (2) |
| 위장 감지 SOTA 달성 | p.5, Table 1 |
| 그림자 감지 SOTA 달성 | p.5-6, Table 2 |
| 용종 분할 성능 향상 | p.7, Table 3, Figure 6 |
| 구현 상세 (optimizer, epoch 등) | p.5, Section 4.2 |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

**연구 주제:**
> "SAM이 특수 저수준 구조 분할 태스크에서 실패하는 현상을 분석하고, 어댑터를 통해 이를 개선한다." (p.1, Abstract)

**방법:**

$$P^i = \text{MLP}_{up}\left(\text{GELU}\left(\text{MLP}^i_{tune}(F_i)\right)\right)$$

$$F_i = F_{hfc} + F_{pe}$$

" $\text{MLP}^i_{tune}$은 32개의 선형 레이어, $\text{MLP}_{up}$은 1개의 선형 레이어로 구성" (p.5, Section 4.2)

**결과 (저자 직접 보고):**
- 위장 감지(CHAMELEON): SAM 대비 $S_\alpha$ +17.9% 향상 (p.5, Section 4.3)
- 그림자 감지(ISTD): BER 40.51 → 1.43, SOTA 달성 (p.6, Table 2)
- 용종 분할: mDice 0.850, mIoU 0.776, UNet/UNet++ 능가 (p.7, Table 3)

### 분석자(필자)의 해석

- **성능 향상의 주요 원인:** SAM의 범용 특징 추출 능력에 도메인 특화 고주파 정보를 결합함으로써, 시각적 모호성이 높은 영역에서의 경계 구분 능력이 향상된 것으로 판단됨. 이는 SAM이 의미론적(semantic) 수준의 특징에는 강하나, 저수준 텍스처·경계 수준에서 약하다는 것을 시사함.
- **어댑터 경량성의 의미:** 전체 파라미터 대비 극소수의 추가 파라미터만으로 SOTA를 달성한 것은, SAM 인코더 내부에 이미 잠재적으로 유용한 표현이 존재함을 간접적으로 시사함. 즉, 문제는 SAM의 표현력이 아닌 도메인 격차(domain gap)임.
- **일반화 주장의 한계:** 저자들은 "일반화 가능하다(Generalizable)"고 주장하나, 3가지 태스크로 검증 범위가 제한적이어서 이 주장은 과장될 수 있음.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 문제점 | 위치 |
|------|--------|-------|
| ⚠️ 어블레이션 스터디 부재 | $F_{hfc}$, $F_{pe}$ 각각의 개별 기여도 불명확. 두 요소 중 어느 것이 더 중요한지 알 수 없음 | Section 4.2, p.5 |
| ⚠️ CHAMELEON 테스트셋 크기 | CHAMELEON은 76장으로 매우 소규모. 이 결과의 통계적 신뢰성이 낮을 수 있음 | Section 4.1, p.4 |
| ⚠️ SAM 비교 조건 불통일 | SAM을 "different prompting approaches"로 테스트했으나, 어댑터와의 공정한 단일 조건 비교가 불명확 | Figure 3, p.6 |
| ⚠️ 용종 분할 비교군 한계 | UNet(2015), UNet++(2018), SFA(2019) 등 오래된 모델과 비교. 최신 의료 분할 SOTA(예: TransFuse, Polyp-PVT 등)와 비교 없음 | Table 3, p.7 |
| ⚠️ BER 수치의 극단적 차이 | SAM의 BER=40.51은 비교 불가능한 수준으로 낮은 수치. 이는 SAM이 그림자 감지에 전혀 적합하지 않음을 의미하며, 비교 의미가 제한적 | Table 2, p.5 |
| ⚠️ 통계적 유의성 검정 없음 | 모든 성능 비교에서 표준편차, p-value 등 통계 검정 결과 미보고 | Tables 1-3 |
| ⚠️ 단일 실행 결과 | 랜덤 시드에 따른 성능 변동성 미보고 | Section 4.2 |

---

## 6. 문서가 답하지 않는 질문

1. **어댑터 개수 민감도:** 트랜스포머 레이어 수와 어댑터 수의 최적 비율은?
2. **$F_i$ 설계의 일반 원칙:** 새로운 태스크에서 어떤 기준으로 $F_j$를 선택해야 하는가?
3. **SAM 버전 의존성:** ViT-B, ViT-L 등 경량 SAM 버전에서도 동일한 효과가 나타나는가?
4. **학습 데이터 규모 민감도:** 소량 데이터(few-shot)에서의 성능은 어떠한가?
5. **추론 속도 및 계산 비용:** 어댑터 추가로 인한 추론 지연(latency)은 얼마나 발생하는가?
6. **다른 파운데이션 모델 적용 가능성:** DINO, DINOv2 등 다른 모델에도 동일 방식 적용 가능한가?
7. **가중치 $w_j$의 민감도:** $w_j=1$ 이외의 값에서의 성능 변화는?
8. **실패 사례 분석:** SAM-Adapter가 여전히 실패하는 케이스의 패턴은 무엇인가?
9. **교차 도메인 일반화:** 위장 데이터로 학습된 어댑터가 그림자 감지에 전이 가능한가?
10. **동결 vs. 전체 학습 비교:** SAM 인코더를 동결하지 않고 전체 학습할 경우와의 성능 비교는?

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1: SAM-Adapter 아키텍처 (p.4)

SAM-Adapter의 전체 구조를 보여주는 핵심 도식. 이미지가 Patch Embedding을 통해 SAM 인코더에 입력되고, 각 트랜스포머 레이어 사이에 어댑터가 병렬로 연결된다. 어댑터 내부는 Layer-unshared $\text{MLP}^i_{tune}$과 Layer-shared $\text{MLP}_{up}$으로 구성된다. SAM 인코더(파란 박스)는 완전히 동결되어 있고, 어댑터와 SAM 마스크 디코더만 학습된다. 이 구조의 핵심은 **잔차 연결(residual connection, ⊕)**을 통해 어댑터의 출력이 트랜스포머 레이어의 입력에 더해진다는 점이며, 이는 원래 SAM의 표현을 보존하면서 도메인 지식을 점진적으로 주입하는 효과를 낸다.

> 💡 **잔차 연결(Residual Connection):** 이전 레이어의 출력을 현재 레이어의 출력에 직접 더하는 연결 방식. 기울기 소실을 방지하고 원래 정보를 보존하는 데 효과적.

### Figure 2: 위장 객체 분할 시각화 (COD-10K) (p.6)

4행(RGB Image, SAM, Ours, GT)으로 구성된 비교 시각화. SAM 행을 보면 위장된 동물의 극히 일부분만 검출하거나 완전히 실패하는 경우가 많다. SAM-Adapter(Ours) 행은 GT와 매우 유사하게 위장 객체 전체를 포착한다. 특히 배경과 색상/질감이 거의 동일한 경우에도 경계를 비교적 정확히 찾아냄을 확인할 수 있다. 이는 고주파 성분($F_{hfc}$)이 경계 감지에 실질적으로 기여함을 시각적으로 지지한다.

### Figure 5: 그림자 감지 시각화 (ISTD) (p.7)

6개 샘플에 대한 그림자 감지 결과. SAM 행에서는 그림자 영역과 일반 어두운 배경을 혼동하여 과분할(oversegmentation) 또는 누락이 심각하게 나타난다. SAM-Adapter(Ours) 행은 그림자 윤곽을 훨씬 정확하게 포착한다. 일부 샘플에서 SAM-Adapter의 결과가 GT와 완벽히 일치하지는 않지만, SAM 대비 압도적인 개선이 관찰된다. BER 40.51→1.43이라는 수치적 개선이 시각적으로도 명확히 확인된다.

### Figure 6: 용종 분할 시각화 (kvasir-SEG) (p.8)

대장 내시경 이미지에서의 용종 분할 결과. SAM 단독으로는 용종 경계를 제대로 검출하지 못하거나 주변 조직을 포함하는 경우가 많다. SAM-Adapter는 용종 영역을 더 정확하게 분리하며, GT와의 시각적 유사도가 높다. 이 그림은 SAM-Adapter가 의료 영상이라는 완전히 다른 도메인에서도 적용 가능함을 보여주는 증거로서 중요하다.

### Figure 3: SAM의 다양한 프롬프팅 방식 비교 (p.6)

SAM을 "SAM online"(전체 이미지에 균등 분포 점 프롬프트), "SAM"(전체 이미지 크기 박스 프롬프트), "SAM-Adapter"(어댑터 적용)로 비교한 그림. 어떤 프롬프팅 방식을 사용해도 SAM 단독으로는 위장 객체를 제대로 감지하지 못하는 반면, SAM-Adapter는 명확히 성공한다. 이는 SAM의 실패가 프롬프트 설계 문제가 아니라, 모델 자체의 도메인 격차 문제임을 시사하는 중요한 대조 실험이다.

> 💡 **프롬프트(Prompt):** SAM에서 분할하고자 하는 영역을 지정하는 입력 신호. 점(point), 박스(bounding box), 마스크 등의 형태가 있음.

---

## 8. 결론, 시사점 및 후속 연구

### 저자 제시 시사점 (p.8, Section 5-6)

- SAM 같은 파운데이션 모델도 특수 도메인에서는 명확한 한계가 있음
- 경량 어댑터를 통한 지식 주입이 전체 파인튜닝 없이도 SOTA 달성 가능
- 의료, 농업, 원격탐사 등 다양한 분야로 확장 가능성 제시

### 저자 제시 후속 연구 (p.8, Section 6)

- 더 어려운 이미지 분할 태스크로의 SAM-Adapter 확장
- 태스크 특화 어댑터 설계의 전문화
- 적용 분야 다각화

---

### 8-1. 모델 일반화 성능 향상 가능성

**현재 일반화 능력의 증거 및 한계:**

저자들은 위장/그림자/의료 영상이라는 이질적 3개 태스크에서 동일한 어댑터 구조가 유효함을 보여, 구조적 일반화 가능성을 시사한다. 그러나 다음의 측면에서 일반화 능력 향상이 필요하다:

**① 도메인 외 일반화 (Out-of-Domain Generalization)**

현재 어댑터는 각 태스크별로 개별 학습된다. 멀티태스크 학습(multi-task learning) 또는 메타러닝(meta-learning) 프레임워크를 도입하면, 새로운 태스크에 대한 적응 속도와 성능을 향상시킬 수 있다.

> 💡 **메타러닝(Meta-Learning):** "학습하는 방법을 학습"하는 방법론. 적은 수의 새로운 샘플로도 빠르게 새 태스크에 적응할 수 있도록 설계됨.

**② Few-Shot 적응 가능성**

현재 SAM-Adapter는 각 태스크의 전체 훈련 데이터를 사용한다. 극소수 데이터(few-shot) 환경에서의 성능 검증이 없다. 프롬프트 튜닝(prompt tuning) 기법과 결합하면 데이터 효율성을 높일 수 있다:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \cdot \mathcal{L}_{reg}$$

여기서 $\mathcal{L}_{reg}$는 어댑터 가중치가 사전학습 특성에서 너무 멀리 벗어나지 않도록 하는 정규화 항.

> 💡 **프롬프트 튜닝(Prompt Tuning):** 모델 가중치를 변경하지 않고 입력 프롬프트(텍스트 또는 벡터)만을 학습하는 파라미터 효율적 방법.

**③ $F_i$ 자동화를 통한 일반화**

현재 $F_i = F_{hfc} + F_{pe}$는 수동 설계된 특징이다. 자동 특징 탐색(AutoML, Neural Architecture Search)을 통해 태스크별 최적 $F_i$ 조합을 자동으로 찾는다면 일반화 능력이 크게 향상될 것이다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 발표 | 방법 | 주요 특징 | SAM-Adapter와의 관계 |
|------|------|------|-----------|---------------------|
| **EVP** [4] (Liu et al., 2023) | 2023 | Explicit Visual Prompting | 저수준 구조 분할을 위한 명시적 시각 프롬프트 | SAM-Adapter의 $F_i$ 설계 직접 참조 |
| **ViT-Adapter** [35] (Chen et al., 2022) | 2022 | Adapter for ViT | 범용 ViT를 다운스트림 태스크에 적응 | SAM-Adapter의 어댑터 구조 설계 참조 |
| **SAM** [2] (Kirillov et al., 2023) | 2023 | 파운데이션 모델 | 범용 이미지 분할 | SAM-Adapter의 백본 |
| **MedSAM** (Ma et al., 2023) | 2023 | SAM 의료 파인튜닝 | 의료 영상 전체 파인튜닝 | SAM-Adapter와 유사한 목표, 다른 방법 |
| **SAM-Med2D** (Cheng et al., 2023) | 2023 | 어댑터 기반 의료 SAM | 의료 영상 어댑터 방식 | SAM-Adapter 이후 파생 연구 |
| **HQ-SAM** (Ke et al., 2023) | 2023 | High-Quality SAM | 고품질 마스크 생성을 위한 SAM 개선 | 상보적 접근: 디코더 개선 vs. 인코더 적응 |

> ⚠️ **주의:** MedSAM, SAM-Med2D, HQ-SAM은 SAM-Adapter 논문(2023.04)과 근접한 시기에 발표되어, 논문 내에서 직접 비교되지 않음. 위 비교는 필자의 분석이며, 정확한 수치 비교는 각 원본 논문 참조 필요.

---

**해당 논문이 앞으로의 연구에 미치는 영향:**

1. **파운데이션 모델 적응 패러다임 제시:** SAM을 동결하고 경량 모듈만 학습하는 방식이 이후 수많은 SAM 기반 연구(MedSAM, SAM-Med2D 등)의 설계 원칙이 됨
2. **저수준 구조 분할의 새 벤치마크:** SAM을 기준선(baseline)으로 활용하는 연구 방향 확립
3. **의료 AI와 파운데이션 모델 접목 가속화:** 소규모 의료 데이터로도 SOTA 달성 가능성 제시

---

**앞으로 연구 시 고려할 점:**

| 고려사항 | 설명 |
|----------|------|
| **어블레이션 설계** | 각 구성 요소($F_{hfc}$, $F_{pe}$, 어댑터 수 등)의 개별 기여도 검증 필수 |
| **통계적 엄밀성** | 다중 실행(multiple runs) 및 표준편차 보고, 통계적 유의성 검증 |
| **최신 비교군** | TransFuse, Polyp-PVT, HQ-SAM 등 2022-2023년 SOTA와의 공정한 비교 |
| **계산 효율성** | 파라미터 수, FLOPs, 추론 시간 등 실용적 지표 보고 |
| **$F_i$ 자동화** | 수동 설계된 특징에서 벗어난 학습 기반 특징 탐색 연구 |
| **대규모 사전실험** | SAM의 실패 사례를 더 체계적으로 분류하고 분석하는 진단 프레임워크 개발 |
| **윤리적 고려** | 의료 영상 적용 시 임상 검증 없는 과도한 성능 주장 자제 |

---

## 참고자료

- **주 논문:** Chen, T., Zhu, L., Ding, C., Cao, R., Wang, Y., Li, Z., Sun, L., Mao, P., & Zang, Y. (2023). "SAM Fails to Segment Anything? – SAM-Adapter: Adapting SAM in Underperformed Scenes: Camouflage, Shadow, Medical Image Segmentation, and More." *arXiv preprint arXiv:2304.09148v3*.
- **SAM 원본:** Kirillov, A. et al. (2023). "Segment Anything." *arXiv:2304.02643*.
- **EVP (Explicit Visual Prompting):** Liu, W. et al. (2023). "Explicit Visual Prompting for Low-Level Structure Segmentations." *arXiv:2303.10883*.
- **ViT-Adapter:** Chen, Z. et al. (2022). "Vision Transformer Adapter for Dense Predictions." *arXiv:2205.08534*.
- **Foundation Models 개요:** Bommasani, R. et al. (2021). "On the Opportunities and Risks of Foundation Models." *arXiv:2108.07258*.
- **GELU:** Hendrycks, D. & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)." *arXiv:1606.08415*.
- **COD10K 데이터셋:** Fan, D.P. et al. (2020). "Camouflaged Object Detection." *CVPR 2020*.
- **kvasir-SEG:** Jha, D. et al. (2020). "Kvasir-SEG: A Segmented Polyp Dataset." *MMM 2020*.
- **ISTD 데이터셋:** Wang, J. et al. (2018). "Stacked Conditional GAN for Shadow Detection and Removal." *CVPR 2018*.
- **프로젝트 페이지:** http://tianrun-chen.github.io/SAM-Adaptor/

> ⚠️ **정확도 고지:** MedSAM, SAM-Med2D, HQ-SAM 등 SAM-Adapter 이후 발표된 파생 연구들의 구체적 수치 비교는 해당 논문 원문을 직접 확인하시기 바랍니다. 본 보고서에서 해당 부분은 정확한 수치 비교 없이 방향성만 서술하였습니다.
