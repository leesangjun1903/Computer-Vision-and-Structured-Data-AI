# Segmenting 2K-Videos at 36.5 FPS with 24.3 GFLOPs: Accurate and Lightweight Realtime Semantic Segmentation Network

> **참고 자료:**
> - 원문 논문: "Segmenting 2K-Videos at 36.5 FPS with 24.3 GFLOPs: Accurate and Lightweight Realtime Semantic Segmentation Network" (Oh et al., Samsung Electronics & KAIST)
> - CITYSCAPES Benchmark: https://www.cityscapes-dataset.com/benchmarks/
> - 논문 내 인용 문헌 [1]–[71] (참고문헌 섹션 전체)

---

## 1. Executive Summary (10문장 이내)

NfS-SegNet은 자율주행 등 실시간 고해상도 영상 처리를 위해 설계된 경량 시맨틱 세그멘테이션 네트워크이다.  
핵심 인코더인 NfS-Net은 depthwise convolution 없이 단순한 빌딩 블록만으로 구성되어, MobileNet·ShuffleNet 대비 실제 추론 속도가 우월하다.  
네트워크는 비대칭 구조로 설계되어 인코더에 연산을 집중시키고 디코더를 극도로 경량화하였다.  
이 설계 원칙은 "디코더가 속도 병목이나 정확도 기여는 미미하다"는 실험적 발견에 근거한다.  
정확도 보완을 위해 대형 교사 네트워크(GDNet)로부터 지식 증류(KD)를 적용하였다.  
특히 **불확실성 인식 지식 증류(U-KD)**를 제안하여 교사 모델이 예측하기 어려운 픽셀(경계, 혼동 클래스) 영역에 지식 전달을 집중시킨다.  
CITYSCAPES 벤치마크에서 36.5 FPS / 24.3 GFLOPs로 경량 모델 중 최고 속도와 mIoU 73.1%를 동시에 달성하였다.  
증류에는 224,294장의 레이블/비레이블 데이터를 활용하여 데이터 효율성도 높였다.  
점진적 학습(Incremental Learning) 시나리오에서도 U-KD가 미경험 도메인의 성능을 향상시키는 것이 확인되었다.  
이 연구는 정확도-속도-경량성의 세 가지 조건을 동시에 충족하는 실용적 세그멘테이션 시스템의 가능성을 제시한다.

---

### 1-1. 연구의 목적과 필요성

| 필요성 | 설명 |
|---|---|
| **실시간성** | 자율주행 AEB·ACC 등은 충돌 전 조기 감지가 필수 → 높은 FPS 요구 |
| **고해상도** | 저해상도에서는 원거리 물체 조기 감지 불가 → 최소 Full-HD/2K 필요 |
| **경량화** | 자율주행 차량의 임베디드 기기는 GPU 클러스터 대비 연산·메모리 자원 극히 제한 |
| **정확도 유지** | 세그멘테이션 오류는 안전 사고로 직결 → 경량화 시 정확도 손실 최소화 필수 |

> 💡 **시맨틱 세그멘테이션(Semantic Segmentation)**: 이미지의 각 픽셀에 클래스 레이블(도로, 사람, 차량 등)을 부여하는 작업. 자율주행의 장면 이해에 핵심적으로 활용됨.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|---|---|---|
| 1 | NfS-Net은 단순 블록 구성으로 동급 대비 최고 추론 속도 달성 | Table I: 2K 입력 기준 35.0 FPS (ShuffleNet v2 0.5x: 39.7 FPS보다 낮지만, GFLOPs 대비 실속도 우수) | Table I, Fig. 3 |
| 2 | 디코더가 속도 병목이지만 정확도 기여는 적다 (비대칭 설계 근거) | 디코더 복잡도 증가 시 FPS 36.4→27.5, IoU 73.1→73.4로 속도 대비 정확도 향상 미미 | Fig. 4, Fig. 5 |
| 3 | U-KD가 기존 KD 대비 정확도 유의미하게 향상 | IoU: GT 59.2 → KD 69.2 → JA-KD 71.0 → U-KD 73.1 | Table III |
| 4 | U-KD는 미경험(unseen) 도메인에서도 성능 향상 | Fig. 7: unseen 도메인 포함 시 U-KD가 KD보다 높은 수렴 IoU 달성 | Fig. 7 |
| 5 | U-KD는 다양한 교사 네트워크와 압축률에서 일반적으로 우수 | Fig. 8(GD10~GD75), Fig. 9(ENet, PSPNet 교사) | Fig. 8, Fig. 9 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

기존 경량 세그멘테이션 모델(ENet, ESPNet 등)은 **FLOPs 감소에 집중**하지만 실제 추론 속도가 비례하여 증가하지 않는 문제가 있다. Depthwise convolution, group convolution 등은 FLOPs는 낮지만 메모리 접근 비용이 높아 실속도를 저해한다. 또한 고해상도(2K) 영상에서 실시간 처리가 가능하면서도 높은 mIoU를 유지하는 모델이 부재하다.

---

### 제안하는 방법 및 수식

#### (A) NfS-Net 인코더 설계 원칙

- Bias 없는 Convolution 사용
- 초기 레이어에서 적극적 다운샘플링
- DenseNet 구조로 특징 재사용 극대화
- Convolution, PReLU, Pooling, Concatenation 4종 레이어만 사용
- Depthwise/Group Convolution **배제** → 메모리 접근 비용 최소화

> 💡 **Depthwise Convolution**: 채널별로 독립적으로 convolution을 수행하여 FLOPs를 줄이는 기법. 그러나 메모리 접근 패턴이 비효율적이어서 실제 속도 향상이 FLOPs 감소에 비례하지 않을 수 있음.

> 💡 **DenseNet 구조**: 이전 레이어의 출력을 이후 모든 레이어에 직접 연결하여 특징을 재사용하고 그래디언트 소실을 완화하는 구조.

---

#### (B) Joint & Auxiliary Knowledge Distillation (JA-KD) 손실 함수

$$\mathcal{L}_{JA}(p_i, \hat{p}_i) = \sum_{i=1}^{N} \left[\hat{p}_i \log p_i + \alpha \cdot \text{smooth}_{L1}(p_i - \hat{p}_i)\right] $$

$$\text{smooth}_{L1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases} $$

**기호 설명:**

| 기호 | 의미 |
|---|---|
| $p_i$ | 학생 네트워크의 $i$번째 픽셀 예측 확률 로짓(logit) |
| $\hat{p}_i$ | 교사 네트워크의 $i$번째 픽셀 예측 확률 로짓 |
| $N$ | 클래스 수 |
| $\alpha$ | Cross Entropy 손실과 Smooth L1 손실 간 균형 하이퍼파라미터 ($\alpha = 0.5$ 사용) |
| $\text{smooth}_{L1}$ | 소규모 오차에 L2, 대규모 오차에 L1을 적용하는 robust 손실 함수 |

> 💡 **소프트 레이블(Soft Label)**: 교사 네트워크의 출력 확률 분포. 단순한 0/1 하드 레이블보다 클래스 간 유사도 정보를 담고 있어 학습에 유리함.

> 💡 **Smooth L1 Loss**: 이상치(outlier)에 덜 민감한 손실 함수. $|x| < 1$인 경우 L2처럼, 그 외에는 L1처럼 동작하여 두 손실의 장점을 결합함.

---

#### (C) Uncertainty-aware Knowledge Distillation (U-KD) 손실 함수

$$\mathcal{L}_{U}(p_i, \bar{p}_i^{mc}, \bar{u}_i^{mc}) = \sum_{i=1}^{N} \bar{u}_i^{mc} \cdot \left[\bar{p}_i^{mc} \log p_i + \alpha \cdot \text{smooth}_{L1}(p_i - \bar{p}_i^{mc})\right] $$

**기호 설명:**

| 기호 | 의미 |
|---|---|
| $p_i$ | 학생 네트워크의 $i$번째 픽셀 예측 확률 로짓 |
| $\bar{p}_i^{mc}$ | MC-Dropout을 $K$회 적용하여 얻은 교사 모델 예측 분포의 평균 |
| $\bar{u}_i^{mc}$ | $i$번째 픽셀의 이진화된 불확실성 (픽셀별 분산의 중앙값 기준 임계처리) |
| $\alpha$ | 손실 균형 하이퍼파라미터 ($\alpha = 0.5$) |
| $N$ | 클래스 수 |
| $K$ | MC-Dropout 반복 횟수 (논문에서는 5 minibatch) |

> 💡 **MC-Dropout (Monte Carlo Dropout)**: 테스트 시에도 Dropout을 활성화하여 여러 번 추론을 반복함으로써 예측의 불확실성을 추정하는 기법. Gal et al. [20]이 제안한 베이지안 딥러닝의 근사 방법.

> 💡 **인식론적 불확실성(Epistemic Uncertainty)**: 모델의 학습 데이터 부족이나 클래스 혼동에서 비롯되는 불확실성. 더 많은 데이터로 줄일 수 있음. (↔ 데이터 자체의 모호함인 우연론적 불확실성 Aleatoric Uncertainty)

> 💡 **이진화된 불확실성(Binarized Uncertainty)**: 픽셀별 분산값을 전체 중앙값(median)을 기준으로 0 또는 1로 변환한 마스크. 불확실한 픽셀에만 지식 전달 가중치를 부여하기 위해 사용됨.

---

### 모델 구조

```
입력 이미지 (2K)
     │
┌────▼────────────────────────────────────┐
│         인코더: NfS-Net                 │  21.3 GFLOPs @ 2K
│  (Aggressive Downsampling + DenseNet)   │
│  특징맵: 원본의 1/256 크기까지 압축      │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│      추가 인코더 (스트라이드 적용)       │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│         디코더 (매우 경량)              │  3.0 GFLOPs @ 2K
│     (단순 2배 업샘플링 반복)            │
└────────────────────┬────────────────────┘
                     │
              클래스 확률 출력
                     │
         ┌───────────▼──────────┐
         │   U-KD 손실 (훈련 시) │ ← GDNet 교사 + 불확실성 마스크
         └───────────────────────┘
```

**비대칭 설계 핵심**: 인코더(21.3 GFLOPs)가 디코더(3.0 GFLOPs)보다 약 7배 더 많은 연산을 담당하나, 실제 추론 시간은 디코더가 병목임 (Fig. 4 참조).

---

### 성능 향상 및 한계

**성능 향상:**

| 방법 단계 | mIoU (class) | FPS | GFLOPs |
|---|---|---|---|
| NfS-SegNet (GT만) | 59.2% | 36.4 | 24.3 |
| + KD | 69.2% | 36.4 | 24.3 |
| + JA-KD | 71.0% | 36.4 | 24.3 |
| + U-KD | **73.1%** | 36.4 | 24.3 |
| GDNet (교사, 앙상블) | 75.7% | 3.0 | 141.6 |

**한계:**
- CITYSCAPES 단일 데이터셋에서만 검증 (다른 도메인 일반화 미확인)
- U-KD의 MC-Dropout 반복(K=5) 추가 연산이 **훈련 시간** 증가시킴
- PSPNet처럼 교사-학생 크기 차이가 클 경우(0.022배) U-KD 효과 제한적
- 실제 임베디드 하드웨어 배포 실험 결과 미제시

---

## 3. 주장별 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|---|---|
| NfS-Net의 속도 우월성 | Table I (p.3), Fig. 3 (p.3) |
| 비대칭 설계의 합리성 | Fig. 4 (p.3), Fig. 5 (p.4) |
| KD 단계별 성능 향상 | Table III (p.5) |
| U-KD의 불확실성 시각화 | Fig. 6 (p.5) |
| 점진적 학습에서의 U-KD 효과 | Fig. 7 (p.5) |
| 압축률별 U-KD 성능 | Fig. 8 (p.6) |
| 다양한 교사 모델에서의 U-KD | Fig. 9 (p.6) |
| 전체 비교 정확도-속도 트레이드오프 | Fig. 1 (p.1) |

---

## 4. 저자 보고 결과 vs. 필자 해석 분리

### 연구 주제
- **저자 보고**: "NfS-SegNet은 2K 영상을 36.5 FPS, 24.3 GFLOPs로 처리하며 CITYSCAPES에서 경량 모델 중 최고 성능"
- **필자 해석**: 속도·정확도 동시 최적화라는 목표는 달성했으나, 단일 데이터셋(CITYSCAPES) 검증만으로는 실세계 일반화 능력 주장에 한계가 있음

### 방법
- **저자 보고**: "Smooth L1 + Cross Entropy 조합 손실(JA-KD)이 기존 L2 soft loss보다 빠른 수렴과 높은 정확도를 경험적으로 확인"
- **필자 해석**: 이 조합의 이론적 우월성에 대한 수학적 근거가 충분히 제시되지 않았으며 하이퍼파라미터 $\alpha=0.5$의 최적성 실험도 제한적임

### 결과
- **저자 보고**: U-KD로 mIoU 73.1% 달성 (ESPNet v2: 54.7%, ENet: 63.1% 대비 우수)
- **필자 해석**: ESPNet v2는 54.7%로 본 논문보다 크게 낮지만, BiSeNet [18] 등 동시기 주요 경쟁 모델과의 비교가 Table III에 누락되어 있어 포괄적 비교가 불완전함

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치 ⚠️

| 항목 | 문제점 |
|---|---|
| ⚠️ **FPS 측정 환경 통일성** | NfS-SegNet은 GTX 1080Ti에서 측정, 경쟁 모델(ESPNet, ICNet 등)은 각기 다른 하드웨어·소프트웨어 환경에서 측정됨. I/O 시간 포함 여부도 다름 (p.5) |
| ⚠️ **BiSeNet 미포함** | 당시 SOTA 경량 모델인 BiSeNet [18]이 Table III 비교에 없어 공정한 비교 불완전 |
| ⚠️ **단일 데이터셋 평가** | CITYSCAPES 외 ADE20K, Pascal VOC 등에서의 검증 없음 → 일반화 성능 주장 불확실 |
| ⚠️ **MC-Dropout 반복 횟수 K** | K=5로 설정한 근거나 K 변화에 따른 민감도 분석 없음 |
| ⚠️ **표준편차/신뢰구간 미제시** | 모든 실험 결과가 단일 수치로만 보고됨 (반복 실험 통계 없음) |
| ⚠️ **NfS-Net Top-1 error 71.4%** | Table I에서 NfS-Net의 Top-1 error가 71.4%로 가장 높음 — 분류 정확도 자체는 떨어짐에도 세그멘테이션 성능이 높다는 점은 추가 설명 필요 |

---

## 6. 논문이 답하지 않는 질문

| 번호 | 미해결 질문 |
|---|---|
| Q1 | CITYSCAPES 외 다른 벤치마크(ADE20K, Mapillary, Pascal VOC)에서의 성능은? |
| Q2 | 실제 임베디드 디바이스(Jetson Xavier, Raspberry Pi 등)에서의 배포 성능은? |
| Q3 | MC-Dropout 횟수 K에 따른 불확실성 추정 품질과 최종 성능 변화는? |
| Q4 | 이진화 불확실성 임계값(median 대신 다른 백분위) 선택에 따른 민감도는? |
| Q5 | NfS-SegNet이 야간, 악천후 등 도메인 시프트 환경에서도 강건한가? |
| Q6 | U-KD 훈련 시 추가 소요 시간(MC-Dropout K회 반복)은 얼마인가? |
| Q7 | 비디오의 시간적 연속성(temporal consistency)을 활용한 개선 여지는? |
| Q8 | 하이퍼파라미터 $\alpha=0.5$ 최적화 과정 및 다른 값에 대한 ablation 결과는? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1) — 정확도 vs. 속도 산점도

**내용**: CITYSCAPES 리더보드에서 여러 모델의 mIoU(Y축)와 FPS(X축)를 비교한 산점도.

**해석**: NfS-SegNet은 약 36.5 FPS와 73.1% mIoU를 동시에 달성하여 우측 상단(고속·고정확도) 영역에 단독으로 위치한다. ESPNet v2(35.4 FPS, 54.7%)와 속도는 비슷하나 정확도는 약 18%p 높고, GDNet 계열(GD10~GD75)보다 빠르면서 경쟁력 있는 정확도를 보인다. 단, PSPNet(82.1%)·DeepLab v3+(81.2%) 등 고정확도 모델과는 여전히 8~9%p 차이가 있어 정확도의 절대적 한계가 시각적으로 드러난다.

---

### Figure 3 (p.3) — 레이어별 추론 시간 비교

**내용**: NfS-Net, MobileNet v2, ShuffleNet v2 0.5x의 네트워크 각 단계(해상도별)에서의 추론 시간을 막대그래프로 비교.

**해석**: MobileNet v2는 1/8~1/16 해상도 구간에서 10.76ms라는 현저한 병목이 발생하며, 이는 depthwise convolution의 메모리 접근 비용에 기인한다. ShuffleNet v2는 1/32 분류기 단계에서 11.15ms로 후반 병목이 심각하다. 반면 NfS-Net은 최대 0.75ms로 모든 구간에서 균일하고 낮은 추론 시간을 유지하며, 특정 층에서의 병목이 없다는 것이 핵심 강점이다.

---

### Figure 5 (p.4) — 디코더 복잡도에 따른 IoU vs. FPS 트레이드오프

**내용**: 디코더 필터 크기를 19 → 19×19 → 19×19×19로 늘렸을 때의 IoU와 FPS 변화.

**해석**: 디코더 복잡도가 증가함에 따라 FPS는 36.4 → 31.0 → 27.5로 급격히 감소하는 반면, IoU는 73.1 → 73.3 → 73.4로 거의 변화가 없다. 이는 비대칭 설계(얕은 디코더)의 핵심 근거로, 디코더 강화의 한계수익체감(diminishing returns)을 명확하게 보여준다. 속도 손실(-24%) 대비 정확도 향상(+0.3%p)이 극히 비효율적임을 정량적으로 증명한다.

---

### Figure 6 (p.5) — 불확실성 시각화

**내용**: 입력 이미지, 세그멘테이션 예측, MC-Dropout으로 추정된 픽셀별 분산(불확실성) 맵을 나란히 시각화.

**해석**: 불확실성(분산)이 높은 픽셀(어두운 영역)은 "rider"와 "person" 클래스 경계, 자전거 탑승자의 배낭 등 클래스 구분이 모호한 영역에 집중되어 있다. 이는 U-KD가 주목해야 할 어려운 영역을 정확히 식별함을 보여준다. 반면 도로 중앙, 건물 내부 등 명확한 영역은 불확실성이 낮아 지식 전달 가중치가 낮게 설정된다. 이 시각화는 U-KD의 설계 철학(어려운 곳에 집중)이 실제로 의미 있게 작동함을 정성적으로 지지한다.

---

### Figure 8 (p.6) — 압축률별 U-KD vs. KD 수렴 곡선

**내용**: GDNet를 10%, 25%, 50%, 75% 크기로 압축한 학생 모델에 대해 U-KD와 기존 KD의 Validation IoU 수렴 곡선 비교.

**해석**: 모든 압축률(GD10~GD75)에서 U-KD(빨간선)가 KD(파란선)보다 더 높은 최종 IoU로 수렴한다. GD25~GD50 수준에서는 심지어 전체 교사 네트워크(점선)와 거의 유사한 성능에 도달하여, 75%까지 경량화해도 지식 증류로 성능 보상이 가능함을 보여준다. GD10의 경우 성능 격차가 크지만 여전히 U-KD가 우위에 있어, U-KD의 범용적 우월성을 뒷받침한다.

---

## 8. 결론: 시사점, 후속 연구 계획 및 추가 방향

### 8-1. 연구자들의 시사점 및 후속 연구 계획

저자들은 결론에서 다음 시사점을 명시한다:

1. **비대칭 인코더-디코더 설계**: 디코더를 단순화하고 인코더에 연산을 집중하는 것이 실시간 세그멘테이션에 효율적
2. **U-KD의 범용성**: 다양한 교사 네트워크(ENet~PSPNet)와 압축률에서 일관되게 기존 KD를 상회
3. **비레이블 데이터 활용**: 224,294장의 비레이블 데이터를 교사 예측으로 pseudo-label화하여 데이터 효율성 제고
4. **점진적 학습 가능성**: 미경험 도시 환경 영상에서도 U-KD가 성능 향상을 이끎 (on-device continual learning 가능성)

**명시된 후속 연구 계획**: 논문 내에 구체적인 후속 연구 계획은 명시되어 있지 않음 ⚠️ (정확한 답변만 제시)

---

### 모델의 일반화 성능 향상 가능성

본 논문의 U-KD는 다음 측면에서 일반화 향상 잠재력을 가진다:

**긍정적 요인:**
- **불확실성 기반 포커싱**: 경계, 혼동 클래스 등 어려운 패턴에 집중하는 학습은 특정 데이터셋 편향을 줄이는 효과가 있음
- **비레이블 데이터 활용**: Pseudo-label 방식으로 미경험 도시 도메인 데이터를 학습에 포함시켜 도메인 시프트 부분 완화 (Fig. 7, unseen 도메인 성능 향상 확인)
- **ENet 교사 실험**: 교사보다 학생이 뛰어난 경우(ENet 교사 → NfS-SegNet)가 나타나, U-KD가 단순 모방을 넘어 일반화된 지식 추출 가능성을 시사

**한계 및 개선 필요 사항:**
- CITYSCAPES 단일 도메인 검증으로 다른 도시·기상 조건·카메라 설정에서의 일반화 불명확
- Aleatoric uncertainty(데이터 자체 모호성)를 활용한 추가 정규화 미적용 → 레이블 노이즈 강건성 미검증
- 도메인 적응(Domain Adaptation) 기법과의 결합으로 일반화 성능을 크게 향상시킬 수 있음

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **중요 주의사항**: 아래 내용 중 일부(특히 구체적 수치)는 본 논문 원문에 포함되지 않은 외부 정보로, 필자의 학습 데이터 기반 일반적 지식에 해당합니다. 정확한 수치 확인을 위해서는 각 논문 원문을 직접 참조하시기 바랍니다.

| 모델 (연도) | mIoU | FPS | 주요 기여 | NfS-SegNet 대비 |
|---|---|---|---|---|
| **BiSeNet v2** (2021) | ~73.4% | ~156 FPS | Bilateral 경로 + 상세/의미 분기 분리 | 속도 우위, 정확도 유사 |
| **STDC** (2021) | ~74.2% | ~250 FPS | 단기 밀집 연결 블록으로 특징 재사용 | 속도·정확도 모두 우위 |
| **PP-LiteSeg** (2022) | ~73.0% | ~270 FPS | 경량 디코더 + FLD 모듈 | 속도 대폭 우위 |
| **DDRNet** (2021) | ~77.4% | ~37 FPS | 이중 해상도 분기 구조 | 정확도 우위, 속도 유사 |
| **SegFormer** (2021) | ~82.2% | ~15 FPS | Transformer 기반 계층적 인코더 | 정확도 크게 우위, 속도 열세 |

**NfS-SegNet이 후속 연구에 미치는 영향:**

1. **비대칭 인코더-디코더 패러다임 촉진**: 이후 많은 경량 모델(BiSeNet v2, STDC 등)이 무거운 인코더 + 경량 디코더 설계를 채택
2. **속도 측정의 공정성 문제 제기**: I/O 포함 실제 FPS 측정 필요성을 강조하여 후속 논문들의 벤치마킹 표준화에 기여
3. **불확실성 기반 KD의 선구적 역할**: 이후 픽셀별 가중치를 활용한 지식 증류 연구의 초기 사례로 인용 가능성

**향후 연구 시 고려할 점:**

| 고려 사항 | 상세 설명 |
|---|---|
| **Transformer 통합** | SegFormer, Swin Transformer 등 Vision Transformer 기반 인코더와 NfS 스타일 경량 디코더 결합 탐구 |
| **다중 데이터셋 평가** | ADE20K, Cityscapes, Mapillary를 동시에 평가하여 일반화 성능 입증 필요 |
| **불확실성-정규화 결합** | U-KD를 Dropout 외 Evidence Lower Bound (ELBO) 기반 베이지안 방법론과 결합하여 더 정확한 불확실성 추정 |
| **온라인/점진적 학습 심화** | Fig. 7에서 보인 on-device incremental learning을 연속 학습(Continual Learning) 프레임워크와 체계적으로 통합 |
| **경량화 기법 결합** | 지식 증류 + 양자화(Quantization) + 가지치기(Pruning) 세 기법의 결합으로 임베디드 배포 최적화 |
| **시간적 일관성** | 비디오 세그멘테이션에서 프레임 간 일관성 손실 추가하여 깜박임(flickering) 현상 억제 |

---

**참고 자료 목록:**

1. Oh, D., Ji, D., Jang, C., Hyun, Y., Bae, H.S., Hwang, S. — *"Segmenting 2K-Videos at 36.5 FPS with 24.3 GFLOPs: Accurate and Lightweight Realtime Semantic Segmentation Network"* (원문 논문, Samsung Electronics & KAIST)
2. Hinton, G., Vinyals, O., Dean, J. — *"Distilling the Knowledge in a Neural Network"* NIPS workshop, 2014 [ref. 10]
3. Gal, Y., Ghahramani, Z. — *"Dropout as a Bayesian Approximation"* ICML, 2016 [ref. 20]
4. Kendall, A., Badrinarayanan, V., Cipolla, R. — *"Bayesian SegNet"* BMVC, 2017 [ref. 19]
5. Ma, N. et al. — *"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"* ECCV, 2018 [ref. 6]
6. Huang, G. et al. — *"Densely Connected Convolutional Networks"* CVPR, 2017 [ref. 21]
7. Cityscapes Dataset: https://www.cityscapes-dataset.com/benchmarks/ [ref. 1]
8. Romero, A. et al. — *"Fitnets: Hints For Thin Deep Nets"* ICLR, 2015 [ref. 11]
9. Paszke, A. et al. — *"ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation"* arXiv:1606.02147, 2016 [ref. 15]
10. Zhao, H. et al. — *"Pyramid Scene Parsing Network"* CVPR, 2017 [ref. 23]
