
# BatchNorm-based Weakly Supervised Video Anomaly Detection

## 1. 논문 개요 및 핵심 주장

### 1.1 기본 정보
"BatchNorm-based Weakly Supervised Video Anomaly Detection (BN-WVAD)"는 2023년 11월 게시된 컴퓨터 비전 논문으로, 동전자 과학기술 대학 연구팀이 약한 감독(비디오 수준 레이블만 사용) 조건에서 비디오 이상탐지의 SOTA(State-of-the-Art) 성능을 달성했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

### 1.2 핵심 기여도

**세 가지 주요 한계점 해결:**

첫째, 기존 방법들은 특성 크기(feature magnitude) 또는 블랙박스 모델에 의존하여 신뢰도가 낮은 이상성 기준을 사용했다. 두 번째로, 고정된 top-k 선택 전략은 비디오 내 이상비율(abnormality ratio)의 변동성을 무시하여, 높은 이상비율의 비디오에서 중요한 스니펫을 놓쳤다. 세 번째로, 잘못 선택된 이상 스니펫으로 인한 라벨 노이즈에 분류기가 민감했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

**제안하는 혁신:**

본 논문은 배치정규화(BatchNorm)의 평균 벡터를 통계적 정규성 참조로 활용하는 근본적인 통찰을 제시했다. 이를 기반으로 세 가지 핵심 기여를 제안했다:

1. **DFM(Divergence of Feature from Mean) 기준**: 마할라노비스 거리를 사용하여 이상 스니펫의 신뢰도 높은 식별
2. **Mean-based Pull-Push(MPP) Loss**: 정상 특성을 가깝게, 이상 특성을 멀게 하는 최적화
3. **Sample-Batch Selection(SBS) 전략**: 표본 수준과 배치 수준 선택의 강점을 결합하여 고이상비율 비디오에서도 정확한 선택

***

## 2. 제안 방법론 상세 분석

### 2.1 BatchNorm의 통계적 역할

배치정규화는 일반적으로 학습 안정성 개선으로 알려져 있으나, 본 논문은 정규성 모델링의 숨은 가치를 발견했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

B개 비디오의 미니배치에서 T개 스니펫의 숨겨진 특성 $X^h \in \mathbb{R}^{B \times T \times C}$에 대해 배치정규화의 평균 벡터는:

$$\mu = E[X^h] = \frac{1}{BT} \sum_{b=1}^{B} \sum_{t=1}^{T} X^h_{b,t}$$

실제 구현에서는 지수이동평균(EMA)으로 운행 통계를 유지한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

$$\mu \leftarrow \alpha \mu + (1-\alpha) \hat{\mu}$$
$$\sigma^2 \leftarrow \alpha \sigma^2 + (1-\alpha) \hat{\sigma}^2$$

여기서 $\alpha = 0.1$이 PyTorch 기본값이며, 운행 통계는 중심극한정리(CLT)에 의해 정규분포를 따르므로, 이상 스니펫은 평균으로부터 이상치(outlier)로 나타난다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

### 2.2 DFM 기준의 수학적 정식화

이상성을 측정하는 핵심 메트릭은 마할라노비스 거리를 기반으로 한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

$$\text{DFM}(X^h_{b,t}, \mu, \Sigma^2) = (X^h_{b,t} - \mu)^T \Sigma^{-1} (X^h_{b,t} - \mu)$$

여기서 공분산 행렬은 대각 행렬 $\Sigma = \text{diag}(\sigma^2)$로 근사한다. 마할라노비스 거리는 다변량 정규분포에서 특성의 비등방성 분포를 고려하여, 유클리드 거리나 코사인 유사성보다 우수한 성능을 보인다. 실험 결과, 유클리드 거리는 UCF-Crime에서 86.51% AUC를 기록했고 코사인 유사성은 85.33% AUC로 떨어졌으나, 마할라노비스 거리는 87.24% AUC를 달성했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

### 2.3 Mean-based Pull-Push Loss

정상과 비정상 특성의 분리를 강화하기 위해, DFM 값이 큰 K개의 가능한 이상 특성 $X^{dfm}\_{a} \in \mathbb{R}^{K \times C}$와 정상 특성 $X^{dfm}_{n} \in \mathbb{R}^{K \times C}$에 대해 트리플렛 손실 기반의 MPP 손실을 정의한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

$$L_{mpp}(X^{dfm}\_{n}, X^{dfm}_{a}, \mu, \Sigma^2) = \frac{1}{K} \sum_{k=1}^{K} \left[ m - \text{DFM}(X^{dfm}\_{n,k}, \mu, \Sigma^2) + \text{DFM}(X^{dfm}_{a,k}, \mu, \Sigma^2) \right]$$

마진 $m = 1$로 설정하여, 정상 특성의 DFM은 감소하고 이상 특성의 DFM은 증가하도록 유도한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

### 2.4 Sample-Batch Selection 전략

기존 표본 수준 선택(SLS)은 각 비디오 $V_i$에서 상위 $\rho_s \times |V_i|$개 스니펫을 선택하지만, 이는 고이상비율 비디오에서 중요한 스니펫을 놓친다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf) 

반대로 제안하는 배치 수준 선택(BLS)은 전체 배치의 이상비율이 상대적으로 안정적이라는 가정 하에, 미니배치 전체에서 상위 $\rho_b \times |B|$개 스니펫을 선택한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf) 그림 4에서 보듯이, $\rho_s = 0.4$, $\rho_b = 0.4$일 때:

- **SLS**: 4번째 비디오의 DFM 점수 0.8, 0.7인 스니펫을 놓침
- **BLS**: 4번째 비디오의 높은 점수 스니펫을 포착하나, 1번째 비디오의 점수 0.3, 0.4인 어려운 스니펫을 놓침
- **SBS (합집합)**: 모든 영상에서 최적의 스니펫을 선택 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

최종 이상 점수는 분류기 예측과 DFM의 곱으로 계산된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

$$\text{Score} = C(\text{ReLU}(\text{BN}(X^h))) \odot \text{DFM}(X^h, \mu, \Sigma^2)$$

### 2.5 전체 손실 함수

정상 비디오의 모든 스니펫을 감독하는 정상 손실: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

$$L_{nor}(X^n) = \left\| C(\text{ReLU}(\text{BN}(X^n_{b,2}))) \right\|_2$$

전체 최적화 목표는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

$$L = L_{nor} + \lambda_1 L_{mpp}^{(1)} + \lambda_2 L_{mpp}^{(2)}$$

여기서 $\lambda_1 = 5$, $\lambda_2 = 20$으로 설정되어, $L_{nor}$의 기본 가중치는 1, $L_{mpp}$ 항들은 더 높은 가중치로 조정된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

***

## 3. 모델 아키텍처 및 구현

### 3.1 전체 프레임워크

모델은 다음과 같은 단계로 구성된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

1. **입력 인코딩**: I3D를 사용하여 비디오를 200개 스니펫의 특성 벡터로 추출 (동결된 백본)
2. **특성 강화**: Transformer 기반 강화기로 512차원 출력 $X^e$ 생성
3. **숨겨진 특성 생성**: 두 개의 Conv1d 층 (커널 크기 1, 출력 차원 32와 16)으로 $X^h_1$, $X^h_2$ 생성
4. **배치정규화**: 각 특성에 배치정규화와 ReLU 활성화 적용
5. **선택 및 손실 계산**: SBS 전략으로 스니펫 선택 후 MPP 손실 계산
6. **이상 점수**: 최종 점수는 두 숨겨진 특성의 DFM 값을 합산

### 3.2 선택 전략의 작동 메커니즘

**표본 수준 선택 (SLS)**: 각 비디오 $i$에서 DFM 점수가 높은 상위 스니펫을 선택
- 정상 비디오: $n_s = \rho_s \times T$개 선택
- 이상 비디오: $a_s = \rho_s \times T$개 선택

**배치 수준 선택 (BLS)**: 미니배치의 모든 이상 비디오에서 전체적으로 선택
- 배치 전체 선택 수: $n_b = \rho_b \times B \times T$개
- 각 비디오별 할당량: 유동적 조정

**통합 전략**: $\text{SBS} = \text{SLS} \cup \text{BLS}$로, 최종 선택된 스니펫은 두 전략의 합집합이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

***

## 4. 성능 평가 및 성능 향상

### 4.1 벤치마크 데이터셋 성능

| 데이터셋 | 메트릭 | BN-WVAD | 이전 SOTA | 향상도 |
|---------|--------|---------|----------|--------|
| UCF-Crime | AUC | **87.24%** | 86.97% (UR-DMU) | +0.27% |
| XD-Violence | AP (비디오) | **84.93%** | 83.59% (SAS) | +1.34% |
| XD-Violence | AP (오디오-시각) | **85.26%** | 83.40% (MACIL-SD) | +1.86% |
| ShanghaiTech | AUC | **97.61%** | 97.48% (S3R) | +0.13% |

특히 XD-Violence에서 큰 폭의 개선을 달성했는데, 이는 높은 이상비율 분포(최대 77.7%)를 가진 데이터셋에서 SBS 전략의 효과가 더 두드러지기 때문이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

### 4.2 절제 연구(Ablation Study)

| 모듈 구성 | UCF-Crime AUC | XD-Violence AP | 설명 |
|----------|--------------|----------------|------|
| Dropout만 | 65.21% | 61.54% | 기본 기준선 (배치정규화 없음) |
| BatchNorm | 82.97% | 72.99% | 정상 손실만 사용 |
| + DFM + MPP | 86.44% | 83.33% | 추가 이상성 기준 |
| + BLS | **87.24%** | **84.93%** | 최종 SOTA |

절제 연구는 각 구성 요소의 중요성을 증명한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

1. **배치정규화의 역할**: Dropout 대체 시 17.76% AUC 격차 (SOTA 기준)
2. **DFM의 필요성**: DFM 제외 시 1.4% AUC 감소 (RTFM 내에서)
3. **SBS 전략**: BLS 단독 사용 시 XD-Violence에서 3.68% AP 감소

### 4.3 손실 항 분석

| 손실 구성 | UCF-Crime AUC | XD-Violence AP |
|----------|--------------|----------------|
| $L_{nor}$만 | 83.0% | 73.0% |
| $L_{nor}$ + $L_{mpp}$ | 85.6% | 81.6% |
| 최종 (두 숨겨진 특성 포함) | **87.24%** | **84.93%** |

정상 손실($L_{nor}$)의 기여도는 초기 3.6% AUC이나, MPP 손실의 통합으로 추가 1.6~3.3% 향상을 이뤄낸다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

***

## 5. 모델의 일반화 성능

### 5.1 일반화 강점

**1. 통계적 적응성**: BatchNorm의 EMA 기반 운행 통계는 장기 정규성 분포를 학습하여, 훈련과 테스트 간 불일치를 자동으로 보정한다. 모멘텀 $\alpha = 0.1$일 때 최적 성능을 보이며, 이는 PyTorch 기본값으로 일반성이 높다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

**2. 다중 데이터셋 일관성**: UCF-Crime(감시 영상), XD-Violence(영화+감시), ShanghaiTech(거리 감시) 세 다양한 소스의 데이터셋에서 모두 SOTA를 달성했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

**3. 라벨 노이즈 강건성**: DFM 기준은 밀집 특성 공간에서 획득되므로, 논문에 따르면 잘못된 선택에 더 강건하다. 실제로 분류기를 정상 스니펫으로만 훈련했을 때도 높은 성능을 유지했다. [arxiv](https://arxiv.org/pdf/2305.18798.pdf)

**4. 모듈 재사용성**: DFM과 SBS 전략을 RTFM에 적용했을 때: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)
- UCF-Crime: +0.63% AUC (86.21% → 86.84%)
- XD-Violence: +2.52% AP (82.62% → 85.14%)

### 5.2 일반화 한계

**1. 소규모 데이터셋에 민감**: ShanghaiTech(238 훈련 비디오)에서 S3R과의 격차가 단 0.13% AUC로, UCF-Crime의 0.27%, XD-Violence의 1.34%와 비교할 때 개선 폭이 현저히 낮다. 이는 배치 통계가 제한된 훈련 데이터에 과적합될 위험을 시사한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)

**2. 배치 크기 의존성**: 최적 배치 크기는 (정상 64개, 이상 64개)으로 고정되어야 한다. 배치를 절반으로 줄일 때: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)
- UCF-Crime: 87.24% → 86.1% (1.14% 감소)
- XD-Violence: 84.93% → 83.0% (1.93% 감소)

**3. 모멘텀 파라미터 민감성**: 모멘텀이 0.1에서 벗어나면 급격한 성능 저하: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)
- $\alpha = 0.2$: 86.98% AUC (0.26% 감소)
- $\alpha = 0.5$: 84.69% AUC (2.55% 감소)
- $\alpha = 1.0$: 81.64% AUC (5.60% 감소, 운행 통계 미업데이트로 과적합)

**4. BLS의 저이상비율 약점**: 논문 4.4절에서 보듯이, BLS는 이상비율이 낮은 비디오의 미묘한 이상을 놓친다. 예를 들어: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/34120dbd-2b72-4136-ae25-33c0ab653c19/2311.15367v1.pdf)
- Abuse (이상비율 낮음): SLS 43.50% AP > BLS 30.71% AP
- Riot (이상비율 높음): BLS 92.17% AP > SLS 95.69% AP (SLS가 더 강함)

**5. 계산 복잡도**: 다중 crops (UCF: 10개, XD-Violence: 5개)를 사용하는 테스트 시간 증가로, 원-샷 예측보다 훨씬 느리다.

***

## 6. 2020년 이후 관련 최신 연구 비교 분석

### 6.1 연구 발전 타임라인

| 연도 | 방법 | UCF-Crime AUC | 핵심 혁신 |
|------|------|--------------|---------|
| 2020 | HL-Net | 82.44% | 계층적 정상성 학습 |
| 2021 | RTFM | 84.30% | 견고한 시간 특성 크기 |
| 2022 | MSL, S3R | 85.30%-85.99% | 자기감독 + 희소 표현 |
| 2023 | BN-WVAD | **87.24%** | BatchNorm 통계 기반 |
| 2024 | REWARD | 86.94% | 실시간 탐지 (6.4초) |
| 2024 | TPWNG | - | CLIP 텍스트 프롬프트 |
| 2024 | STPrompt | - | 시공간 프롬프트 VLM |
| 2025 | GV-VAD | **89.3%** | 확산 모델 기반 합성 데이터 |
| 2025 | IFS-VAD | 86.57% | 클립 간 특성 유사성 |

### 6.2 방법론별 분류

#### A. 통계 기반 접근 (BN-WVAD와 동류)
- **제로샷 이상탐지 (ACR, 2023)**: 배치정규화를 이용한 적응형 중심 표현학습으로, 새로운 데이터셋에 추가 훈련 없이 적응 가능 [arxiv](https://arxiv.org/pdf/2302.07849.pdf)
- **강점**: 계산이 간단하고 일반화 가능
- **한계**: 배치의 정상 샘플 다수(majority)를 가정하므로, 이상 비율이 높은 상황에서 실패

#### B. 언어-시각 모델 기반 (2024-2025)
- **TPWNG (2024)**: CLIP의 텍스트-시각 정렬을 활용하여 정규성 지도 생성 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10658364/)
  - 강점: 사전 학습된 언어 이해 활용
  - 한계: 텍스트 프롬프트 설계 필요, 계산 비용 높음
  
- **STPrompt (2024)**: 시공간 프롬프트로 국소적 이상 위치 특정 [dl.acm](https://dl.acm.org/doi/10.1145/3664647.3681442)
  - 강점: 공간 수준의 세밀한 탐지
  - 한계: 배경 정보에 여전히 민감
  
- **RelVid (2025)**: 관계형 학습과 VLM [mdpi](https://www.mdpi.com/1424-8220/25/7/2037)
  - 강점: 이상 유형 간 관계 모델링
  - 한계: 복잡한 아키텍처

#### C. 생성 모델 기반 (2024-2025)
- **GV-VAD (2025)**: 확산 모델로 합성 이상 비디오 생성 [arxiv](https://arxiv.org/pdf/2508.00312.pdf)
  - **핵심**: 다양한 이상 설명 요소(카메라 각도, 위치, 주체, 행동)로 제어 가능한 합성 데이터 생성
  - **성능**: 89.3% AUC (BN-WVAD 87.24%보다 +2.06%)
  - **한계**: 생성 모델의 도메인 갭, 합성 데이터의 현실성 문제
  - **기여**: 데이터 부족 상황(50% 데이터 규모)에서 50% 베이스라인을 89.3%로 개선 가능

#### D. 실시간 탐지 (2024)
- **REWARD (2024)**: 자기감독 학습 기반 실시간 탐지 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10483693/)
  - **성능**: 6.4초 결정 주기로 86.94% AUC 달성 (기존 SOTA는 273초 필요)
  - **혁신**: 엔드-투-엔드 훈련으로 특성 추출기도 함께 학습
  - **활용**: 감시 시스템에서 즉각적 대응 가능

#### E. 멀티모달 학습 (2022-2025)
- **MACIL-SD (2022)**: 모드 간 대조 인스턴스 학습 [github](https://github.com/aodongli/zero-shot-ad-via-batch-norm)
  - XD-Violence에서 83.40% AP 달성 (오디오-시각)
  
- **AVadCLIP (2025)**: 오디오-시각 협력 CLIP 기반 [arxiv](http://arxiv.org/pdf/2504.04495.pdf)
  - 강점: CLIP의 다중모드 표현 활용
  - 한계: 오디오 없는 감시 영상에 부적용

#### F. 연합 학습 (2024-2025)
- **연합 약감독 WSVAD (2025)**: 다중 클라이언트 간 개인정보 보호 [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/35398)
  - **강점**: 프라이버시 보호하며 여러 기관의 데이터 활용
  - **혁신**: 글로벌-로컬 컨텍스트 분리로 개인화된 탐지

### 6.3 성능 비교 시각화

**UCF-Crime AUC 진화:**
```
85% ├─ RTFM (2021, 84.30%)
     ├─ MSL (2022, 85.30%)
     ├─ S3R (2022, 85.99%)
     ├─ UR-DMU (2023, 86.97%)
     ├─ REWARD (2024, 86.94%)
     ├─ IFS-VAD (2025, 86.57%)
87% ├─ BN-WVAD (2023, 87.24%) ★
     └─ GV-VAD (2025, 89.3%) ★★
```

**XD-Violence AP 진화:**
```
75% ├─ RTFM (2021, 77.81%)
     ├─ MSL (2022, 78.28%)
80% ├─ S3R (2022, 80.26%)
     ├─ UR-DMU (2023, 81.77%)
     ├─ MACIL-SD (2022, 83.40%)
     ├─ SAS (2023, 83.59%)
85% ├─ BN-WVAD (2023, 84.93%) ★
     └─ IFS-VAD (2025, 83.14%)
```

### 6.4 핵심 차별성 분석

#### BN-WVAD의 독특한 특징
1. **단순성 vs 효과성**: 추가 명시적 메모리 모듈 없이 배치정규화만으로 정규성 캡처
2. **통계적 정당성**: 중심극한정리 기반 수학적 기초 제공
3. **재사용성**: DFM과 SBS가 다른 방법에 직접 적용 가능한 범용성

#### GV-VAD가 BN-WVAD를 능가하는 이유
- **데이터 증강의 강력함**: 합성 데이터로 훈련 분포 확장
- **도메인 갭 관리**: 합성 샘플 손실 스케일링으로 가중치 조정
- **상보성**: 실제 이상과 합성 이상의 다양성 학습

***

## 7. 논문의 연구 영향 및 향후 연구 방향

### 7.1 학계에 미친 영향

**1. BatchNorm의 재평가**: 기존에 정규화 기법으로만 여겨진 배치정규화가 이상탐지의 핵심 요소임을 입증, 후속 연구에서 ACR(Zero-Shot AD via Batch Normalization, 2023), AnoOnly 등으로 확대됨. [arxiv](https://arxiv.org/abs/2302.07849)

**2. 통계 기반 패러다임의 부활**: 기계학습에서 통계적 기초의 중요성 강조, 특히 약약한 감독 설정에서 단순한 방법의 효과성 재인식.

**3. 라벨 노이즈 강건성 연구 촉발**: DFM이 밀집 특성 공간에서 노이즈에 강하다는 통찰이, 노이즈 라벨 학습의 새로운 방향 제시. [arxiv](https://arxiv.org/pdf/2305.18798.pdf)

**4. 배치 통계의 이론화**: CLT 기반 이상치 특성화로 심층 이론 연구를 활성화.

### 7.2 앞으로의 연구 시 고려 사항

#### A. 단기 개선 방향 (1-2년)

**1. 멀티모달 통합의 자동화**
- 현재: 오디오-시각 특성을 간단히 연결
- 개선: 각 모드별 독립적 배치정규화 + 가중치 학습
- 기대 효과: XD-Violence에서 85.26%를 86% 이상으로 향상

**2. 작은 데이터셋 적응**
- 문제: ShanghaiTech에서 S3R과의 격차가 0.13% 수준
- 해결안: 메타-러닝 기반 모멘텀 $\alpha$ 적응적 학습
  $$\alpha_{\text{meta}} = f_{\text{meta}}(\text{훈련세트 크기}, \text{이상비율})$$
- 기대 효과: 소규모 데이터셋에서 일반화 성능 +2-3%

**3. 배치 크기 독립성**
- 현재: 고정된 배치 크기 (64, 64) 필수
- 개선안: 정규화된 배치 통계로 크기 무관하게 작동
  $$\mu_{\text{norm}} = \frac{\mu}{|B|}, \quad \sigma^2_{\text{norm}} = \frac{\sigma^2}{|B|^2}$$

#### B. 중기 개선 방향 (2-3년)

**1. 생성 모델과 통합**
- GV-VAD의 성공(89.3% AUC)을 BN-WVAD의 효율성과 결합
- 하이브리드: 배치정규화 기반 필터링 + 확산 모델 증강
- 기대 성능: 89% 이상 달성

**2. 실시간 탐지 확장**
- REWARD의 6.4초 결정 주기와 BN-WVAD의 정확도 결합
- 경량화: 특성 추출에 경량 모델(MobileNet, EdgeConv) 사용
- 온디바이스 배포: 엣지 디바이스에서 직접 추론

**3. 개방형 세계 이상탐지**
- 미지의 이상 유형 탐지 능력 추가
- 기법: OWVAD의 CLIP 기반 접근과 BN-WVAD의 통계 결합
- 기대 범위: 기존 13개 이상 유형 + 미지의 유형 탐지

#### C. 장기 연구 방향 (3년 이상)

**1. 설명 가능한 AI (XAI) 통합**
- 문제: DFM 기준의 해석 어려움
- 해결: 특성 중요도 가시화
  $$\text{Importance}_c = \frac{\partial \text{DFM}}{\partial X^h_c} = \frac{2(X^h_c - \mu_c)}{\sigma^2_c}$$
- 응용: 감시 영상에서 "왜 이 프레임이 이상인지" 설명

**2. 도메인 적응과 전이학습**
- 문제: UCF-Crime 훈련 모델이 다른 데이터셋에 성능 저하
- 방안: 도메인 특정 배치정규화 파라미터 학습
  $$\alpha_{\text{target}} = \alpha_{\text{source}} + \Delta \alpha_{\text{domain}}$$
- 기대: 미세조정으로 새 도메인에 빠르게 적응 (5-10 에포크)

**3. 교차-도메인 일반화**
- 여러 도메인의 데이터로 동시 훈련하되, 각 도메인의 고유성 보존
- 기법: 혼합 전문가(MoE) + 도메인 가중치
- 성과: 이전에 본 적 없는 도메인에서도 85% 이상 성능 유지

**4. 연합 학습의 프라이버시 보호**
- 각 기관의 데이터를 로컬에서만 처리, 그래디언트만 공유
- 기법: 차등 프라이버시(Differential Privacy) 추가
- 제약: 통신 오버헤드 vs 프라이버시 트레이드오프 최적화

#### D. 산업 적용 고려사항

**1. 계산 효율성**
- 현재 한계: 테스트 시 10개 crops 필요 (UCF-Crime), 느린 추론
- 개선: 싱글-패스 추론으로 100배 가속화
- 타겟: 고해상도(4K) 영상을 30fps로 실시간 처리

**2. 배포 방식**
- 클라우드: 고성능 요구 시스템용 (GV-VAD 기반 89.3%)
- 엣지: 저지연 요구 시스템용 (REWARD 기반, 6.4초)
- 하이브리드: 프라이버시 중요 시 연합 학습

**3. 모니터링과 업데이트**
- 데이터 드리프트 감지: 배치 통계의 시간 변화 추적
- 온라인 학습: 새로운 이상 유형이 등장할 때 점진적 적응
- 재훈련 빈도: 월 1회 또는 성능 저하 감지 시

***

## 결론

"BatchNorm-based Weakly Supervised Video Anomaly Detection"은 **단순하면서도 통계적으로 견고한 접근 방식으로 SOTA 성능(87.24% AUC, 84.93% AP)을 달성**했다. 배치정규화의 숨겨진 잠재력을 발견하고, 이를 통계적으로 정당화한 점은 학술적 기여도가 크다.

그러나 최신 연구 동향을 보면, **생성 모델 기반 방법(GV-VAD, 89.3% AUC)이 점진적으로 우위를 차지**하고 있으며, **멀티모달 학습**, **실시간 탐지**, **개방형 세계 학습** 등으로 연구가 다각화되고 있다. 

**향후 연구는 BN-WVAD의 효율성과 해석 가능성을 유지하면서, 생성 모델의 강력함, 언어-시각 모델의 의미적 이해, 실시간 처리의 즉각성을 결합하는 방향으로 나아가야 한다.** 특히 **도메인 적응**, **개인정보 보호를 위한 연합 학습**, **설명 가능한 이상탐지**는 산업 배포에서 필수적 요소가 될 것이다.

***

## 참고문헌 인용

- [^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_11][^1_41][^1_42][^1_43][^1_3][^1_44][^1_45][^1_46][^1_47][^1_48][^1_6][^1_14][^1_7][^1_10][^1_13][^1_8][^1_9][^1_4][^1_5][^1_12][^1_2]

<div align="center">⁂</div>

[^1_1]: 2311.15367v1.pdf

[^1_2]: https://arxiv.org/pdf/2305.18798.pdf

[^1_3]: https://beei.org/index.php/EEI/article/download/3944/3080

[^1_4]: https://arxiv.org/pdf/2302.07849.pdf

[^1_5]: https://arxiv.org/abs/2302.07849

[^1_6]: https://ieeexplore.ieee.org/document/10658364/

[^1_7]: https://dl.acm.org/doi/10.1145/3664647.3681442

[^1_8]: https://www.mdpi.com/1424-8220/25/7/2037

[^1_9]: https://arxiv.org/pdf/2508.00312.pdf

[^1_10]: https://ieeexplore.ieee.org/document/10483693/

[^1_11]: https://openaccess.thecvf.com/content/WACV2024/papers/Karim_Real-Time_Weakly_Supervised_Video_Anomaly_Detection_WACV_2024_paper.pdf

[^1_12]: https://github.com/aodongli/zero-shot-ad-via-batch-norm

[^1_13]: http://arxiv.org/pdf/2504.04495.pdf

[^1_14]: https://ojs.aaai.org/index.php/AAAI/article/view/35398

[^1_15]: https://linkinghub.elsevier.com/retrieve/pii/S0031320324006496

[^1_16]: https://ieeexplore.ieee.org/document/10720820/

[^1_17]: https://linkinghub.elsevier.com/retrieve/pii/S156625352500329X

[^1_18]: https://ieeexplore.ieee.org/document/10948323/

[^1_19]: https://link.springer.com/10.1007/s00138-025-01676-x

[^1_20]: https://ieeexplore.ieee.org/document/10657732/

[^1_21]: https://arxiv.org/pdf/2212.04090.pdf

[^1_22]: http://arxiv.org/pdf/2408.05905.pdf

[^1_23]: https://arxiv.org/pdf/2104.14770.pdf

[^1_24]: https://arxiv.org/pdf/2303.18044.pdf

[^1_25]: https://www.mdpi.com/2313-433X/4/2/36/pdf

[^1_26]: https://arxiv.org/pdf/2108.08996.pdf

[^1_27]: https://arxiv.org/html/2503.13195v1

[^1_28]: https://arxiv.org/pdf/2511.10334.pdf

[^1_29]: https://pubmed.ncbi.nlm.nih.gov/41489948/

[^1_30]: https://www.semanticscholar.org/paper/Zero-Shot-Anomaly-Detection-via-Batch-Normalization-Li-Qiu/ca43e2d53b16a60b3ea1dea017b001b5ac8805c4

[^1_31]: https://openaccess.thecvf.com/content/ICCV2025/papers/Amicantonio_Mixture_of_Experts_Guided_by_Gaussian_Splatters_Matters_A_new_ICCV_2025_paper.pdf

[^1_32]: https://arxiv.org/html/2508.14203v1

[^1_33]: https://www.arxiv.org/abs/2412.20201

[^1_34]: https://arxiv.org/abs/2508.14203

[^1_35]: https://arxiv.org/abs/2410.21991

[^1_36]: https://arxiv.org/html/2510.22056v1

[^1_37]: https://arxiv.org/abs/2311.15367

[^1_38]: https://www.sciencedirect.com/science/article/abs/pii/S0925231222000443

[^1_39]: https://www.sciencedirect.com/science/article/pii/S2590123023001536

[^1_40]: https://arxiv.org/abs/2311.15367v1

[^1_41]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9095345/

[^1_42]: https://ml.cs.rptu.de/publications/2023/FewShotAD.pdf

[^1_43]: https://www.sciencedirect.com/science/article/abs/pii/S0925231224014693

[^1_44]: https://arxiv.org/abs/2408.05905

[^1_45]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8609273/

[^1_46]: https://openaccess.thecvf.com/content/WACV2024/html/Karim_Real-Time_Weakly_Supervised_Video_Anomaly_Detection_WACV_2024_paper.html

[^1_47]: https://arxiv.org/abs/2409.05383

[^1_48]: https://liner.com/review/zeroshot-anomaly-detection-via-batch-normalization
