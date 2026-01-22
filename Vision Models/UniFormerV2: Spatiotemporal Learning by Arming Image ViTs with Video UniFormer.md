
# UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer

## 1. 핵심 주장과 주요 기여 요약

UniFormerV2는 사전학습된 이미지 Vision Transformer(ViT)와 효율적인 UniFormer 설계를 결합하여 비디오 이해의 패러다임을 재정의한 논문입니다. **핵심 주장**은 다음과 같습니다:

- **문제점**: 기존 비디오 트랜스포머는 전역 의존성 포착에는 탁월하나 지역 중복성 처리 능력 부족. UniFormerV1은 이를 해결하지만 처음부터 이미지 사전학습 필요(비효율적)
- **솔루션**: 공개된 고품질 사전학습 ViT에 효율적인 비디오 설계를 결합하는 일반 패러다임 제시
- **주요 기여**:
  1. 지역 및 전역 MultiHead Relation Aggregator(MHRA) 설계로 정확도-계산량 균형 달성
  2. Kinetics-400에서 최초로 90% 정확도 달성
  3. 8개 주요 벤치마크에서 최첨단 성능
  4. 다양한 사전학습 방식(감독학습, 대조학습, 마스킹)과 호환 가능

***

## 2. 해결하고자 하는 문제

### 문제의 근본 원인

기존 비디오 트랜스포머 아키텍처는 두 가지 근본적인 한계가 있습니다:

**1) ViT 기반 접근법의 한계**

$$X_{attention} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

이 표준 자기 주의 메커니즘은 모든 토큰을 전역적으로 비교하므로:
- 시각적 중복성이 많은 얕은 층에서도 과도한 계산 발생
- 지역 구조와 시간적 중복성을 효율적으로 처리 불가

**2) UniFormerV1의 실무적 한계**

UniFormerV1은 시공간 문제를 3D 튜브 $\Omega^{t \times h \times w}_i$에서 처리하는 지역 MHRA로 해결하지만:
- 새로운 아키텍처는 ImageNet에서 처음부터 학습 필요
- 대규모 고품질 사전학습 데이터 미활용

### 동기부여 데이터

논문의 Figure 1에서 보여주는 시각화:
- TimeSformer는 얕은 층에서만 지역 특성 학습, 깊은 층에서 추상화 부족
- UniFormerV1은 다단계 다운샘플링으로 깊은 층에서 세부 정보 손실
- **필요성**: 깊은 층에서도 지역 세부 사항과 전역 의존성을 동시에 포착

***

## 3. 제안하는 방법 (수식 포함)

### 3.1 전체 프레임워크 구조

입력 비디오를 $T \times H \times W$ 크기의 시공간 토큰으로 변환:

$$X_{in} \in \mathbb{R}^{L \times C}, \quad L = T \times H \times W$$

3D 컨볼루션 $3 \times 16 \times 16$ (스트라이드 $2 \times 16 \times 16$)을 통해 처음부터 사전학습된 ViT 호환성 유지.

### 3.2 지역 UniBlock(Local UniBlock)

지역 UniBlock은 공간 ViT 블록 전에 시간 MHRA 삽입:

$$X^T = \text{LT-MHRA}\left(\text{Norm}(X_{in})\right) + X_{in} \quad \text{(식 1)}$$

$$X^S = \text{GS-MHRA}\left(\text{Norm}(X^T)\right) + X^T \quad \text{(식 2)}$$

$$X^L = \text{FFN}\left(\text{Norm}(X^S)\right) + X^S \quad \text{(식 3)}$$

**핵심 설계**: 시공간 어파이니티를 분해

$$A^{LT}_n(X_i, X_j) = a^{i-j}_n, \quad j \in \Omega^{t \times 1 \times 1}_i \quad \text{(식 6)}$$

시간 튜브 $t \times 1 \times 1$ 내에서만 관계 학습 → 계산량을 $O(T)$로 선형화

$$A^{GS}_n(X_i, X_j) = \frac{\exp\{Q_n(X_i)^T K_n(X_j)\}}{\sum_{j' \in \Omega^{1 \times H \times W}} \exp\{Q_n(X_i)^T K_n(X_j')\}} \quad \text{(식 7)}$$

단일 프레임 $1 \times H \times W$ 내 전역 공간 주의 → 사전학습된 ViT 가중치 활용

일반 MHRA 설계:

$$R_n(X) = A_n V_n(X) \quad \text{(식 4)}$$

$$\text{MHRA}(X) = \text{Concat}(R_1(X); R_2(X); \cdots; R_N(X))U \quad \text{(식 5)}$$

여기서 $A_n$은 어파이니티 행렬, $V_n(\cdot)$는 선형 사영, $U \in \mathbb{R}^{C \times C}$는 학습 가능한 융합 행렬.

### 3.3 전역 UniBlock(Global UniBlock)

전역 UniBlock은 교차 주의(Cross-Attention) 스타일의 MHRA 사용:

$$X^C = \text{DPE}(X^L) + X^L \quad \text{(식 8)}$$

$$X^{ST} = \text{C-MHRA}\left(\text{Norm}(q), \text{Norm}(X^C)\right) \quad \text{(식 9)}$$

$$X^G = \text{FFN}\left(\text{Norm}(X^{ST})\right) + X^{ST} \quad \text{(식 10)}$$

교차 MHRA 정의:

$$R^C_n(q, X) = A^C_n(q, X)V_n(X) \quad \text{(식 11)}$$

$$\text{C-MHRA}(q, X) = \text{Concat}(R^C_1(q, X); R^C_2(q, X); \cdots; R^C_N(q, X))U \quad \text{(식 12)}$$

교차 어파이니티(복잡도 $O(L)$ 달성):

$$A^C_n(q, X_j) = \frac{\exp\{Q_n(q)^T K_n(X_j)\}}{\sum_{j' \in \Omega^{T \times H \times W}} \exp\{Q_n(q)^T K_n(X_j')\}} \quad \text{(식 13)}$$

**설계 관점**: 
- 학습 가능한 쿼리 토큰 $q \in \mathbb{R}^{1 \times C}$ (영초기화)
- 모든 시공간 토큰으로부터 최대 정보 추출
- 복잡도 $O(L^2)$ → $O(L)$로 감소

### 3.4 다단계 융합 블록(Multi-Stage Fusion)

각 전역 UniBlock $i$에서 생성된 비디오 토큰 $X^G_i = G_i(q_i, X^L_i)$를 융합:

**순차적 융합(Sequential Fusion)** - 가장 효과적:

$$X^G_i = G_i(X^G_{i-1}, X^L_i)$$

이전 전역 토큰을 현재 쿼리로 사용하는 계층적 구조.

**최종 표현 결합**:

$$Z = \alpha F + (1-\alpha)F_C$$

여기서:
- $F$: 최종 전역 토큰
- $F_C$: 최종 지역 UniBlock의 클래스 토큰
- $\alpha = \text{Sigmoid}(\alpha_{param})$: 학습 가능한 가중치

***

## 4. 모델 구조 상세 설명

### 4.1 아키텍처 레이아웃

```
입력 비디오 (T×H×W)
    ↓
3D 컨볼루션 (3×16×16 패치)
    ↓
[지역 UniBlock] → [전역 UniBlock] (Stage 1)
    ↓
[지역 UniBlock] → [전역 UniBlock] (Stage 2)
    ↓
...
    ↓
[지역 UniBlock] → [전역 UniBlock] (Stage N)
    ↓
다단계 융합 블록
    ↓
최종 분류
```

### 4.2 정규화 전략의 혁신

UniFormerV1과 달리 정규화 선택을 개선:

| 층 | 정규화 방식 | 목적 |
|---|---|---|
| LT-MHRA 전 | Batch Norm | 시간적 중복성 감소 |
| GS-MHRA/FFN 전 | Layer Norm | 사전학습 ViT 호환 |
| DPE 적용 | Depth-wise Conv | 시공간 위치 특성 |

Dynamic Position Encoding (DPE) 없음 (지역 UniBlock):
- 이유: ViT의 위치 인코딩이 이미 토큰 위치 특성 제공
- Table 9b 결과: DPE 추가 시 성능 향상 없음

### 4.3 시간 다운샘플링 설계

시간 다운샘플링 계수: 2배
- 공간: 16배, 시간: 2배
- 프레임 입력을 2배 증가: 동등한 계산량 유지
- SSV2처럼 세밀한 시간 차이가 중요한 데이터셋에서 특히 효과적
- 더 큰 시간적 수용야 달성

***

## 5. 성능 향상 및 벤치마크 결과

### 5.1 Kinetics 계열 성능

| 데이터셋 | 모델 | 프레임 | 정확도(Top-1) | 파라미터 | FLOPs |
|---------|------|-------|---|------|-------|
| **Kinetics-400** | UniFormerV2-L/14 (336↑) | 64×3×2 | **90.0%** | 354M | 75.3T |
| | UniFormerV1-B | 32×3×4 | 83.0% | 50M | 3.1T |
| | TimeSformer-L | 96×3×1 | 80.7% | 121M | 7.1T |
| | Swin-L | 32×5×10 | 84.9% | 200M | 105.4T |
| **Kinetics-600** | UniFormerV2-L/14 (336↑) | 64×3×2 | **90.1%** | 354M | 75.3T |
| **Kinetics-700** | UniFormerV2-L/14 (336↑) | 64×3×2 | **82.7%** | 354M | 75.3T |

**의미**: 최초로 Kinetics-400에서 90% 달성 - 비디오 이해의 새로운 마일스톤

### 5.2 시간 민감도 높은 데이터셋 (Something-Something V2)

| 모델 | 프레임 | 정확도 | FLOPs |
|------|-------|-------|-------|
| UniFormerV2-L/14 | 16×3×1 | 72.1% | 2.6T |
| UniFormerV2-L/14 | 32×3×1 | 73.0% | 5.2T |
| EVL-L (CLIP-400M) | 32×3×1 | 66.7% | 9.6T |
| **개선율** | | **+6.3%** | **55% FLOPs** |

세밀한 시간 관계 학습 능력 입증.

### 5.3 미삭제(Untrimmed) 비디오 벤치마크

| 데이터셋 | 모델 | 정확도 | 이전 최고 |
|---------|------|-------|---------|
| **ActivityNet** | UniFormerV2-L/14 | **94.7%** | 90.2% (NSNet-Swin-L) |
| **HACS** | UniFormerV2-L/14 | **95.4%** | 91.9% (ViViT-B) |

**의미**: 장기 시간 의존성 모델링 능력 우수

### 5.4 다양한 사전학습 원천별 성능 (Table 8)

| 사전학습 유형 | 데이터 | K400 | SSV2 | 비고 |
|-----------|-------|------|------|------|
| 감독학습(SL) | IN-21K | 81.6% | 67.5% | ViT 기준 |
| 대조학습(CL) | CLIP-400M | **84.4%** | **69.5%** | **최고 성능** |
| 마스크 이미지 모델링(MIM) | BeiT/MAE | 82.2% / 78.8% | 67.7% / 65.1% | 우수 |

**일반화 성능**: 다양한 사전학습 방식에 일관되게 효과적 (TimeSformer 대비 항상 우수)

### 5.5 Moments in Time V1 (장면 이해)

- UniFormerV2-B: 42.7% top-1
- UniFormerV2-L/14: 47.8% top-1  
- ViViT-L 대비: **+4.2% 정확도, 85% 파라미터, 15% FLOPs**

***

## 6. 모델의 일반화 성능 향상 가능성 - 핵심 분석

### 6.1 전이 학습 효율성

**Kinetics-710 사전학습의 효과**:

$$\text{훈련 비용 절감} = 1 - \frac{0.66M \times 55 + 1.14M \times 5}{1.14M \times 55} \approx 33\%$$

(식 14에서 인용)

- K400/600/700 병합 후 중복 제거 (1.14M → 0.66M)
- 불과 5 에포크 미세조정으로 1% 이상 성능 향상

### 6.2 동결(Frozen) 모델의 경이로운 일반화

| 설정 | K400 | K600 | K700 | 특징 |
|------|------|------|------|------|
| UniFormerV2-L/14 (frozen, CLIP) | 87.8% | 89.1% | 80.6% | NN 파라미터 0 |
| 미세조정 버전 | 89.7% | 89.9% | 82.1% | +1.9% 개선 |

CLIP 사전학습만으로도 87% 달성 → **강력한 초기 표현학습**

### 6.3 데이터셋 특성별 최적화

**지역 vs 전역 블록의 역할**:

| 데이터셋 특성 | 최적 구성 | 근거 |
|-----------|---------|------|
| 장면 기반 (K400) | 마지막 4층만 전역 | 전역 토큰이 분류에 핵심 (84.4% vs 81.8%) |
| 시간 기반 (SSV2) | 모든 층 + 시간 다운샘플 | 세밀한 시간 차이 중요 (69.5% 달성) |

### 6.4 도메인 간 전이 성능

**웹 스케일 데이터와의 비교**:

- MTV-H (60M 비디오-텍스트): 89.9% (K400), 4개 모델 앙상블
- UniFormerV2 단일: 90.0%, 1% 사후사전학습, 16% 미세조정

**결론**: 작은 동질 데이터셋(K710)이 대규모 이질 데이터셋보다 우수한 전이 성능 제공

### 6.5 Frozen 모델의 Cross-Dataset Generalization

| 소스→타겟 | 사전학습 | SSV2 성능 | 특성 |
|--------|--------|---------|------|
| CLIP→SSV2 | 동결 | 69.5% | 미세조정 필요 없음 |
| CLIP+K710→SSV2 | 동결 | 72.1%* | +2.6% with K710 |

*비동결 K710 미세조정: 73.0%

***

## 7. 한계(Limitations)

논문의 "More Discussions" (E.5) 명시:

### 7.1 확장성 제약

성능이 사전학습 데이터 규모에 의존:
- 소규모 데이터: MAE, BeiT 대비 덜 효과적
- 필요성: 대규모 기초 모델 활용 시 확장성 검증

### 7.2 설계 선택의 한계

**쿼리 토큰 수**:
- 1개: 69.5% (SSV2)
- 4개: 69.1% (심각한 과적합)
- 16개: 68.6%

→ 단일 쿼리 토큰이 최적; 다중 쿼리 불가

**K400 사전학습의 부정적 효과**:

| 설정 | SSV2 |
|------|------|
| CLIP만 | 69.5% |
| CLIP + K400 | 68.4% |
| **악화**: -1.1% |

→ SSV2는 시간 정보에 특화, CLIP 표현 손상

### 7.3 계산 효율 vs 정확도 트레이드-오프

**고프레임 샘플링 (64×3×4)**:
- FLOPs: 150.6T (높음)
- 정확도: 90.0% (한계 효과)

한계 개선: 0.1%/75T FLOPs 비효율적

***

## 8. 관련 최신 연구 비교 분석 (2020년 이후)

### 8.1 비디오 트랜스포머 진화

| 연도 | 모델 | 핵심 기여 | K400 정확도 | 한계 |
|------|------|---------|-----------|------|
| 2021 | ViViT | 순수 ViT 비디오 확장 | 84.9% | 계산량 과다 |
| 2021 | TimeSformer | 공간-시간 분리 주의 | 80.7% | 지역 중복성 처리 불가 |
| 2021 | MViT | 계층 구조 + 풀링 | 86.1% | 해상도 손실 |
| 2022 | UniFormer V1 | MHRA 통합 설계 | 83.0% | 사전학습 필요 |
| 2022 | **UniFormerV2** | **ViT + UniFormer** | **90.0%** | **규모 의존성** |

### 8.2 사전학습 전략 비교

| 방식 | 데이터 | 학습 시간 | K400 성능 | 범용성 |
|------|-------|---------|---------|-------|
| 비디오 전문 | K400: 110 에포크 | 매우 김 | 83.0% (V1) | 낮음 |
| 이미지 + 비디오 | IN-1K + K400 | 적중 | 82.7% | 중간 |
| 웹 스케일 | JFT-300M | 불명 | 85.8% | 높음 |
| **CLIP 이미지** | **400M 쌍** | **짧음** | **84.4%** | **높음** |
| **CLIP + K710** | **이미지 + 0.66M 비디오** | **단기** | **90.0%** | **매우 높음** |

### 8.3 구조적 설계 혁신 비교

| 설계 요소 | UniFormerV1 | UniFormerV2 | 개선점 |
|---------|-----------|-----------|-------|
| 지역 MHRA | 3D 튜브 (t×h×w) | 시간 튜브 (t×1×1) | 매개변수 감소 |
| 전역 MHRA | 자기 주의 O(L²) | 교차 주의 O(L) | 복잡도 99% 감소 |
| 위치 인코딩 | 동적(DPE) 필수 | 선택적 (깊은 층만) | 유연성 증대 |
| 정규화 | 통일된 BN | BN+LN 혼합 | 사전학습 호환성 |

### 8.4 최근 경쟁 모델 (2023-2024)

**VideoMAE v2** (자감독, 2024):
- 사전학습: 매우 비효율적 (2400+ 에포크 필요)
- K400: 75.3% (UniFormerV2 대비 15% 낮음)
- 미세조정: 40 에포크 필요

**CoCa** (멀티모달 파운데이션):
- K400: 88.9% (1% 낮음)
- 파라미터: 1000M+ (2.8배 많음)
- 범용성 (비디오 분류 외): 높음

**MTV-H** (멀티뷰):
- K400: 89.9% (0.1% 낮음)
- 모델: 4개 앙상블 필요
- UniFormerV2: 단일 모델

***

## 9. 논문의 앞으로의 연구에 미치는 영향

### 9.1 패러다임 전환

**이전 패러다임**: 비디오 특화 아키텍처 설계 → 이미지 사전학습

```
3D 설계 → 이미지 학습 (110 에포크) → 비디오 미세조정 (50 에포크)
```

**UniFormerV2 패러다임**: 이미지 사전학습 ← 활용 → 비디오 효율 설계

```
공개 ViT (사전학습 완료) + 비디오 모듈 삽입 (5-22 에포크만 필요)
```

### 9.2 산업 적용성 혁신

**배포 효율**:
- 훈련 시간: UniFormerV1 (110+50=160 에포크) → UniFormerV2 (22 에포크, 0.9% 시간)
- 동결 모델: CLIP-400M으로 즉시 배포 가능 (미세조정 선택사항)
- 예상 영향: 기업의 비디오 AI 구축 시간 90% 단축

### 9.3 멀티모달 학습 기초 마련

CLIP 사전학습 기반:
- 텍스트-이미지 정렬 → 텍스트-비디오 정렬 (자연스러운 확장)
- 비디오 캡셔닝, VQA 등 다운스트림 작업 용이
- 오픈 어휘 비디오 이해 가능성 제시

### 9.4 부족한 리소스 환경의 민주화

```
필요 GPU 메모리 비교:
- UniFormerV1-B 사전학습: ~128GB (여러 GPU)
- UniFormerV2-B (CLIP 활용): ~8GB (단일 GPU, 미세조정만)

결과: 기업/중소 연구팀도 최고 성능 모델 접근 가능
```

### 9.5 이론적 기여: 지역-전역 상호작용 이해

**핵심 발견**:
- 지역 MHRA (시간 튜브): 얕은 층에서 효과적 (Table 9a 포착)
- 전역 MHRA (교차 주의): 깊은 층에서 필수 (장기 의존성)
- 데이터셋 특성에 따라 최적 배치 다름

**이론적 시사**: 시공간 표현 학습의 계층적 본질 규명

***

## 10. 앞으로 연구 시 고려할 점

### 10.1 확장성 검증 필요

**제안 방향**:
1. **초대형 기초 모델과의 결합**
   - DINOv2, SAM, EVA-02 같은 1B+ 파라미터 모델
   - 현재: CLIP-400M 텍스트 쌍 → 향후: 순수 이미지 파운데이션
   
2. **지식 증류(Knowledge Distillation)** 활용
   - 대형 UniFormerV2 → 소형 모델 압축
   - 모바일/엣지 배포 최적화

### 10.2 도메인 특화 적응

**한계**: 현재 순수 웹/오픈데이터에 의존

**향후 연구**:
1. **의료 비디오 특화**
   - 미세한 움직임 감지 (수술, 진단)
   - 도메인 적응 효율성 연구

2. **극단적 조건 강건성**
   - 야간/저조도 비디오
   - 극도로 빠르거나 느린 동작
   - 도메인 일반화 능력 평가

### 10.3 효율성-정확도 파레토 최적화

**현 한계**: 고프레임(64×3×4)의 한계 개선 미미 (0.1%/75T FLOPs)

**개선 방안**:
1. **적응적 프레임 샘플링**
   - 동작 강도에 따른 동적 샘플링
   - 계산 절감: ~30-40% 예상

2. **토큰 가지치기(Token Pruning)**
   - 중복 토큰 제거 (시공간)
   - 초기 레이어에서 50% 토큰 제거 가능

### 10.4 시간 모델링의 한계 극복

**현 제약**: 로컬 시간 MHRA는 여전히 t×1×1로 제한

**제안 개선**:
1. **적응적 시간 창**
   - 레이어별 다양한 시간 범위 (t=3,5,7,...)
   - SSV2 같은 데이터셋에서 성능 추가 향상 예상

2. **시간 어텐션 헤드 특화**
   - 일부 헤드: 로컬 시간(현재)
   - 일부 헤드: 전역 시간(새로운)
   - 하이브리드 모델링

### 10.5 해석 가능성 연구

**Table 4의 시각화 외 미흡한 부분**:

1. **주의 맵 분석**
   - 각 MHRA가 실제로 학습한 패턴
   - 지역 vs 전역 역할 정량화

2. **특성 공간 기하학**
   - 임베딩 공간에서 유사 동작의 클러스터링
   - 표현 학습의 질 평가

### 10.6 제약 환경 최적화

**현재 가정**: 충분한 계산 자원

**향후 고려**:
1. **경량 버전** (UniFormerV2-Tiny)
   - 파라미터 10M 미만
   - 모바일: 최소 20fps 유지

2. **양자화/낮은 정밀도**
   - INT8 또는 FP16 사전조율
   - 배포 에너지 효율성 50% 개선 기대

***

## 11. 결론

UniFormerV2는 비디오 이해 분야에서 **패러다임 시프트**를 제시합니다. 공개된 고품질 이미지 기초 모델을 효과적으로 활용하면서도 비디오 특화 설계(지역-전역 MHRA)를 통해 최고 성능을 달성하는 **우아한 솔루션**입니다.

### 핵심 가치:

1. **효율성**: 비용 33% 절감, 미세조정 22 에포크만 필요
2. **성능**: 8개 벤치마크에서 최첨단, Kinetics-400 최초 90% 달성
3. **확장성**: 다양한 사전학습 원천과 호환 가능
4. **일반화**: 도메인 간 강력한 전이 능력

이 연구는 **리소스 제약이 있는 연구팀도 최고 성능에 접근 가능**하게 만들어 AI 민주화에 기여하며, 멀티모달 기초 모델 시대에서 **효율적인 도메인 적응**의 청사진을 제시합니다.

<span style="display:none">[^1_1][^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_4][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_5][^1_50][^1_51][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2211.09552v1.pdf

[^1_2]: https://ieeexplore.ieee.org/document/10798341/

[^1_3]: https://arxiv.org/abs/2406.05352

[^1_4]: https://arxiv.org/abs/2405.04404

[^1_5]: https://ieeexplore.ieee.org/document/10645423/

[^1_6]: https://link.springer.com/10.1007/s10278-024-01322-4

[^1_7]: https://link.springer.com/10.1007/s10462-024-11019-3

[^1_8]: https://www.semanticscholar.org/paper/47741d9e57f7a986a350f5cb20de287b1b1b8ff8

[^1_9]: https://dl.acm.org/doi/10.1145/3674500

[^1_10]: https://hdl.handle.net/1912/69773

[^1_11]: https://cvr.ac.in/ojs/index.php/cvracin/article/view/917/740

[^1_12]: https://arxiv.org/pdf/2201.05991v2.pdf

[^1_13]: https://arxiv.org/pdf/2403.07542.pdf

[^1_14]: https://arxiv.org/pdf/2111.06091.pdf

[^1_15]: https://arxiv.org/pdf/2304.09854.pdf

[^1_16]: https://arxiv.org/pdf/2310.12296.pdf

[^1_17]: https://arxiv.org/pdf/2311.16673.pdf

[^1_18]: http://arxiv.org/pdf/2305.09880.pdf

[^1_19]: https://arxiv.org/pdf/2109.09920.pdf

[^1_20]: https://aclanthology.org/2024.findings-acl.217.pdf

[^1_21]: https://www.sciencedirect.com/science/article/abs/pii/S0167865518302058

[^1_22]: https://www.emergentmind.com/topics/spatiotemporal-transformer

[^1_23]: https://arxiv.org/html/2312.17432v5

[^1_24]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8659437/

[^1_25]: https://openreview.net/pdf?id=k4OHdGFTCIR

[^1_26]: https://jina.ai/vision-encoder-survey.pdf

[^1_27]: https://openaccess.thecvf.com/content/CVPR2024/papers/Qu_LLMs_are_Good_Action_Recognizers_CVPR_2024_paper.pdf

[^1_28]: https://proceedings.neurips.cc/paper/2021/file/5edc4f7dce28c711afc6265b4f99bf57-Paper.pdf

[^1_29]: https://github.com/nguyentthong/video-language-understanding

[^1_30]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9045967/

[^1_31]: https://openaccess.thecvf.com/content/ICCV2021/papers/Yan_Learning_Spatio-Temporal_Transformer_for_Visual_Tracking_ICCV_2021_paper.pdf

[^1_32]: https://ieeexplore.ieee.org/iel8/6287639/10820123/11007557.pdf

[^1_33]: https://www.nature.com/articles/s41598-024-58074-y

[^1_34]: https://arxiv.org/abs/2201.04676

[^1_35]: https://arxiv.org/html/2402.00045v4

[^1_36]: https://arxiv.org/pdf/2510.08480.pdf

[^1_37]: https://pubmed.ncbi.nlm.nih.gov/37334006/

[^1_38]: https://arxiv.org/html/2407.07816v1

[^1_39]: https://arxiv.org/pdf/2507.16151.pdf

[^1_40]: https://openaccess.thecvf.com/content/CVPR2021/papers/Feichtenhofer_A_Large-Scale_Study_on_Unsupervised_Spatiotemporal_Representation_Learning_CVPR_2021_paper.pdf

[^1_41]: https://arxiv.org/html/2508.16527v1

[^1_42]: https://arxiv.org/html/2507.04465v1

[^1_43]: https://arxiv.org/html/2506.18052

[^1_44]: https://arxiv.org/pdf/2507.04465.pdf

[^1_45]: https://arxiv.org/html/2505.22976v1

[^1_46]: https://openaccess.thecvf.com/content/CVPR2025W/SAIAD/papers/Anand_Detecting_Localized_Deepfake_Manipulations_Using_Action_Unit-Guided_Video_Representations_CVPRW_2025_paper.pdf

[^1_47]: https://arxiv.org/abs/2303.03856

[^1_48]: https://www.sciencedirect.com/science/article/pii/S2666307424000214

[^1_49]: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.70003

[^1_50]: https://www.nature.com/articles/s41598-025-98763-w

[^1_51]: https://dl.acm.org/doi/10.1145/3671151.3671267
