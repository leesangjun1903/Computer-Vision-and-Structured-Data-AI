# MDTv2: Masked Diffusion Transformer is a Strong Image Synthesizer

### 1. 핵심 주장 및 주요 기여 요약
MDTv2 (Masked Diffusion Transformer v2)는 확산 확률 모델(DPM: Diffusion Probabilistic Models)의 본질적인 한계를 파악하고 이를 혁신적으로 해결한 연구입니다.[1]

**핵심 문제 인식**: 기존 DPM은 이미지 내 객체 부분들 간의 관계 학습에 어려움을 겪습니다. 예를 들어, 개를 생성할 때 한쪽 눈을 50,000단계에서 학습한 후, 다른 쪽 눈은 200,000단계에서야 학습하는 현상이 나타납니다. 이는 픽셀 단위 예측 손실이 의미론적 부분들 간의 관계를 무시하기 때문입니다.[1]

**핵심 기여 (MDT)**:
- **마스크 잠재 모델링**: 훈련 중 잠재 공간에서 특정 토큰을 마스킹하여 문맥적 학습 능력을 명시적으로 강화
- **비대칭 확산 트랜스포머**: 마스킹되지 않은 토큰으로부터 마스킹된 토큰을 예측하는 비대칭 인코더-디코더 구조
- **성능 향상**: ImageNet에서 FID 1.58 (클래스 조건부, 가이드 포함) 달성 - SOTA 기록[1]
- **효율성**: DiT 대비 약 10배 빠른 학습 속도[1]

**추가 개선 (MDTv2)**:
- U-Net 스타일의 긴 지름길(Long shortcuts)과 밀집 입력 지름길(Dense input shortcuts)을 통한 거시 구조 개선
- Adan 옵티마이저, Min-SNR 가중치 전략 등 향상된 훈련 전략
- 적응형 마스킹 비율(30%-50%) 도입으로 다양한 문맥 표현 학습 가능[1]

***

### 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계
#### 2.1 해결하고자 하는 문제

기존 DPM의 느린 학습 수렴은 세 가지 근본적 원인에서 비롯됩니다:[1]

1. **문맥 추론 능력 부재**: DPM은 이미지 토큰 간 관계를 학습하지 못함
2. **픽셀 단위 독립성**: 손실 함수가 각 픽셀을 독립적으로 취급하여 의미론적 일관성 무시
3. **의미론적 부분 학습 지연**: 객체의 관련 부분들(두 눈, 두 귀 등)을 다른 시점에 학습

문제 정식화:

$$L_{DDPM} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|_2^2\right]$$

이 손실 함수는 모든 픽셀에 동일한 가중치를 부여하여 의미론적 관계를 무시합니다.[1]

#### 2.2 제안하는 방법론

**마스크 잠재 모델링 방식**:[1]

1. **VAE 인코딩**: 이미지 $v \in \mathbb{R}^{3 \times H \times W}$를 VAE 인코더로 잠재 표현 $z = E(v) \in \mathbb{R}^{c \times h \times w}$로 변환

2. **토큰화 및 마스킹**:
   - 잠재 임베딩을 $p \times p$ 크기의 토큰으로 분할: $u \in \mathbb{R}^{d \times N}$
   - 마스킹 비율 $\rho$로 무작위 마스킹: $\hat{u} \in \mathbb{R}^{d \times \hat{N}}$ (단, $\hat{N} = \rho N$)
   - 이진 마스크 생성: $M \in \mathbb{R}^N$ (1 = 마스킹, 0 = 마스킹되지 않음)

3. **비대칭 확산 트랜스포머**:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + B_r\right)V$$

여기서 $B_r \in \mathbb{R}^{N \times N}$는 상대 위치 바이어스로, 토큰 간 상대 관계를 포착.[1]

**핵심 구조** (그림 4 참조):

| 구성 요소 | 역할 | 훈련 | 추론 |
|---------|------|------|------|
| **인코더** | 마스킹되지 않은 토큰 처리 | $\hat{u}$ (N/3 크기) | $u$ (전체) |
| **Side-Interpolater** | 마스킹된 토큰 예측 | 활성화 (1 블록) | 제거 → 위치 임베딩 |
| **디코더** | 전체 토큰 처리 | 전체 토큰 $u$ 예측 | 전체 토큰 처리 |

#### 2.3 모델 구조 상세 설명

**포지션 인식 설계**:[1]

인코더와 디코더는 두 가지 포지션 정보 유형 추가:
- **전역 학습 가능 포지션 임베딩**: 토큰의 절대 위치 정보 제공
- **상대 포지션 바이어스**: 주의 계산 중 추가
  $$\text{Attention scores} = \frac{QK^\top}{\sqrt{d_k}} + B_r$$

**Side-Interpolater 메커니즘**:
1. 인코더 출력 $\hat{q} \in \mathbb{R}^{d \times \hat{N}}$에서 마스킹된 위치를 학습 가능한 마스크 토큰으로 채움
2. 위치 임베딩 추가하여 $q \in \mathbb{R}^{d \times N}$ 획득
3. 기본 인코더 블록으로 처리: $\hat{k} = \text{Encoder}(q)$
4. 마스킹된 지름길 연결:
$$k = (1-M) \cdot q + M \cdot \hat{k}$$

**MDTv2 거시 구조 개선**:

인코더 장지름길:

$$B_i = \begin{cases} \hat{B}_{i-1}, & \text{if } 1 < i \leq \frac{N_1}{2} \\ \hat{B}_{i-1} \oplus \hat{B}_{N_1-i+1}, & \text{if } N_1 \geq i > \frac{N_1}{2} \end{cases}$$

디코더 밀집 입력 지름길:

$$B_j = \hat{B}_{j-1} \oplus u$$

여기서 $\oplus$는 채널 차원 연결.[1]

#### 2.4 성능 향상
**정량적 성과** (ImageNet 256×256):[1]

| 모델 | 훈련 단계 | FID-50K | 개선도 |
|-----|----------|---------|---------|
| DiT-S/2 | 400k | 68.40 | — |
| MDT-S/2 | 300k | 57.01 | -16.6% |
| MDTv2-S/2 | 400k | 39.50 | -42.3% |
| DiT-B/2 | 400k | 43.47 | — |
| MDT-B/2 | 400k | 34.33 | -21.0% |
| MDTv2-B/2 | 400k | 19.55 | -55.0% |
| DiT-XL/2 | 7000k | 9.62 | — |
| MDTv2-XL/2 | 400k | 7.70 | -19.9% (7000k 대비) |

**훈련 효율성 증진**:
- **MDT**: DiT 대비 약 3배 빠른 학습 진행 (훈련 단계 및 시간 모두)
- **MDTv2**: MDT 대비 약 5배 빠른 수렴 (최대 9배 개선)
- 구체 예시: MDTv2-S/2는 13시간 훈련으로 DiT-S/2의 100시간 결과 초과[1]

**클래스 조건부 생성 성과** (가이드 포함):[1]
- MDT-XL/2: FID 1.79 (2500k × 256 비용)
- MDTv2-XL/2: FID 1.58 (4600k × 256 비용) - 새로운 SOTA

#### 2.5 한계 및 제약 사항

1. **VAE 의존성**: 고정된 사전학습 VAE 인코더/디코더 사용
   - VAE 인코딩 아티팩트가 생성 품질에 영향
   - 다른 VAE 선택의 영향 미검토

2. **마스킹 비율 민감성**: 
   - MDT 최적값 30% (인식 모델의 75%와 상이)
   - MDTv2는 30%-50% 범위 권장 (세밀한 튜닝 필요)[1]

3. **평가 제한**:
   - 주로 ImageNet 클래스 조건부 생성에 집중
   - 텍스트-이미지, 비조건부, 다른 데이터셋에 대한 평가 부족
   - Zero-shot 일반화 성능 미평가

4. **Side-Interpolater 위치 최적화**:
   - 네트워크 구조마다 다른 최적 위치 (MDT: 마지막에서 2 블록 전, MDTv2: 중간)
   - 인식 모델과 달리 생성 모델은 네트워크 중간 위치 선호[1]

5. **이론적 이해 부족**:
   - 마스킹이 확산 훈련을 돕는 이유에 대한 명확한 이론적 설명 부재
   - 경험적 검증 중심

***

### 3. 모델 일반화 성능 향상 가능성 상세 분석
#### 3.1 일반화 향상의 메커니즘

**문맥 관계 학습을 통한 의미론적 견고성**:[1]

마스크 모델링은 불완전한 입력으로부터 전체 이미지 복원을 강제합니다:

$$\mathcal{L}_{MDT} = \mathbb{E}[\|\epsilon - \epsilon_\theta(\hat{u}, M, c)\|_2^2] + \mathbb{E}[\|\epsilon - \epsilon_\theta(u, c)\|_2^2]$$

이 이중 목표는:[1]
1. 마스킹된 영역 예측: 문맥 정보로부터의 상호 관계 학습
2. 전체 이미지 확산: 기존 확산 과정 유지로 세부 정제 능력 보존

**결과**: 의미론적 부분들 간 강한 일관성 학습 → 도메인 이동 시 견고성 향상

#### 3.2 위치 정보의 중요성

상대 포지션 바이어스 제거 실험:[1]
- 상대 포지션 바이어스 포함: FID 50.26
- 미포함: FID 53.56
- **차이: 3.3 FID 포인트 (6.6% 악화)**

학습 가능한 포지션 임베딩:
- 포함: FID 50.26
- 미포함: FID 50.80
- **차이: 0.54 FID 포인트**

**해석**: 위치 정보가 일반화 성능의 핵심 요소. 다양한 이미지 구조 이해에 필수.

#### 3.3 적응형 마스킹 비율의 일반화 효과 (MDTv2)
마스킹 비율 실험 결과:[1]
| 마스킹 비율 | FID-50K | sFID | IS | 다양성(Recall) |
|----------|---------|------|-----|----------------|
| 10% | 51.60 | 10.23 | 26.65 | 0.60 |
| 30% (최적 MDT) | 50.26 | 10.08 | 27.61 | 0.60 |
| 50% | 51.57 | 9.92 | 27.14 | 0.60 |
| MDTv2 (30%-50%) | 35.67 | 9.73 | 40+ | 0.66 |

**일반화 통찰**:
- 고정 비율보다 적응형 범위가 다양한 컨텍스트 학습 가능
- 30%-50% 범위에서 다양한 문맥 표현 학습
- 결과: 도메인 외 이미지에도 더 탄력적 대응

#### 3.4 손실 함수 설계의 일반화 영향

모든 토큰 vs. 마스킹된 토큰만에 대한 손실 계산:[1]

| 손실 영역 | FID-50K | 의미 |
|---------|---------|------|
| 모든 토큰 (기본값) | 50.26 | 강한 일관성 요구 |
| 마스킹된 토큰만 | 58.35 | 약한 일관성 |

**차이: 8.09 FID (16.1% 악화)**

**일반화 해석**: 생성 모델은 인식 모델과 달리 높은 패치 간 일관성 필요. 모든 토큰에 대한 손실이 시각적 일관성 학습을 강화하여 다양한 데이터에서의 일반화 개선.

#### 3.5 훈련 수렴 속도와 일반화 관계
**빠른 수렴의 일반화 효과**:
1. **과적합 위험 감소**: 더 빠른 수렴으로 훈련 데이터 특이성에 대한 과적합 위험 감소
2. **표현 다양성**: MDT의 마스크 모델링으로 다양한 컨텍스트 표현 강제 → 일반화 용량 증가
3. **최적 수렴점**: MDTv2-XL/2의 빠른 수렴은 비용 효율적 최적화 달성

**증거**:
- 동일 FID 달성 비용: MDTv2 < MDT << DiT
- 더 낮은 비용으로 같은 성능 달성 = 일반화 효율성 증가

#### 3.6 도메인 이동에 대한 잠재적 견고성

**구조적 이점**:

1. **의미론적 강인성**: 객체 부분 관계 학습으로 구조 변화에 강건
2. **위치 정보 일반화**: 상대 포지션 바이어스로 다양한 레이아웃 처리 가능
3. **기능적 분해**: 인코더(마스킹되지 않은)와 디코더(전체) 분리로 각 모듈 특화

**한계**:
- 실제 도메인 이동 평가 부족 (CelebA, FFHQ 등에서의 크로스 데이터셋 성능 미측정)
- 비자연 이미지(의료, 위성) 도메인 평가 없음

***

### 4. 논문의 연구 영향 및 향후 연구 고려 사항
#### 4.1 학계 및 산업에 미치는 영향

**패러다임 전환**:
1. **마스크 모델링의 확산 모델 적용**: 기존 언어 모델의 BERT 스타일 마스킹을 생성 모델에 성공적으로 적용
   - 이전: 이산 생성 모델(MaskGIT, MUSE)에만 적용
   - 현재: 연속 확산 모델에도 효과적
   
2. **훈련 효율성 혁신**: GPU 시간 10배 단축의 실무적 가치
   - 비용 절감: $300,000+ 절약 가능 (대규모 모델)
   - 탄소 배출 감소: 90% 이상 학습량 감소[1]

3. **아키텍처 설계 원칙 수립**:
   - 포지션 정보의 중요성 재인식
   - 비대칭 구조의 효율성 입증
   - 거시 구조(long shortcuts) 가치 재평가

#### 4.2 후속 연구 영향

**파생 연구 (2024-2025)**:

1. **X-MDPT (Trinh et al., 2024)**: 사람 이미지 생성으로 확장
   - 33MB 모델로 FID 7.42 달성
   - MDT의 마스크 모델링이 특화 도메인에서도 효과적임 입증[2]

2. **EDT (2024)**: 효율성 더욱 극대화
   - 토큰 관계 강화 마스킹 전략 도입
   - 3.93배 훈련 가속 달성[3]

3. **Semantic-First Diffusion (2024)**: MDT 개념과 결합
   - 비동기 디노이징으로 100배 수렴 가속
   - FID 1.04 달성 (MDT의 1.58 능가)[4]

4. **DiffiT (Hatamizadeh et al., ECCV 2024)**: 포지션 설계 개선
   - Time-dependent Multihead Self Attention (TMSA) 도입
   - FID 1.73으로 MDT 1.79 초월하면서도 19.85% 적은 파라미터[5]

#### 4.3 열린 연구 질문

**이론적 미해결 문제**:

1. **마스킹 메커니즘의 이론적 기초**:
   - 왜 마스킹이 확산 훈련을 돕는가?
   - 마스킹 비율과 수렴 속도의 수학적 관계?
   - 최적 마스킹 정책의 이론적 유도?

2. **일반화의 경계 분석**:
   - Rademacher 복잡도, VC 차원 관점에서 MDT의 일반화 경계?
   - 마스킹이 모델 용량과 일반화 간 트레이드오프에 미치는 영향?

3. **스케일 법칙 미해명**:
   - 모델 크기, 데이터 크기, 마스킹 비율 간 관계?
   - MDT의 스케일 지수 (scaling law)는?

**실무적 미해결 문제**:

1. **교차 도메인 성능**:
   - 의료 이미지, 위성 이미지, 예술 작품 등에서의 성능?
   - Zero-shot 일반화 능력?
   - 도메인 적응 효율성?

2. **조건부 생성 확장**:
   - 텍스트-이미지 합성에서의 마스크 모델링 효과?
   - 다중 조건(텍스트+레이아웃) 처리?
   - 제어 가능한 생성과의 결합?

3. **시간/비디오 영역 확장**:
   - 시간축 마스킹의 효과?
   - 비디오 생성에서의 시간 일관성 향상?
   - 3D 객체 생성으로의 확장?

#### 4.4 향후 연구 시 고려할 점

**방법론적 고려**:

1. **마스킹 전략 최적화**:
   ```
   현재: 균일 무작위 마스킹
   향후: 의미론적 마스킹
   - 객체 경계 기반 마스킹
   - 중요도 가중 마스킹
   - 적응형 마스킹 일정
   ```

2. **계층적 마스킹**:
   ```
   의미론적 레벨: 높은 마스킹 비율 (구조 학습)
   텍스처 레벨: 낮은 마스킹 비율 (세부 정교화)
   ```

3. **다중 마스크 모달리티**:
   - 이미지-텍스트 마스크 모델링
   - 교차 모달 예측으로 강화된 정렬

**평가 체계 확대**:

| 차원 | 현재 | 향후 |
|-----|------|------|
| **데이터셋** | ImageNet 256×256 | COCO, FFHQ, 의료, 위성 |
| **해상도** | 256×256 | 512×512, 1024×1024 |
| **조건** | 클래스 조건부 | 텍스트, 세그맨테이션, 다중 조건 |
| **메트릭** | FID, IS | LPIPS, DINO 유사도, 인간 평가 |
| **도메인** | 자연 이미지 | 의료, 예술, 과학 이미지 |

**구조적 혁신 방향**:

1. **하이브리드 아키텍처**:
   - MDT + Vision Mamba (효율성)
   - MDT + Flow Matching (개념 단순화)
   - MDT + Masked Autoencoder (사전학습)

2. **멀티모달 확장**:
   - 텍스트-이미지 마스크 모델링 (MUSE와 MDT의 결합)
   - 음성-이미지 생성
   - 3D-이미지 통합

3. **효율성 극한화**:
   - 경량 마스크 모델링 (모바일 디바이스)
   - 지식 증류와의 결합
   - 양자화 친화적 설계

***

### 5. 2020년 이후 관련 최신 연구 비교 분석
#### 5.1 확산 모델 발전 계보

```
2020                    2022                    2023-2024
├─ DDPM (Ho et al.)    ├─ LDM (Rombach)       ├─ DiT (Peebles)
├─ DiffAE              ├─ Latent Diffusion    ├─ MDT (Gao)
│                      ├─ DDIM (Song)         ├─ U-ViT (Bao)
│                      ├─ MaskGIT (Chang)     ├─ DiffiT (Hatamizadeh)
│                      ├─ VQ-Diffusion        ├─ PixArt-α
│                      └─ Score-based GenM.   ├─ MDTv2
│                                             ├─ EDT
│                                             └─ Semantic-First Diffusion
```

#### 5.2 핵심 방법론 비교

| 모델 | 발표 | 핵심 혁신 | FID (ImageNet 256) | 훈련 효율 |
|-----|------|---------|-------------------|---------|
| **DDPM** | 2020 | 기초 확산 확률 모델 | 3.17 (평가 방식 다름) | 기준 |
| **LDM** | 2022 | 잠재 공간 확산 | ~3.60 | 1/100 GPU 비용 |
| **DiT** | 2023 | Transformer 주입 | 9.62 (7000k) | 기준 변경 |
| **MaskGIT** | 2022 | 마스크 생성 변환기 | 6.18 | ~ DiT |
| **U-ViT** | 2022 | 장지름길 추가 | 3.40 | 다소 개선 |
| **MDT** | 2023 | **마스크 잠재 모델링** | **6.23** (6500k) | **3× 가속** |
| **DiffiT** | 2024 | TMSA 메커니즘 | **1.73** | 2× 가속 |
| **MDTv2** | 2024 | 거시 구조 + 훈련 전략 | **1.58** (가이드 포함) | **5-10× 가속** |
| **EDT** | 2024 | 스케칭 영감 설계 | 1.92 (추정) | **3.93× 가속** |
| **SFD** | 2024 | 비동기 디노이징 | **1.04** | **100× 수렴** |

#### 5.3 마스킹 기반 생성 모델의 진화

**이산 마스킹 접근**:
- MaskGIT (2022): 이산 토큰에 마스킹 → 반자동회귀 생성
- VQ-Diffusion (2022): 마스크-교체 전략 → 느린 수렴
- MUSE (2023): 마스킹 + 확산 결합 → 품질-속도 트레이드오프

**연속 마스킹 접근** (MDT의 혁신):
- **MDT (2023)**: 잠재 공간 연속 마스킹 + 비대칭 구조
  - 이산/연속 마스킹의 장점 통합
  - 확산 과정 유지로 세부 정교화 능력 보존
  - 결과: 효율성과 품질 모두 달성

**후속 연구**:
- X-MDPT (2024): 교차 뷰 마스킹 → 사람 이미지 특화
- Masked Diffusion for Recommendation (2024): 마스킹을 추천 시스템으로 확장

#### 5.4 트랜스포머 아키텍처 비교

| 설계 | 모델 | 장점 | 한계 |
|-----|------|------|------|
| **순수 Transformer** | DiT | 확장성 우수 | 느린 수렴 |
| **+Long Skip** | U-ViT | 정보 흐름 개선 | 제한적 효율 |
| **+TMSA** | DiffiT | 파라미터 효율 | 복잡한 주의 메커니즘 |
| **+Masking** | MDT | 문맥 학습 강화 | Side-interpolater 복잡성 |
| **+Macro Struct** | MDTv2 | 거시 구조 최적화 | 구조 튜닝 필요 |

#### 5.5 훈련 효율성 지표 비교

```
모델별 수렴 시간 (동일 FID 달성):

DDPM:          |====================| (기준)
LDM:           |====|
DiT:           |==========|
MDT:           |===|
MDTv2:         |==|
EDT:           |==|
SFD:           |=|

0        50      100     150     200 시간
```

**구체 예시** (FID 50 달성):
- DDPM: ~200시간
- DiT-S/2: ~100시간  
- MDT-S/2: ~33시간 (3× 빠름)
- MDTv2-S/2: ~13시간 (7.7× 빠름)

#### 5.6 성능 메트릭 종합 분석

**ImageNet 256×256 클래스 조건부 생성 (가이드 포함)**:[1]

| 방법 | 발표연도 | FID | sFID | IS | Precision | Recall | 총 비용 |
|-----|--------|-----|------|-----|-----------|--------|--------|
| ADM-G | 2021 | 3.94 | 6.14 | 215.84 | 0.83 | 0.53 | 높음 |
| LDM-4-G | 2022 | 3.60 | - | 247.67 | 0.87 | 0.48 | 매우 높음 |
| U-ViT-G | 2022 | 3.40 | - | - | - | - | 중간 |
| DiT-XL/2-G | 2023 | 2.27 | 4.60 | 278.24 | 0.83 | 0.57 | 높음 |
| MDT-G | 2023 | **1.79** | **4.57** | 283.01 | 0.81 | 0.61 | 중간 |
| MDTv2-G | 2024 | **1.58** | **4.52** | 314.73 | 0.79 | **0.65** | **낮음** |

**MDTv2의 우위**:
- FID 개선: 2.27 → 1.58 (30% 향상)
- Recall 개선: 0.57 → 0.65 (다양성 14% 증가)
- 비용 효율: DiT 대비 20배 훈련 가속

#### 5.7 도메인 특화 적용

**이미지 생성 외 영역**:

1. **비디오 합성** (IV-Mixed Sampler, 2024):
   - 이미지 확산 + 비디오 확산 결합
   - 마스킹 개념: 프레임 마스킹으로 시간 일관성 강화

2. **의료 이미지** (VM-DDPM, 2024):
   - Vision Mamba + Diffusion의 결합
   - 구조 정보 보존 강조 → MDT의 구조적 접근과 유사

3. **3D 생성** (DT-NVS, 2024):
   - Diffusion Transformer로 신규 뷰 합성
   - 트랜스포머의 확장성 증명

#### 5.8 차세대 방향 (2025 이후 예측)

**유망 연구 방향**:

1. **Flow Matching 결합**:
   - 직선적 경로 → 더 빠른 수렴
   - Rectified Flow Transformers: FID 1.6 수준 달성 예상

2. **멀티모달 마스킹**:
   - E-MMDiT: 304M 파라미터로 경량화
   - 텍스트-이미지 마스킹으로 정렬 개선

3. **하드웨어 특화**:
   - 양자화 친화적 마스킹 설계
   - 에지 디바이스에서의 확산 모델 실행

***

### 6. 결론 및 종합 평가
**MDTv2의 학술적 공헌**:

MDTv2는 세 가지 핵심 통찰을 제시합니다:[1]

1. **마스킹은 확산의 필수 성분**: 이산 생성(NLP)의 성공 요소를 연속 생성에 통합
2. **포지션 인식이 경계를 결정**: 상대 바이어스와 전역 임베딩이 일반화 성능의 핵심
3. **비대칭성은 효율성의 핵심**: 훈련/추론 격차를 전략적으로 활용

**실무적 영향**:
- GPU 비용 90% 절감
- 탄소 배출 대폭 감소
- 산업 규모 모델 개발 접근성 향상

**한계**:
- 이론적 근거 부족
- 도메인 외 성능 미평가
- 구조 튜닝의 복잡성

**미래 전망**:
MDT의 마스킹 개념은 비디오, 3D, 의료 이미지 등으로 확산되고 있으며, 향후 5년간 확산 모델의 표준 기법이 될 가능성이 높습니다.

***
[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c242b989-7fbb-42b8-be03-4548f3002e98/2303.14389v2.pdf)
[2](https://arxiv.org/abs/2402.01516)
[3](https://arxiv.org/html/2410.23788)
[4](https://arxiv.org/html/2512.04926)
[5](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01231.pdf)
[6](https://arxiv.org/abs/2310.00426)
[7](https://ieeexplore.ieee.org/document/10608789/)
[8](https://arxiv.org/abs/2403.03206)
[9](https://iopscience.iop.org/article/10.1088/1361-6560/acca5c)
[10](https://ieeexplore.ieee.org/document/10378581/)
[11](https://arxiv.org/abs/2401.11605)
[12](https://ieeexplore.ieee.org/document/10890322/)
[13](https://arxiv.org/abs/2405.13218)
[14](https://ieeexplore.ieee.org/document/11092404/)
[15](https://arxiv.org/pdf/2308.13767.pdf)
[16](http://arxiv.org/pdf/2301.09515.pdf)
[17](https://arxiv.org/pdf/2112.05744v3.pdf)
[18](https://arxiv.org/pdf/2303.14389.pdf)
[19](https://arxiv.org/html/2412.12888v1)
[20](http://arxiv.org/pdf/2112.10752.pdf)
[21](https://www.emergentmind.com/topics/masked-diffusion-models)
[22](https://pure.kaist.ac.kr/en/publications/cross-view-masked-diffusion-transformers-for-person-image-synthes/)
[23](https://proceedings.neurips.cc/paper_files/paper/2024/file/ecd92623ac899357312aaa8915853699-Paper-Conference.pdf)
[24](https://www.ijfmr.com/papers/2025/3/45572.pdf)
[25](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Exploring_the_Deep_Fusion_of_Large_Language_Models_and_Diffusion_CVPR_2025_paper.pdf)
[26](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_Masked_Diffusion_Transformer_is_a_Strong_Image_Synthesizer_ICCV_2023_paper.pdf)
[27](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffit/)
[28](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/sana/)
[29](https://arxiv.org/html/2511.08823v1)
[30](https://www.arxiv.org/abs/2511.23021)
[31](https://arxiv.org/abs/2410.13925)
[32](https://arxiv.org/html/2510.27135v1)
[33](https://arxiv.org/abs/2409.02908)
[34](https://arxiv.org/html/2507.01467v2)
[35](https://arxiv.org/abs/2312.02139)
[36](https://arxiv.org/abs/2511.19152)
[37](https://arxiv.org/html/2512.04969)
[38](https://arxiv.org/pdf/2403.03206.pdf)
[39](https://arxiv.org/abs/2403.12008)
[40](https://link.springer.com/10.1007/s11263-024-02240-2)
[41](https://dl.acm.org/doi/10.1145/3690624.3709392)
[42](https://arxiv.org/abs/2409.04768)
[43](https://arxiv.org/abs/2405.05667)
[44](https://arxiv.org/abs/2412.04296)
[45](https://ieeexplore.ieee.org/document/10943743/)
[46](https://ieeexplore.ieee.org/document/10678648/)
[47](https://ieeexplore.ieee.org/document/10587849/)
[48](https://arxiv.org/abs/2410.04171)
[49](http://arxiv.org/pdf/2405.09806.pdf)
[50](https://arxiv.org/pdf/2406.18547.pdf)
[51](http://arxiv.org/pdf/2402.05035v1.pdf)
[52](http://arxiv.org/pdf/2412.04106.pdf)
[53](https://arxiv.org/html/2504.06897v1)
[54](https://arxiv.org/html/2409.14128v1)
[55](https://arxiv.org/html/2410.15027v1)
[56](https://www.emergentmind.com/topics/multimodal-diffusion-transformer-mmdit)
[57](https://proceedings.neurips.cc/paper_files/paper/2024/file/be30024e7fa2c29cac7a6dafcbb8571f-Paper-Conference.pdf)
[58](https://www.sciencedirect.com/science/article/abs/pii/S0893608023007529)
[59](https://proceedings.neurips.cc/paper_files/paper/2024/file/f1aa53cac69ef6980ca4a911ffcf278b-Paper-Conference.pdf)
[60](https://openreview.net/pdf/2c9877508b7505c4c6e730f2b3f8e054edbe33ea.pdf)
[61](https://openaccess.thecvf.com/content/WACV2025/papers/Nauen_Which_Transformer_to_Favor_A_Comparative_Analysis_of_Efficiency_in_WACV_2025_paper.pdf)
[62](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Structure-Guided_Adversarial_Training_of_Diffusion_Models_CVPR_2024_paper.pdf)
[63](https://arxiv.org/html/2506.02528v1)
[64](https://arxiv.org/html/2308.09372v4)
[65](https://arxiv.org/html/2411.03177v1)
[66](https://arxiv.org/html/2408.15178v1)
[67](https://arxiv.org/html/2510.03206v1)
[68](https://arxiv.org/html/2507.21156v1)
[69](https://arxiv.org/html/2504.16064v1)
[70](https://arxiv.org/abs/2303.14389)
[71](https://arxiv.org/html/2507.01467v1)
[72](https://transformerstheory.github.io/pdf/14_piskorz_et_al.pdf)
[73](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393140/)
[74](https://kimjy99.github.io/categories/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/CV/)
