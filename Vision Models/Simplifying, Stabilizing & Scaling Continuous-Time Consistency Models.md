
# Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models

## 1. 핵심 주장과 주요 기여

### 1.1 문제 정의 및 제안된 해결책

논문의 핵심 주장은 **연속시간 일관성 모델(Continuous-Time Consistency Models, CMs)은 이론적으로 우월하지만 훈련 불안정성으로 인해 실용화되지 못했다**는 것이다. 저자들은 이 불안정성의 근본 원인을 분석하고, 세 가지 차원에서의 개선을 제시한다:

1. **TrigFlow**: 편미분 방정식 공식화의 단순화
2. **아키텍처 개선**: 시간 조건화 및 정규화 최적화
3. **훈련 목표 재구성**: 적응형 가중치 및 정규화 추가

### 1.2 주요 기여도 (Contributions)

| 기여 항목 | 설명 | 임팩트 |
|---------|------|--------|
| **TrigFlow 프레임워크** | EDM과 Flow Matching을 통합하는 단순화된 공식 | 이론적 명확성 증대, 분석 용이성 |
| **연속시간 안정화** | 불안정성의 근본 원인 파악 및 5가지 기술적 해결 | 최초의 안정적 연속시간 CM 훈련 |
| **대규모 확장** | 1.5B 파라미터 모델까지 훈련 가능 증명 | CM의 스케일 한계 돌파 |
| **성능 달성** | 2-step으로 state-of-the-art FID 달성 | 실용적 생성 모델로서의 가치 입증 |

***

## 2. 문제 정의, 제안 방법, 모델 구조

### 2.1 해결하고자 하는 문제

#### A. 이산시간 CM의 근본적 한계
기존 연속시간 이전의 모든 CM은 **이산화 오류(discretization error)**를 포함한다:

$$\text{이산시간 CM 목표: } \mathbb{E}_{x_t,t}[w(t)d(f_\theta(x_t, t), f_{\theta^-}(x_{t-\Delta t}, t-\Delta t))]$$

이 공식은:
- 수치 ODE 솔버(numerical ODE solver)를 사용하여 $x_{t-\Delta t}$ 추정 필요
- $\Delta t$ 크기에 민감한 하이퍼파라미터 튜닝 필요
- $\Delta t \to 0$일 때 수렴하지만, 실제로는 유한한 $\Delta t$만 사용 가능

#### B. 연속시간 CM의 훈련 불안정성
연속시간 CM의 공식(Song et al., 2023):

$$\nabla_\theta \mathbb{E}_{x_t,t}\left[w(t)f_\theta^T(x_t, t) \frac{df_{\theta^-}(x_t, t)}{dt}\right]$$

**불안정성의 원인**:

$$\frac{df_{\theta^-}(x_t, t)}{dt} = \underbrace{-\cos(t)(\sigma_d F_{\theta^-} - \frac{dx_t}{dt})}_{\text{상대적으로 안정}} - \underbrace{\sin(t)(x_t + \sigma_d\frac{dF_{\theta^-}}{dt})}_{\text{극히 불안정}}$$

특히 $\sin(t)\frac{\partial F_{\theta^-}}{\partial t}$ 항이 시간 단계에서 극심한 진동을 유발

#### C. 대규모 모델 훈련 실패
- Song et al. (2023): 최대 500M 파라미터 정도에서만 시도
- Song & Dhariwal (2023): ImageNet 512×512에서 iCT-deep으로 ~4.5 FID에 멈춤
- 1B 이상의 모델: 훈련 불가능 상태

### 2.2 제안된 방법의 완전한 수식화

#### A. TrigFlow 프레임워크

**Diffusion Process**:
$$x_t = \cos(t)x_0 + \sin(t)z, \quad t \in [0, \pi/2], \quad z \sim \mathcal{N}(0, \sigma_d^2 I)$$

**특성**: 
- $x_0$에서 $x_{\pi/2} \sim \mathcal{N}(0, \sigma_d^2 I)$로의 선형 보간
- 삼각함수의 간결한 성질로 미분 계산 용이

**Probability Flow ODE**:
$$\frac{dx_t}{dt} = \sigma_d F_\theta\left(\frac{x_t}{\sigma_d}, c_{\text{noise}}(t)\right)$$

**Diffusion 모델 목표**:

$$L_{\text{Diff}}(\theta) = \mathbb{E}_{x_0,z,t}\left[\left\|\sigma_d F_\theta\left(\frac{x_t}{\sigma_d}, c_{\text{noise}}(t)\right) - v_t\right\|_2^2\right]$$

여기서 $v_t = \cos(t)z - \sin(t)x_0$는 속도(velocity) 벡터

**Consistency 모델 파라미터화**:

$$f_\theta(x_t, t) = \cos(t)x_t - \sin(t)\sigma_d F_\theta\left(\frac{x_t}{\sigma_d}, c_{\text{noise}}(t)\right)$$

경계 조건: $f_\theta(x, 0) = x$ 자동 만족

#### B. 연속시간 CM 개선된 훈련 목표

Tangent 함수의 분해:
$$\frac{df_{\theta^-}(x_t,t)}{dt} = -\cos(t)\left(\sigma_d F_{\theta^-} - \frac{dx_t}{dt}\right) - \sin(t)\left(x_t + \sigma_d\frac{dF_{\theta^-}}{dt}\right)$$

**1단계: Tangent Normalization** - 극심한 기울기 분산 억제

$$\frac{df_{\theta^-}}{dt} \rightarrow \frac{df_{\theta^-}/dt}{||df_{\theta^-}/dt|| + 0.1}$$

또는 clip version: $\text{clip}(df_{\theta^-}/dt, -1, 1)$

**2단계: Adaptive Weighting** - 시간 단계별 손실 분산 균형

$$L_{\text{sCM}}(\theta, \phi) = \mathbb{E}_{x_t,t}\left[e^{w_\phi(t)}\frac{1}{D}\left\|F_\theta\left(\frac{x_t}{\sigma_d}, t\right) - F_{\theta^-}\left(\frac{x_t}{\sigma_d}, t\right) - \cos(t)\frac{df_{\theta^-}(x_t, t)}{dt}\right\|_2^2 - w_\phi(t)\right] \quad (8)$$

여기서:
- $e^{w_\phi(t)}$는 시간 단계별 손실 크기 적응
- 사전 가중치: $w(t) = \frac{1}{\sigma_d \tan(t)}$ 통합

**3단계: JVP Rearrangement** - 수치 오버플로우 방지

$$\cos(t)\sin(t)\frac{dF_{\theta^-}}{dt} = (\nabla_{x_t/\sigma_d}F_{\theta^-}) \cdot (\cos(t)\sin(t)\frac{dx_t}{dt}) + \partial_t F_{\theta^-} \cdot (\cos(t)\sin(t)\sigma_d)$$

FP16 훈련에서 중간 레이어 오버플로우 해결

**4단계: Tangent Warmup** - 초기 훈련 안정화

$$\sin(t) \rightarrow r \cdot \sin(t), \quad r = \min(1, \text{iterations}/10000)$$

### 2.3 모델 구조 및 아키텍처 개선

#### A. Time Conditioning 개선

**기존 EDM 공식의 문제**:
$$c_{\text{noise}}(t) = \log(\sigma_d \tan(t)) \Rightarrow \sin(t) \cdot \partial_t c_{\text{noise}} = \frac{1}{\cos(t)} \rightarrow \infty \text{ as } t \to \pi/2$$

**sCM의 해결책**:
$$c_{\text{noise}}(t) = t \quad \text{(Identity transformation)}$$

결과: 시간 도함수의 안정성 Figure 4에서 명확히 증명
- EDM (Fourier scale 16.0): 불안정
- EDM (positional embedding): 개선되지만 여전히 문제
- TrigFlow (positional embedding): 안정적

#### B. Adaptive Double Normalization

**표준 AdaGN** (Dhariwal & Nichol, 2021):
$$y = \text{norm}(x) \odot s(t) + b(t)$$

**문제**: CM 훈련에서 발산 유발 (Song & Dhariwal, 2023)

**sCM의 개선**:
$$y = \text{norm}(x) \odot \text{pnorm}(s(t)) + \text{pnorm}(b(t))$$

Pixel normalization $\text{pnorm}(a) = a / \sqrt{\text{mean}(a^2) + \epsilon}$를 두 번 적용하여:
- AdaGN의 표현력 유지
- CM 훈련의 안정성 확보

#### C. Network Architecture 선택

- **주요 선택**: EDM2 기반 (Karras et al., 2024)
- **이유**: 
  - U-Net 구조의 검증된 효율성
  - Efficient backbone (CNN 기반)
  - Transformer 대비 ImageNet 우월성 입증
- **크기**: S, M, L, XL, XXL (280M ~ 1.5B 파라미터)

***

## 3. 성능 향상 분석

### 3.1 벤치마크 성능

#### CIFAR-10

| 방법 | 1-step FID | 2-step FID | 참고 |
|------|-----------|-----------|------|
| Song et al. (2023) CD | 3.55 | 2.93 | 기존 최선 |
| Song & Dhariwal (2023) iCT-deep | 2.51 | 2.24 | 개선된 이산시간 |
| Geng et al. (2024) ECT | 3.60 | 2.11 | 더 나은 distillation |
| **sCM (ours) sCD** | **3.66** | **2.52** | 연속시간, 우수 |
| **sCM (ours) sCT** | **2.85** | **2.06** | 최고 성능 ⭐ |

**분석**:
- sCT 2-step FID 2.06은 기존 대비 2.1% 개선
- 이산시간 이론의 한계를 연속시간으로 극복

#### ImageNet 64×64

| 방법 | 1-step FID | 2-step FID | 파라미터 |
|------|-----------|-----------|---------|
| Song et al. (2023) CD | 6.20 | 4.70 | 작음 |
| Geng et al. (2024) ECT | 2.49 | 1.67 | ~500M |
| **sCM sCD (S size)** | 2.44 | 1.66 | 280M |
| **sCM sCD (XL size)** | **2.40** | **1.93** | 1.1B |
| **sCM sCT (XL size)** | **2.04** | **1.48** | 1.1B ⭐ |
| EDM2-XXL (teacher) | - | 1.33 | 1.5B |

**주요 통찰**:
- sCD의 교사 모델 대비 FID 비율이 모든 모델 크기에서 일정 (Figure 6b)
- sCT는 소규모에서 효율적, 대규모에서 분산 증가
- 2-step sCT는 교사 모델 FID의 111% 수준 (비교: 다른 방법 120~150%)

#### ImageNet 512×512 (최대 규모 실험)

| 방법 | 1-step FID | 2-step FID | 파라미터 |
|------|-----------|-----------|---------|
| EDM2-XXL (teacher) | - | 1.73 | 1.5B |
| **sCM sCD-XXL** | 2.28 | **1.88** | 1.5B ⭐ |
| **sCM sCT-XXL** | 4.29 | 3.76 | 1.5B |

**critical insight**:
- 2-step sCD: 교사 모델 대비 1.88/1.73 = 1.087 (10.9% 격차)
- 논문의 claim: "10% 이내 격차 달성" 달성
- sCT의 latent space 한계 (높은 분산) 명확

### 3.2 스케일링 동역학 (Figure 6)

연속시간 CM의 **스케일 일관성** 증명:

$$\text{FID Ratio} = \frac{\text{FID}_{sCD}}{\text{FID}_{\text{teacher}}}$$

결과:
- 모든 모델 크기 (S, M, L, XL, XXL)에서 비율 약 1.10~1.15
- Step 수 증가 시 비율 감소 (수렴)
- **의의**: sCD가 교사 모델과 동일한 스케일링 법칙 따름

### 3.3 VSD와의 비교 (Figure 7)

**정밀도(Precision) vs 재현율(Recall) 분석**:

| 가이던스 수준 | Precision | Recall | FID |
|-------------|-----------|--------|-----|
| 1.0 (기준) | 0.87 | 0.60 | 5.2 |
| **VSD 1-step** | 0.89 ↑ | 0.54 ↓ | 6.1 |
| **sCD 2-step** | 0.87 | 0.60 | 4.2 |
| **Diffusion 기준** | 0.85 | 0.62 | 5.0 |

**결론**:
- VSD: 높은 가이던스에서 모드 붕괴 (recall ↓↓)
- sCM: 다양성-품질 균형 유지

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 이론적 근거

**정리 (논문 Figure 5c 실증)**:
$$\lim_{\Delta t \to 0} \text{이산시간 CM 성능} = \text{연속시간 CM 성능}$$

실험 결과:
- N (이산화 스텝)가 증가할 때:
  - N ≤ 1024: 성능 개선
  - N > 1024: 수치 정밀도 문제로 악화
  - 연속시간: 항상 최고 성능

### 4.2 확장 가능성 차원별 분석

#### A. 모델 크기 확장
✓ **검증됨**: S(280M) → XXL(1.5B) 안정적 성능
- sCD: 모든 크기에서 일정한 FID 비율
- sCT: 분산 증가 (latent space 인코더/디코더 최적화 필요)

#### B. 해상도 확장  
✓ **검증됨**: 
- CIFAR-10 (32×32)
- ImageNet 64×64
- ImageNet 512×512 (최대 실험)

✓ **예상 확장**:
- 1024×1024: 아키텍처 수정으로 가능성 높음
- 비디오 생성: 시간축 확장 가능성 높음

#### C. 아키텍처 다양화
△ **제약 있음**:
- Adaptive group norm 수정 필요
- Positional embedding 요구
- Transformer 아키텍처 호환성 미검증

#### D. 데이터 도메인
△ **부분 검증**:
- 이미지(픽셀/latent): 검증됨
- 3D: 미탐색
- 오디오: 미탐색
- 비디오: 미탐색 (기술적으로 가능성 높음)

### 4.3 실제 일반화 한계

#### 명시된 한계 (논문 Limitations)

1. **sCT의 Latent Space 비효율성**
   ```
   원인: 인코더/디코더의 ill-conditioned 매핑
   해결책: 더 나은 VAE/VQGAN 개발 필요
   ```

2. **아티팩트 존재**
   ```
   ImageNet의 클래스 레이블 조건화 한계
   → Caption 기반 데이터에서 개선 예상
   ```

3. **CFG(Classifier-Free Guidance) 불호환**
   ```
   sCT는 CFG 미지원
   sCD는 지원하나 안정성 확인 필요
   ```

4. **아키텍처 의존성**
   ```
   네트워크 특정 수정(adaptive norm 등) 필수
   일반적 적용성 제한
   ```

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 연구 진화 시간선

```
2020-2022: 기초 - Diffusion 기반 설립 (DDPM, EDM)
    ↓
2023: 변곡점 - Consistency Models 도입
    ├─ Song et al.: 첫 CM (이산시간)
    ├─ Song & Dhariwal: 개선 기법 (이차 미분 고려)
    └─ Lipman et al.: Flow Matching (대안 패러다임)
    ↓
2024: 고도화 - 안정화 및 스케일링
    ├─ Karras et al.: EDM2 (적응형 가중치, 아키텍처)
    ├─ Geng et al.: ECT (더 나은 distillation)
    ├─ This paper: sCM (연속시간 안정화 ⭐)
    ├─ Yang et al.: Consistency-FM (velocity consistency)
    └─ Wang et al.: VSD (또 다른 distillation)
    ↓
2025: 통합 - Hybrid 접근
    ├─ LCFM: Latent + Flow Matching + Consistency
    ├─ SCFM: Flow Matching distillation 고도화
    └─ sLCT: Latent CM 훈련 안정화
```

### 5.2 주요 연구별 비교표

| 논문 | 시기 | 핵심 기여 | 최대 규모 | 주요 성과 | sCM과의 관계 |
|------|------|---------|---------|---------|-----------|
| **Song et al.** | 2023.3 | 첫 번째 CM | ~400M | FID 3.55 (CIFAR) | 기초 개념 제공 |
| **Song & Dhariwal** | 2023.10 | iCT, 이차 고려 | ~800M | FID 2.24 (CIFAR) | 불안정 연속시간 모델 |
| **Lipman et al.** | 2023.1 | Flow Matching | ~600M | 우수한 샘플 품질 | sCM의 TrigFlow와 유사 |
| **Karras et al. EDM2** | 2024.3 | 적응형 가중치 | 1.5B | FID 1.81 (IN512) | 적응형 가중치 상용 |
| **Geng et al. ECT** | 2024.6 | 더 나은 타겟 | ~1B | FID 2.11 (CIFAR) | 여전히 이산시간 |
| **sCM (This)** | 2024.10 | 연속시간 안정화 | **1.5B** | **FID 1.88 (IN512)** | **최고 수준** ⭐ |
| **Yang et al. CFM** | 2024.7 | Velocity consistency | ~700M | 빠른 수렴 | 유사한 안정화 목표 |
| **Wang et al. VSD** | 2024.5/6 | 직접 최적화 | ~600M | 높은 FID | 모드 붕괴 문제 |
| **LCFM** | 2025.1 | Hybrid 접근 | ~1B | 이론적 보장 | 상호 보완 가능 |
| **SCFM** | 2025.2 | FM distillation | ~5B | 3-step 생성 | 최신 FM 확장 |

### 5.3 sCM이 해결한 미해결 문제

#### 문제 1: 연속시간 CM의 훈련 불안정성
| 논문 | 해결? | 방법 |
|------|------|------|
| Song et al. (2023) | ✗ | "불안정성 발견, 미해결" |
| Song & Dhariwal (2023) | △ | AdaGN 제거 (임시방편) |
| Geng et al. (2024) | △ | ECT로 개선 (이산시간만) |
| **sCM** | **✓** | TrigFlow + 5가지 기술 |

#### 문제 2: 이산화 오류의 원칙적 해결
| 접근 | 이산화 오류 | 제한사항 |
|------|-----------|--------|
| 이산시간 CM | 있음 | Δt > 0 필수 |
| 고차 solver | 약간 감소 | 계산 비용 증가 |
| **연속시간 CM (sCM)** | **없음** | 훈련 불안정 → 이제 해결! |

#### 문제 3: 대규모 모델 확장 불가능
| 모델 | 최대 규모 | 상태 |
|------|---------|------|
| Song et al. | ~500M | 제한됨 |
| Song & Dhariwal | ~800M | 제한됨 |
| Geng et al. ECT | ~1B | 근처 |
| **sCM** | **1.5B** | ✓ 첫 번째 |

### 5.4 패러다임 비교: 3가지 고속 샘플링 방식

#### A. 이산시간 CM (Song et al., 2023 ~ Geng et al., 2024)
```
장점: 훈련 상대적 안정
단점: 이산화 오류, 하이퍼파라미터 민감
예시: iCT-deep (FID 2.24), ECT (FID 2.11)
```

#### B. 연속시간 CM (sCM - 이 논문)
```
장점: 원칙적 우월성, 큰 규모 지원, 이산화 오류 없음
단점: 훈련 복잡성 높음
성과: FID 2.06 (sCT), 1.88 (sCD)
```

#### C. Flow Matching + Consistency (LCFM, CFM, SCFM - 2024~2025)
```
장점: 더 직선적 궤적, 이론적 수렴 보장
단점: sCM보다 최근 (비교 부족)
성과: 높은 효율성, 멀티모달 응용
```

***

## 6. 앞으로의 연구에 미치는 영향과 고려사항

### 6.1 즉각적 영향 (Short-term: 2025)

#### A. 확산 모델 가족의 확대
- **Impact**: 연속시간 CM이 이제 실용적 → 새로운 연구 방향 열림
- **응용**: 
  - 멀티모달 생성 (3D, 비디오, 음성)
  - Domain adaptation
  - 세밀한 제어 (guided generation)

#### B. 산업 적용 가능성 증가
- **2-step 생성으로 배포 가능**
  - 모바일 디바이스 (정제된 버전)
  - 실시간 생성 (비디오 처리)
  - 저지연 시스템

#### C. 이론적 기여의 파급
- **TrigFlow의 통일 프레임워크**
  - EDM, Flow Matching, Velocity Prediction을 하나로
  - 향후 변형 모델의 설계 기초

### 6.2 중기 영향 (Medium-term: 2025-2026)

#### A. Latent Space 최적화 연구
sCM의 한계: **sCT의 Latent Space 비효율성**

```
현재: Encoder/Decoder → Ill-conditioned mapping
향후 연구:
1. Better VAE/VQGAN 설계
2. sCT 특화 인코더 학습
3. Hybrid: sCD for latent, sCT for pixel
```

#### B. 아키텍처 확장성 연구
```
sCM 현상태: Adaptive norm + Positional embedding 필수
향후:
- Transformer 완전 호환성
- Vision Transformer (ViT) 기반 모델
- 하이브리드 아키텍처 (CNN-ViT)
```

#### C. 도메인 확장
| 도메인 | 가능성 | 난제 |
|--------|------|------|
| **3D 생성** | 높음 | 기하학 표현 정의 |
| **비디오** | 높음 | 시간축 일관성 |
| **음성** | 중간 | Spectrogram vs raw audio |
| **분자** | 중간 | Graph 구조 처리 |
| **로봇** | 중간 | Action space 설계 |

### 6.3 장기 영향 (Long-term: 2026+)

#### A. 생성 모델 패러다임 통합
```
현재 상황:
├─ Diffusion (느리지만 안정적)
├─ Flow Matching (빠르지만 새로움)
├─ GAN (빠르지만 불안정)
└─ Autoregressive (느리지만 정확)

2026 이후 전망:
→ 하이브리드 프레임워크 (sCM + FM + Consistency)
→ Task별 최적화 모델 생태계
```

#### B. Theoretical Understanding 심화
```
sCM이 열어주는 이론 문제:
1. 연속시간 PF-ODE의 명시적 해석
2. Consistency 학습의 최적성 조건
3. Generalization bounds
```

#### C. 에너지 효율성
```
2-step 생성 = 기존 대비 10-100배 빠름
→ 탄소 발자국 극적 감소
→ 엣지 디바이스 배포 현실화
```

### 6.4 향후 연구 시 고려할 점 (Critical Considerations)

#### 1. **Latent Space 인코더/디코더 최적화**

**문제**: sCT의 높은 분산은 VAE/VQGAN의 ill-conditioned 특성에서 비롯

```python
# 연구 방향:
1. Variational bottleneck 약화 (β-VAE 접근)
2. 연속시간 특화 인코더 설계
3. Differentiable quantization (VQ-GAN-2)
```

**Expected Outcome**: sCT를 sCD 수준으로 끌어올리면 
- 사전훈련된 모델 불필요 (완전 독립 훈련 가능)
- 도메인 특화 모델 빠른 개발

#### 2. **Architecture-specific modifications 통일**

**현재 문제**: Adaptive norm, positional embedding 등 임시방편이 많음

```
개선 방향:
1. Normalization 기법의 일반화 이론
2. Time-conditioning 최적 설계
3. Guidance-compatible architecture
```

#### 3. **다중 스텝 생성 최적화**

**현재 상황**: 2-step이 목표, 1-step은 FID 격차 큼

```
향후 연구:
- 1-step과 2-step의 trade-off 분석
- 적응형 step 선택 (동적 계산)
- 계층적 생성 (coarse-to-fine)
```

#### 4. **Guidance Mechanism의 재설계**

**VSD 문제**: High guidance에서 모드 붕괴

```
sCM의 잠재력:
- Guidance 호환 연속시간 CM
- Semantic control의 정밀성
- Instruction-following generation
```

#### 5. **Theoretical Convergence Analysis**

**현재 부족**: 수치적 증거는 있지만 이론적 수렴 보장 없음

```
해결 필요:
1. Lipschitz continuity 증명
2. Convergence rate 분석 (O(1/T^α) 형태)
3. Generalization error bounds
```

***

## 7. 결론 및 전략적 의의

### 7.1 논문의 전략적 위치

sCM은 **일관성 모델 연구의 임계점**을 나타낸다:

```
2023 (발견): CM의 개념적 우월성 입증
  ↓
2023-2024 (고민): 연속시간 안정성 미해결
  ↓
2024.10 (돌파): sCM으로 연속시간 안정화 ⭐
  ↓
2025+ (확산): 다양한 도메인으로 응용 전개
```

### 7.2 성능의 의미

| 메트릭 | 달성 | 의미 |
|--------|------|------|
| FID 2.06 (CIFAR, 2-step) | 기존 대비 2% 개선 | 이산시간 한계 극복 |
| FID 1.88 (IN512, 2-step) | 교사 모델 대비 10% 격차 | 실용적 대체 가능 |
| 1.5B 파라미터 | 역대 최대 CM | 스케일 한계 돌파 |
| 안정적 훈련 | 5개의 기술 통합 | 공학적 성숙도 증명 |

### 7.3 미래 전망

**긍정적 시나리오 (2025-2026)**:
- Latent space 최적화로 sCT 개선 → 사전훈련 불필요
- 다양한 도메인 적용 (비디오, 3D, 음성)
- 멀티모달 일관성 모델 출현
- 1-step 생성의 실현 (현재는 2-step 최적)

**도전과제**:
- 아키텍처 복잡성으로 인한 채택 저해
- Latent space 한계의 기본적 해결 필요
- 이론적 이해 부족

**결론**: sCM은 **기술적으로는 해결책을 제시했지만, 실무 적용을 위해서는 추가 최적화가 필수**이다. 다만 연속시간 CM의 가능성을 증명함으로써 향후 10년 생성 모델 연구에 새로운 방향을 제시했다.

***

<span style="display:none">[^1_1][^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_4][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_5][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_6][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_8][^1_80][^1_81][^1_82][^1_83][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2410.11081v2.pdf

[^1_2]: http://arxiv.org/pdf/2406.04485.pdf

[^1_3]: http://arxiv.org/pdf/2310.14189v1.pdf

[^1_4]: https://arxiv.org/abs/2303.01469

[^1_5]: https://arxiv.org/pdf/2502.17440.pdf

[^1_6]: https://arxiv.org/pdf/2301.04655.pdf

[^1_7]: https://arxiv.org/html/2503.08117v1

[^1_8]: https://arxiv.org/pdf/2307.01898.pdf

[^1_9]: http://arxiv.org/pdf/2407.13072.pdf

[^1_10]: https://syncedreview.com/2023/03/08/openais-consistency-models-support-fast-one-step-generation-for-diffusion-models/

[^1_11]: https://proceedings.mlr.press/v202/zheng23d/zheng23d.pdf

[^1_12]: https://www.openaccess.thecvf.com/content/CVPR2025/papers/Schusterbauer_Diff2Flow_Training_Flow_Matching_Models_via_Diffusion_Model_Alignment_CVPR_2025_paper.pdf

[^1_13]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5348747

[^1_14]: https://qsh-zh.github.io/deis/

[^1_15]: https://www.youtube.com/watch?v=7NNxK3CqaDk

[^1_16]: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Fast_ODE-based_Sampling_for_Diffusion_Models_in_Around_5_Steps_CVPR_2024_paper.pdf

[^1_17]: https://openreview.net/forum?id=PqvMRDCJT9t

[^1_18]: https://cacm.acm.org/blogcacm/the-challenge-of-consistency-in-generative-ai-will-we-adapt-or-fix-the-system/

[^1_19]: https://arxiv.org/abs/2211.13449

[^1_20]: https://arxiv.org/abs/2506.02221

[^1_21]: https://papers.cumincad.org/data/works/att/caadria2025_567.pdf

[^1_22]: https://arxiv.org/abs/2204.13902

[^1_23]: https://arxiv.org/abs/2506.02070

[^1_24]: https://arxiv.org/pdf/2510.11677.pdf

[^1_25]: https://arxiv.org/abs/2106.00132

[^1_26]: https://arxiv.org/html/2510.17858v1

[^1_27]: https://pubmed.ncbi.nlm.nih.gov/40966479/

[^1_28]: https://arxiv.org/abs/2402.17376?utm

[^1_29]: https://arxiv.org/html/2510.20771v1

[^1_30]: https://pdfs.semanticscholar.org/4fa5/eccda27b3ff4932ec7bc46d60829484dc4f9.pdf

[^1_31]: https://arxiv.org/abs/2401.01008

[^1_32]: https://arxiv.org/html/2512.15657v1

[^1_33]: https://arxiv.org/html/2510.13852v1

[^1_34]: https://arxiv.org/abs/2410.18804

[^1_35]: https://arxiv.org/html/2512.02826v1

[^1_36]: https://arxiv.org/abs/2505.18825

[^1_37]: https://arxiv.org/abs/2402.09970

[^1_38]: https://arxiv.org/html/2506.08604

[^1_39]: https://www.semanticscholar.org/paper/8b7cce220c3b19f9b2d4a6c531907ed3b592b55e

[^1_40]: https://arxiv.org/abs/2311.05556

[^1_41]: https://doi.apa.org/doi/10.1037/xge0001344

[^1_42]: https://ieeexplore.ieee.org/document/10331300/

[^1_43]: https://doi.apa.org/doi/10.1037/tra0001499

[^1_44]: https://arxiv.org/abs/2310.20003

[^1_45]: https://archive.johs.org.uk/article/doi/10.54531/tzfd6375

[^1_46]: https://arxiv.org/abs/2306.05004

[^1_47]: https://doi.apa.org/doi/10.1037/pspp0000487

[^1_48]: https://doi.apa.org/doi/10.1037/tra0001465

[^1_49]: http://arxiv.org/pdf/2406.00356.pdf

[^1_50]: http://arxiv.org/pdf/2310.04378.pdf

[^1_51]: https://arxiv.org/abs/2312.09109

[^1_52]: https://arxiv.org/html/2408.02993

[^1_53]: http://arxiv.org/abs/2503.12615

[^1_54]: https://arxiv.org/html/2503.08377v1

[^1_55]: https://arxiv.org/html/2502.01441v2

[^1_56]: http://arxiv.org/pdf/2405.02791.pdf

[^1_57]: https://kimjy99.github.io/논문리뷰/latent-consistency-model/

[^1_58]: https://proceedings.neurips.cc/paper_files/paper/2024/file/dd540e1c8d26687d56d296e64d35949f-Paper-Conference.pdf

[^1_59]: https://www.emergentmind.com/topics/latent-consistency-flow-matching-lcfm

[^1_60]: https://www.youtube.com/watch?v=y0Tw9Zb4Sy4

[^1_61]: https://arxiv.org/html/2506.13763v1

[^1_62]: https://neurips.cc/virtual/2025/poster/116548

[^1_63]: https://arxiv.org/abs/2310.04378

[^1_64]: https://openreview.net/pdf?id=sn1kl4Dbm7

[^1_65]: https://liner.com/review/consistency-flow-matching-defining-straight-flows-with-velocity-consistency

[^1_66]: https://blog.outta.ai/17

[^1_67]: https://openreview.net/forum?id=OHZRUCa1HW

[^1_68]: https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Fast_Image_Super-Resolution_via_Consistency_Rectified_Flow_ICCV_2025_paper.pdf

[^1_69]: https://github.com/luosiallen/latent-consistency-model

[^1_70]: https://sander.ai/2024/06/14/noise-schedules.html

[^1_71]: https://openreview.net/forum?id=bS76qaGbel

[^1_72]: https://arxiv.org/pdf/2311.05556.pdf

[^1_73]: https://arxiv.org/pdf/2510.12537.pdf

[^1_74]: https://arxiv.org/html/2508.14807v1

[^1_75]: https://arxiv.org/html/2509.01819v1

[^1_76]: https://arxiv.org/pdf/2310.04378.pdf

[^1_77]: https://arxiv.org/html/2410.11081v1

[^1_78]: https://openaccess.thecvf.com/content/ICCV2025/papers/You_Consistency_Trajectory_Matching_for_One-Step_Generative_Super-Resolution_ICCV_2025_paper.pdf

[^1_79]: https://arxiv.org/html/2310.04378

[^1_80]: https://arxiv.org/html/2508.07926v1

[^1_81]: https://arxiv.org/html/2510.12537v1

[^1_82]: https://arxiv.org/html/2502.03500v2

[^1_83]: https://github.com/NVlabs/edm2/blob/main/README.md

