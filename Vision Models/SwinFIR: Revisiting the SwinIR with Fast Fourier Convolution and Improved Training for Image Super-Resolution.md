# SwinFIR: Revisiting the SwinIR with Fast Fourier Convolution and Improved Training for Image Super-Resolution

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

SwinFIR은 SwinIR의 **윈도우 기반 로컬 어텐션의 한계**(초기 레이어에서 글로벌 정보 포착 불가)를 극복하기 위해, **Fast Fourier Convolution(FFC)**을 도입하여 이미지 전체 수준의 수용 영역(image-wide receptive field)을 확보하고, 다양한 학습 개선 기법을 종합적으로 적용해 이미지 초해상도(SR) 성능을 향상시킨다는 것이다.

### 주요 기여 (3가지)

| 기여 항목 | 설명 |
|---|---|
| **Spatial Frequency Block (SFB)** | FFC 기반 글로벌 특징 추출기 도입, SwinIR의 Conv 3×3 대체 |
| **데이터 증강 재검토** | 채널 셔플, Mixup 등 픽셀 도메인 증강이 SR 성능 향상에 유효함을 실증 |
| **Feature Ensemble** | 추론 시간 증가 없이 성능을 향상시키는 새로운 앙상블 전략 제안 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

SwinIR은 Swin Transformer 기반으로 우수한 SR 성능을 보이지만, **윈도우 기반 Self-Attention과 Shifted Window 전략**은 근본적으로 **로컬 수용 영역**에 한정된다. 이로 인해:

- 초기 레이어에서 글로벌 컨텍스트를 효과적으로 포착하지 못함
- LAM(Local Attribution Map) 분석에 따르면, SwinIR의 Diffusion Index(DI)가 낮아 활용 픽셀 범위가 제한됨

LAM 실험 결과:

$$\text{DI}_{\text{SwinIR}} = 4.38 \quad < \quad \text{DI}_{\text{EDT}} = 11.95 \quad < \quad \text{DI}_{\text{SwinFIR}} = 19.31$$

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 Spatial Frequency Block (SFB)

SFB는 두 가지 병렬 브랜치로 구성된다.

**전체 출력:**
$$X_{SFB} = H_{SFB}(X) $$

**공간 브랜치 (Spatial Branch):**
$$X_{spatial} = H_{spatial}(X) = H_{CLC}(X) + X $$

여기서 $H_{CLC}(\cdot)$는 헤드와 테일에 3×3 Conv를 가지고, 사이에 LeakyReLU를 적용하는 잔차 모듈이다.

**주파수 브랜치 (Frequency Branch):**
$$X = H_{CL}(X) $$

$$X_{frequency} = H_C(H_{FLF}(X) + X) $$

여기서 $H_{FLF}(\cdot)$는 다음 연산 시퀀스를 포함한다:

$$H_{FLF}: \underbrace{\text{Real FFT2D}}_{\text{공간} \to \text{주파수}} \to \underbrace{\text{Conv + LeakyReLU}}_{\text{주파수 도메인 처리}} \to \underbrace{\text{Inv Real FFT2D}}_{\text{주파수} \to \text{공간}}$$

**최종 융합:**
$$X_{SFB} = H_C([X_{spatial} \| X_{frequency}]) $$

여기서 $\|$는 채널 방향 연결(concatenation), $H_C(\cdot)$는 1×1 Conv이다.

#### 2.2.2 손실 함수 (Charbonnier Loss)

L1, L2 대신 더 안정적인 Charbonnier Loss를 사용한다:

$$L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \sqrt{(SwinFIR(I_L^i, \theta) - I_H^i)^2 + \varepsilon} $$

여기서 $\theta$는 모델 파라미터, $N$은 학습 이미지 수, $\varepsilon$는 수치 안정성을 위한 작은 상수이다.

#### 2.2.3 Feature Ensemble

학습 중 검증 데이터셋에서 우수한 성능을 보인 $n$개의 중간 체크포인트 모델을 가중 평균으로 결합한다:

$$SwinFIR(\theta) = \sum_{i=1}^{n} SwinFIR(\theta)^i \cdot \alpha^i $$

본 논문에서는 균등 가중치 $\alpha = \frac{1}{n}$을 사용한다.

---

### 2.3 모델 구조

```
입력 LR 이미지
      ↓
┌─────────────────────────┐
│  Shallow Feature        │  (Conv 3×3, SwinIR과 동일)
│  Extraction             │
└──────────┬──────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│  Deep Feature Extraction                            │
│                                                     │
│  RSTB₁ → RSTB₂ → RSTB₃ → RSTB₄ → RSTB₅ → SFB    │
│  (각 RSTB = STL × 6 + Conv)                        │
│  (SFB = Spatial Branch ∥ Frequency Branch)          │
└──────────┬──────────────────────────────────────────┘
           ↓
┌─────────────────────────┐
│  HQ Image               │  (Upsampling + Conv)
│  Reconstruction         │
└─────────────────────────┘
      ↓
    SR 이미지
```

**RSTB (Residual Swin Transformer Block) 내부:**
- 여러 STL(Swin Transformer Layer) 스택
- **마지막 Conv 3×3 → SFB로 교체** (SwinFIR의 핵심 변경점)
- 잔차 연결(Residual Connection) 유지

**학습 설정:**
- 옵티마이저: Adam ($\beta_1 = 0.9$, $\beta_2 = 0.99$, weight decay = 0)
- 사전학습: ImageNet (초기 학습률 $2 \times 10^{-4}$, 1,000,000 iterations)
- 파인튜닝: DF2K (학습률 $1 \times 10^{-5}$)
- 윈도우 크기: 12, 패치 크기: 60 (SwinIR 대비 확대)

---

### 2.4 성능 향상

#### Classical SR (×4 스케일) 주요 결과

| 방법 | Urban100 (PSNR) | Manga109 (PSNR) |
|---|---|---|
| SwinIR | 27.45 dB | 32.03 dB |
| EDT† | 27.75 dB | 32.39 dB |
| **SwinFIR† (제안)** | **28.12 dB** | **32.83 dB** |
| 향상 (vs SwinIR) | **+0.67 dB** | **+0.80 dB** |

#### Set5 기준 SwinIR → SwinFIR 진화 경로

$$32.72 \xrightarrow{+SFB} 32.78 \xrightarrow{+Charbonnier} 32.80 \xrightarrow{+Aug} 32.93 \xrightarrow{+LargeWindow} 33.08 \xrightarrow{+Pretrain} 33.20 \xrightarrow{+FE} 33.22 \text{ (dB)}$$

#### Lightweight SR (×4, SwinFIR-T)
- Manga109: SwinIR 대비 **+0.58 dB** 향상 (30.92 → 31.50 dB)
- 비슷한 파라미터 수(약 880K~900K) 유지

#### 스테레오 SR (SwinFIRSSR, ×2)
- Flickr1024: NAFSSR-L 대비 **+0.46 dB** 향상

---

### 2.5 한계

논문에서 명시적으로 기술된 한계와 분석을 통해 파악된 한계는 다음과 같다:

1. **계산 비용**: 윈도우 크기 확대(8→12), 패치 크기 확대(48→60), ImageNet 사전학습 등은 학습 시간과 메모리를 증가시킨다.
2. **어려운 샘플 한계**: 논문 자체에서 "CNN/Transformer 기반 방법 모두 challenging samples에 대해 불충분하다"고 인정한다 (Figure 8 참조).
3. **CutMix/CutMixup 부적합**: 픽셀 도메인 일부 증강 기법은 시각적 연속성을 파괴하여 SR에서 역효과를 낸다.
4. **HSTL 포기**: 파라미터가 2.6배(37.20M vs 14.03M) 증가함에도 불구하고 일부 데이터셋에서만 우세하여 채택하지 않았다.
5. **실세계(Real-World) SR 미검증**: 합성 열화(bicubic downscaling)에만 집중하며, 실제 노이즈·블러 혼합 열화에 대한 일반화는 검증되지 않았다.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화에 기여하는 요소별 분석

#### (1) FFC 기반 전역 수용 영역

FFC는 2D FFT를 통해 주파수 도메인에서 이미지 전체 픽셀을 동시에 고려한다. 이는 수학적으로 컨볼루션 정리에 근거한다:

$$\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}$$

주파수 도메인에서의 연산은 공간 도메인의 **전역 컨볼루션**과 동등하므로, 모델이 단순한 로컬 패턴 암기가 아닌 **구조적 패턴(주기적 텍스처, 에지 방향성 등)**을 학습할 수 있다. 이는 훈련 데이터에 없는 새로운 이미지 구조에 대한 일반화를 향상시킨다.

LAM 실험($\text{DI}_{\text{SwinFIR}} = 19.31$)은 SwinFIR이 더 넓은 범위의 픽셀을 활용함을 직접적으로 보여주며, 이는 **컨텍스트 의존적 복원**을 가능하게 한다.

#### (2) 데이터 증강 전략의 일반화 효과

논문은 픽셀 도메인 데이터 증강이 SR에서도 유효함을 실증한다:

| 증강 방법 | Set5 PSNR (×4) | 일반화 메커니즘 |
|---|---|---|
| Flip + Rotation만 | 32.78 dB | 기하학적 불변성 |
| + RGB Channel Shuffle | 32.89 dB | 색상 편향 제거 |
| + Mixup | **32.93 dB** | 결정 경계 정규화 |

**RGB Channel Shuffle**은 모델이 특정 색상 채널 배열에 과적합되는 것을 방지하여, 다양한 색상 분포를 가진 이미지에 대한 일반화를 향상시킨다.

**Mixup**은 훈련 분포를 선형 보간으로 확장한다:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

$$\lambda \sim \text{Beta}(\alpha, \alpha), \quad \alpha \in (0, \infty)$$

이는 모델이 **소프트한 결정 경계**를 학습하게 하여 분포 외(out-of-distribution) 샘플에 대한 강건성을 높인다.

#### (3) 다양한 Task에서의 일반화 검증 (Robustness 실험)

논문은 SFB 모듈의 다양한 태스크 적용 가능성을 명시적으로 검증한다:

```
SFB 모듈 적용 결과:
├── Classical SR (SwinFIR)       → SOTA 달성
├── Lightweight SR (SwinFIR-T)  → SOTA 달성
├── Stereo SR (SwinFIRSSR)       → SOTA 달성
├── HAT 기반 (HAT_FIR)           → +0.16 dB 향상
├── ABPN 기반 (non-Transformer)  → +0.08 dB 향상
├── 흑백 이미지 노이즈 제거 (DN-G) → 향상 확인
├── 컬러 이미지 노이즈 제거 (DN-C) → 향상 확인
└── JPEG 압축 아티팩트 제거       → 향상 확인
```

특히 **비-Transformer 구조인 ABPN**에서도 SFB 없이 학습 전략만으로 향상을 달성한 점은, 제안 방법이 특정 아키텍처에 종속되지 않음을 시사한다.

#### (4) Feature Ensemble의 일반화 관점

Feature Ensemble은 훈련 중 학습 경로상의 여러 체크포인트를 앙상블한다. 이는 **Stochastic Weight Averaging(SWA)** 이론과 연결되는데:

$$\theta_{ensemble} = \frac{1}{n}\sum_{i=1}^{n} \theta_i$$

이렇게 평균화된 파라미터는 손실 함수의 **더 넓고 평탄한 최솟값(flat minima)**에 위치하는 경향이 있으며, 이는 일반화 성능과 강한 양의 상관관계를 가진다고 알려져 있다 (Hochreiter & Schmidhuber, 1997; Izmailov et al., 2018).

#### (5) ImageNet 사전학습의 일반화 기여

$$\text{PSNR}(\text{SwinFIR, DF2K only}) = 33.20 \to \text{PSNR}(\text{SwinFIR}^{\dagger}, \text{ImageNet+DF2K}) = 33.22 \text{ (Set5 ×4)}$$

사전학습은 특히 텍스처가 복잡한 데이터셋(Urban100, Manga109)에서 더 큰 향상을 보이며, 이는 대규모 데이터로 학습된 일반적 시각 표현이 SR 일반화에 기여함을 보여준다.

### 3.2 일반화 한계 및 미해결 과제

- **도메인 갭**: DF2K(고품질 자연 이미지)로 학습된 모델이 의료 이미지, 위성 이미지, 문서 이미지 등 특수 도메인에서의 일반화는 검증되지 않음
- **열화 유형 한계**: Bicubic 다운샘플링 외 실세계 복합 열화(노이즈+블러+압축)에 대한 종합적 일반화 미검증
- **해상도 스케일 한계**: ×2, ×3, ×4 스케일만 검증, ×8 이상의 대배율 SR 일반화 불명확

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 주요 관련 연구 타임라인

| 연도 | 모델 | 핵심 기술 | Manga109 ×4 PSNR |
|---|---|---|---|
| 2020 | RCAN (Zhang et al., 2018, 이후 벤치마크) | 채널 어텐션 | 31.22 dB |
| 2021 | SwinIR (Liang et al.) | Swin Transformer, W-MSA | 32.03 dB |
| 2021 | IPT (Chen et al.) | ImageNet 사전학습 Transformer | - |
| 2021 | EDT (Li et al.) | Efficient Transformer + 사전학습 | 32.39 dB |
| 2022 | HAT (Chen et al.) | Hybrid Attention Transformer | 32.48 dB |
| 2022 | **SwinFIR (제안)** | **FFC + 개선된 학습** | **32.83 dB** |

### 4.2 아키텍처별 비교 분석

#### vs. SwinIR (Liang et al., ICCV 2021)
- **공통점**: Swin Transformer 기반 3단계 구조 (얕은 특징 추출 → 깊은 특징 추출 → HQ 재구성)
- **차이점**: SwinFIR은 RSTB 내 Conv 3×3을 SFB로 교체 → 글로벌 수용 영역 확보
- **성능 차이** (×4): Urban100 +0.67dB, Manga109 +0.80dB

#### vs. HAT (Chen et al., 2022)
- HAT는 채널 어텐션 + Self-Attention + Overlapping Cross-Attention의 하이브리드
- SwinFIR은 HAT와 상호보완적: SFB를 HAT에 적용(HAT_FIR)하면 추가 향상 (+0.16 dB)
- 이는 FFC가 어텐션 기반 방법과 **직교적(orthogonal)**임을 시사

#### vs. LaMa (Suvorov et al., WACV 2022)
- LaMa는 인페인팅에서 FFC를 처음 효과적으로 적용
- SwinFIR은 LaMa의 Spectral Transform(FB)만 사용 시 성능 저하를 확인하고, **SFB(공간+주파수 혼합)**로 개선: $\text{FB}: 32.65\text{ dB} < \text{SFB}: 32.78\text{ dB}$

#### vs. NAFSSR (Chu et al., CVPRW 2022)
- SwinFIRSSR이 NAFSSR-L 대비 스테레오 SR ×2에서 Middlebury +0.64 dB, Flickr1024 +0.46 dB 향상
- NAFNet은 비선형 활성화 없는 단순 구조 vs. SwinFIR은 Transformer + FFC 복잡 구조

### 4.3 방법론적 트렌드 비교

```
CNN 시대 (2015-2020)            Transformer 시대 (2020-현재)
ResNet → Channel Attention  →  Window Attention → Hybrid Attention
SRCNN → EDSR → RCAN → HAN  →  IPT → SwinIR → EDT → HAT → SwinFIR
                                                              ↑
                                              주파수 도메인 통합 (FFC)
```

SwinFIR은 **Transformer + 주파수 도메인 분석**의 결합이라는 새로운 방향을 제시한다.

---

## 5. 미래 연구에 대한 영향 및 고려사항

### 5.1 앞으로의 연구에 미치는 영향

#### (1) FFC의 Low-Level Vision 표준화 가능성
FFC가 SR 외 노이즈 제거, JPEG 아티팩트 제거, 인페인팅 등 다양한 저수준 시각 태스크에서 일관된 향상을 보임으로써, 향후 저수준 시각 모델의 **표준 구성 요소**로 자리잡을 가능성이 있다.

#### (2) Transformer-Frequency Domain 하이브리드 설계 방향성
Self-Attention(로컬/글로벌 의미적 관계) + FFC(이미지 전역 주파수 특성) + CNN(로컬 텍스처)의 삼중 결합이 최적 성능을 내는 방향으로 아키텍처 연구가 진화할 것으로 예상된다.

#### (3) 데이터 증강 재검토 패러다임
"픽셀 삽입 증강은 SR에 해롭다"는 기존 통념을 깬 것은, 저수준 시각 태스크에서의 데이터 증강 연구를 재활성화할 것으로 보인다.

#### (4) Feature Ensemble의 일반화
훈련 시간 추가 없는 성능 향상 기법으로서, SWA 및 Model Soup(Wortsman et al., 2022)과 연결되는 이 접근법은 다른 컴퓨터 비전 태스크에도 광범위하게 적용될 수 있다.

### 5.2 향후 연구 시 고려해야 할 점

#### 아키텍처 측면

1. **FFC 계산 효율성 개선**: 현재 Real FFT2D는 이미지 해상도에 따라 $O(N \log N)$ 복잡도를 가지지만, 고해상도 이미지에서 메모리 병목이 발생할 수 있다. 희소 FFT나 로컬-글로벌 FFT 하이브리드 연구가 필요하다.

2. **스케일 적응형 주파수 분석**: 고정된 전역 FFT가 아닌, 멀티스케일 주파수 분석(Wavelet Transform 등)과의 결합 가능성을 탐색할 필요가 있다:

$$\mathcal{W}\{f(t)\} = \int_{-\infty}^{\infty} f(\tau) \psi^*\left(\frac{\tau - t}{s}\right) d\tau$$

3. **경량화**: SFB가 추가하는 연산량( $\sim$ 0.9G FLOPs 증가)을 줄이면서 성능을 유지하는 구조 탐색

#### 학습 전략 측면

4. **최적 앙상블 가중치 학습**: 현재 균등 가중치( $\alpha = 1/n$ )를 사용하나, 검증 성능 기반의 **적응적 가중치** 또는 메타러닝 기반 가중치 최적화 연구 필요

5. **증강 전략의 이론적 토대**: Mixup/Channel Shuffle이 SR에서 왜 효과적인지에 대한 이론적 분석 부재. 정보 이론적 또는 PAC-Bayes 프레임워크에서의 분석이 필요하다.

6. **실세계 열화 대응**: Blind SR(열화 유형 미지)을 위한 증강 전략 개발 필요. BSRGAN(Zhang et al., 2021)이나 Real-ESRGAN(Wang et al., 2021) 스타일의 복잡한 열화 시뮬레이션과의 결합 탐색

#### 일반화 측면

7. **도메인 적응 (Domain Adaptation)**: 의료(MRI/CT), 위성, 적외선 이미지 등 특수 도메인에서의 FFC 기반 SR 일반화 연구

8. **OOD(Out-of-Distribution) 강건성 평가**: 현재 논문은 표준 벤치마크만 평가. 새로운 열화 패턴, 미지의 도메인에 대한 체계적인 OOD 평가 프레임워크 구축 필요

9. **대배율 SR (×8, ×16)**: 극단적 업스케일링에서 FFC의 글로벌 정보가 더 중요해질 수 있으나 미검증

#### 실용화 측면

10. **온디바이스 최적화**: Feature Ensemble이 추론 시간을 증가시키지 않는다고 하나, 모바일 환경에서 복수 체크포인트 저장에 따른 메모리 부담 해결 방안 연구 필요

---

## 참고 문헌

**주요 참고 자료:**

1. **Zhang, D., Huang, F., Liu, S., Wang, X., & Jin, Z. (2022/2023).** "SwinFIR: Revisiting the SwinIR with Fast Fourier Convolution and Improved Training for Image Super-Resolution." arXiv:2208.11247v3. *(본 논문, 제공된 PDF)*

2. **Liang, J., et al. (2021).** "SwinIR: Image Restoration Using Swin Transformer." ICCVW 2021.

3. **Chi, L., Jiang, B., & Mu, Y. (2020).** "Fast Fourier Convolution." NeurIPS 2020.

4. **Suvorov, R., et al. (2022).** "Resolution-Robust Large Mask Inpainting with Fourier Convolutions." WACV 2022.

5. **Chen, X., et al. (2022).** "Activating More Pixels in Image Super-Resolution Transformer (HAT)." arXiv:2205.04437.

6. **Li, W., et al. (2021).** "On Efficient Transformer and Image Pre-training for Low-level Vision (EDT)." arXiv:2112.10175.

7. **Chu, X., Chen, L., & Yu, W. (2022).** "NAFSSR: Stereo Image Super-Resolution Using NAFNet." CVPRW 2022.

8. **Liu, Z., et al. (2021).** "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows." ICCV 2021.

9. **Gu, J., & Dong, C. (2021).** "Interpreting Super-Resolution Networks with Local Attribution Maps." CVPR 2021.

10. **Lai, W.-S., et al. (2018).** "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks." IEEE TPAMI. *(Charbonnier Loss 원전)*

---

> **정확도 관련 고지**: 본 답변은 제공된 논문 PDF(arXiv:2208.11247v3)를 1차 출처로 하며, HAT, NAFSSR 등 비교 대상 논문의 세부 수치는 SwinFIR 논문 내 인용 데이터를 기반으로 한다. 2020년 이후 최신 연구 비교에서 SwinFIR 논문에 직접 언급되지 않은 연구(예: Real-ESRGAN, Model Soup 등)는 연구 방향성 제시 수준으로만 기술하였으며, 해당 연구들의 구체적 수치는 포함하지 않았다.
