# Polynomial Implicit Neural Representations For Large Diverse Datasets

---

## 1. 핵심 주장과 주요 기여 요약

이 논문은 기존 Implicit Neural Representation(INR)이 사인파(sinusoidal) 기반 위치 인코딩(positional encoding)에 의존하여 **대규모·다양한 데이터셋으로의 확장이 제한**되는 문제를 지적하고, 이를 **다항 함수(polynomial function)** 기반의 INR, 즉 **Poly-INR**로 대체할 것을 제안한다.

### 주요 기여:
1. **위치 인코딩 없이** 이미지를 좌표 위치의 다항 함수로 표현하는 Poly-INR 모델 제안
2. ImageNet에서 StyleGAN-XL 대비 **3~4배 적은 파라미터**(46M vs. 134~168M)로 비견할 만한 성능 달성
3. FFHQ 데이터셋에서 기존 INR 기반 GAN(CIPS, INR-GAN)을 **크게 능가**
4. 보간(interpolation), 스타일 믹싱, 외삽(extrapolation), 고해상도 샘플링, 이미지 인버전 등 다양한 응용 시연

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 INR 모델은 두 가지 근본적 한계를 가진다:

**첫째**, 사인파 위치 인코딩의 **유한한 임베딩 크기**가 표현력을 제한하여 소규모·단일 이미지 표현에는 적합하나, ImageNet과 같은 대규모·다양한 데이터셋에는 부적합하다.

**둘째**, ReLU 기반 MLP는 **고차 도함수 정보를 포착하지 못한다**. ReLU의 구간별 선형(piecewise linear) 특성 때문에 2차 이상의 도함수가 0이 되어, 테일러 급수 관점에서 고차 다항식을 근사하는 능력이 부족하다. 이는 곧 **고주파 정보 생성 실패**로 이어진다.

### 2.2 제안하는 방법 (수식 포함)

#### 다항 함수로서의 이미지 표현

이미지를 좌표 위치 $(x, y)$의 다항 함수로 모델링한다:

$$G(x,y) = g_{00} + g_{10}x + g_{01}y + \cdots + g_{pq}x^{p}y^{q} $$

여기서 $(x, y)$는 크기 $H \times W$의 좌표 그리드에서 샘플링한 정규화된 픽셀 위치이며, 다항식의 계수 $g_{pq}$는 알려진 분포에서 샘플링한 잠재 벡터 $z$에 의해 매개변수화되고 픽셀 위치와 독립적이다.

#### 이미지 생성

주어진 고정된 $z$에 대해 모든 픽셀 위치에서 $G$를 평가하여 이미지를 형성한다:

$$I = \{G(x,y;z) \mid (x,y) \in \text{CoordinateGrid}(H,W)\} $$

여기서 

```math
\text{CoordinateGrid}(H,W) = \left\{\left(\frac{x}{W-1}, \frac{y}{H-1}\right) \mid 0 \leq x < W, 0 \leq y < H\right\}
```

이다.

#### 합성 네트워크의 수학적 표현

핵심 메커니즘은 각 레벨에서 **아핀 변환된 좌표와 특징 간의 요소별 곱셈(element-wise multiplication)**을 통해 다항식 차수를 점진적으로 증가시키는 것이다:

$$G_{syn} = \cdots \sigma\left(W_2\left((A_2 X) \odot \sigma\left(W_1\left((A_1 X) \odot \sigma(W_0(A_0 X))\right)\right)\right)\right) $$

여기서:
- $X \in \mathbb{R}^{3 \times HW}$: 바이어스 차원을 포함한 좌표 그리드
- $A_i \in \mathbb{R}^{n \times 3}$: 매핑 네트워크로부터의 레벨- $i$ 아핀 변환 행렬
- $W_i \in \mathbb{R}^{n \times n}$: 레벨- $i$의 선형 레이어 가중치
- $\sigma$: Leaky-ReLU 활성화 함수 (negative slope = 0.2)
- $\odot$: 요소별 곱셈

### 2.3 모델 구조

Poly-INR은 두 개의 서브네트워크로 구성된다 (Figure 2 참조):

#### (1) 매핑 네트워크 (Mapping Network)
- 입력: 잠재 코드 $z \in \mathbb{R}^{64}$ + 사전학습된 클래스 임베딩(512차원)
- 출력: 아핀 파라미터 공간 $\mathbf{W} \in \mathbb{R}^{512}$
- 구조: 2층 MLP → 추가 선형 레이어들이 각 레벨의 아핀 파라미터 $(a_j, b_j, c_j)$ 생성

#### (2) 합성 네트워크 (Synthesis Network)
- **10개 레벨**로 구성
- **Level-0**: 좌표 그리드를 아핀 변환 → Linear → Leaky-ReLU
- **Level 1~9**: 이전 레벨의 특징과 아핀 변환된 좌표 그리드 간 **요소별 곱셈** → Linear → Leaky-ReLU
- 마지막 레벨 이후 Linear 레이어로 RGB 출력
- 채널 차원: ImageNet은 $n=1024$, FFHQ는 $n=512$

**핵심 설계 원리**: 각 레벨에서의 요소별 곱셈을 통해 네트워크가 $x$ 또는 $y$ 좌표의 차수를 증가시킬지, 아핀 변환 계수 $a_j = b_j = 0$으로 설정하여 차수를 유지할지를 **학습**한다.

#### StyleGAN과의 관계
StyleGAN은 Poly-INR의 **특수한 경우**로 해석 가능하다. 아핀 변환 행렬에서 $a_j = b_j = 0$이면 바이어스 항 $c_j$가 스타일 코드 역할을 한다. 그러나 Poly-INR은 스타일 코드에 **위치 바이어스(location bias)**를 추가하여, 특정 이미지 영역에만 스타일 코드를 적용하는 유연성을 제공한다. 또한 Poly-INR은 weight modulation/demodulation, low-pass 필터, 합성곱 레이어, 공간 노이즈 주입을 사용하지 않는다.

### 2.4 성능 향상

#### ImageNet 결과 (Table 1)

| 해상도 | 모델 | FID ↓ | IS ↑ | 파라미터(M) |
|--------|------|-------|------|------------|
| $128^2$ | StyleGAN-XL | 1.81 | 200.55 | 158.7 |
| $128^2$ | **Poly-INR** | **2.08** | **179.64** | **46.0** |
| $256^2$ | StyleGAN-XL | 2.30 | 265.12 | 166.3 |
| $256^2$ | **Poly-INR** | **2.86** | **241.43** | **46.0** |
| $512^2$ | StyleGAN-XL | 2.41 | 267.75 | 168.4 |
| $512^2$ | **Poly-INR** | **3.81** | **267.44** | **46.0** |

- BigGAN, ADM 등을 모든 해상도에서 능가
- StyleGAN-XL에 비견할 만한 성능을 **3~4배 적은 파라미터**로 달성
- 모든 해상도에서 **동일한 46M 파라미터** 사용 (해상도 독립적)

#### FFHQ 결과 (Table 2)

| 모델 | 파라미터(M) | FID ↓ |
|------|------------|-------|
| CIPS | 45.9 | 4.38 |
| INR-GAN | 72.4 | 4.95 |
| StyleGAN2 | 30.0 | 3.83 |
| **Poly-INR** | **13.6** | **2.72** |

- 기존 INR 기반 GAN들을 **크게 능가**하면서도 파라미터 수가 현저히 적음
- StyleGAN2도 능가하며, StyleGAN-XL(FID 2.19)에 근접

#### 추가 성능 지표
- **고해상도 샘플링**: 32×32에서 학습 후 512×512로 업샘플링 시 FID 65.15 (Bicubic 73.86 대비 우수)
- **이미지 인버전**: PSNR 26.52, SSIM 0.76 (StyleGAN-XL의 PSNR 13.5, SSIM 0.33 대비 크게 우수)
- **1024×1024 업샘플링 (FFHQ-256 학습)**: Poly-INR FID 13.69 vs. INR-GAN 18.51 vs. CIPS 29.59

### 2.5 한계

1. **고해상도에서의 높은 계산 비용**: INR은 각 픽셀을 독립적으로 생성하므로 모든 계산이 출력 해상도에서 수행됨. CNN의 멀티스케일 파이프라인 대비 추론 속도가 느림 (512² 기준 0.720 sec/img)
2. **고해상도에서 FID 하락 가속**: 해상도 증가에도 레이어를 추가하지 않아 StyleGAN-XL 대비 성능 격차가 커짐
3. **Recall 값 상대적 저하**: 고해상도에서 다양성(recall) 지표가 StyleGAN-XL, 확산 모델 대비 낮음 (작은 모델 크기로 인한 용량 제한)
4. **일반적 GAN 아티팩트**: 다중 머리/팔다리, 누락된 팔다리, 잘못된 기하학적 합성 등
5. **아핀 파라미터 공간의 높은 차원**: 인버전 시 임베딩 공간이 커 학습 분포로부터 벗어난 점이 생길 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 구조적 일반화 이점

#### (a) 해상도 독립적 아키텍처
Poly-INR의 가장 강력한 일반화 특성은 **해상도에 무관한 고정 파라미터 수**이다. 좌표 기반 연속 함수로 이미지를 표현하므로, 학습 해상도와 다른 해상도에서도 이미지를 생성할 수 있다. Table 3에서 보듯이, 32×32에서 학습한 모델이 512×512에서 Bicubic 보간보다 우수한 FID(65.15 vs. 73.86)를 달성한다. 이는 모델이 **연속적인 이미지 함수의 일반적 구조**를 학습했음을 시사한다.

#### (b) 다항식 차수의 적응적 학습
각 레벨에서 아핀 변환 계수 $a_j, b_j$를 0으로 설정하거나 비영(non-zero)으로 설정하여 다항식 차수를 증가시킬지를 네트워크가 **자율적으로 학습**한다. 이는 고정된 위치 인코딩 크기에 의존하는 기존 방법 대비, 데이터의 복잡도에 맞게 **표현력을 적응적으로 조절**할 수 있는 메커니즘이다.

#### (c) 계층적 표현 학습
Figure 3의 히트맵 시각화는 Poly-INR이 일반화 가능한 계층적 표현을 학습함을 보여준다:
- **초기 레벨 (0-3)**: 객체의 기본 구조/형태 (저차 다항식)
- **중간 레벨 (4-6)**: 객체의 전체 형상 (중차 다항식)
- **후기 레벨 (7-9)**: 세밀한 디테일 (고차 다항식)

이러한 계층적 분해는 스타일 믹싱, 보간 등에서의 **의미적으로 일관된 일반화**를 가능하게 한다.

### 3.2 외삽(Extrapolation) 능력

Figure 4에서 보듯이, $[0,1]^2$ 좌표 그리드에서 학습된 모델이 $[-0.25, 1.25]^2$로 외삽할 때 **연속적이고 기하학적으로 일관된 이미지**를 생성한다. 이는 다항 함수의 연속성에 기반한 자연스러운 일반화이며, 학습 영역 외부로의 확장 가능성을 보여준다.

### 3.3 OOD(Out-of-Distribution) 이미지에 대한 일반화

Figure 7에서 분포 외(OOD) 이미지와의 보간에서도 **매끄러운 전환**을 보여주며, 아핀 파라미터 공간의 구조적 연속성을 입증한다.

### 3.4 일반화 성능 향상을 위한 잠재적 방향

1. **모델 용량 확장**: Table 4의 ablation에서 레벨 수와 특징 차원 증가 시 일관된 성능 향상 확인 (Lvl-2 FID 27.01 → Lvl-14 feat.1024 FID 1.12)
2. **고해상도 학습 데이터**: 현재 ImageNet의 원본 해상도 한계로 512×512에서의 성능이 제한적이며, 고해상도 데이터셋이 확보되면 성능 개선 기대
3. **Pivotal Tuning Inversion** 등 최신 기법 적용으로 인버전 후 보간 품질 개선 가능
4. **Weight modulation/demodulation**, 노이즈 주입 등 기존 GAN 기법 통합으로 추가 성능 향상 가능
5. **3D 데이터셋**으로의 확장을 통한 모달리티 간 일반화

### 3.5 Recall 지표 개선 가능성

Table 4에서 모델 용량이 매우 작을 때 recall이 0.01로 극히 낮지만, 파라미터 증가에 따라 0.63까지 향상됨을 확인할 수 있다. 이는 **더 큰 모델**이나 **더 깊은 네트워크**를 통해 다양성(일반화)이 개선될 수 있음을 시사한다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (a) INR 기반 생성 모델의 확장성 입증
이 논문은 INR 기반 생성 모델이 **대규모 다양한 데이터셋**(ImageNet 1K 클래스, 1.2M 이미지)에서도 경쟁력 있는 성능을 달성할 수 있음을 최초로 보여주었다. 이는 INR의 적용 범위를 단일 장면/이미지 표현에서 **범용 생성 모델링**으로 확장하는 전환점이 된다.

#### (b) 합성곱 없는 생성 모델의 가능성
Linear+ReLU만으로 SOTA에 근접한 이미지 생성이 가능함을 보여줌으로써, **합성곱에 의존하지 않는 새로운 생성 모델 패러다임**의 가능성을 열었다. 이는 합성곱의 공간적 귀납 편향(inductive bias) 없이도 고품질 이미지를 생성할 수 있음을 의미한다.

#### (c) 파라미터 효율성의 새로운 기준
46M 파라미터로 168M+ 파라미터 모델에 비견하는 성능을 달성함으로써, **경량 생성 모델** 설계에 대한 새로운 방향을 제시한다.

#### (d) 연속 함수 표현의 실용적 가치
외삽, 임의 해상도 생성, 부드러운 보간 등 연속 함수 표현의 장점이 실제로 대규모 생성 모델에서도 유효함을 입증했다.

### 4.2 향후 연구 시 고려할 점

#### (a) 계산 효율성 개선
INR의 픽셀 독립 생성 방식은 고해상도에서 계산 비용이 급증한다 ($512^2$에서 0.720 sec/img). 향후 연구에서는:
- **계층적/멀티스케일 INR 아키텍처** 설계
- **병렬 처리 최적화** 기법
- **선택적 해상도 렌더링** (coarse-to-fine)

등을 통한 추론 속도 개선이 필요하다.

#### (b) 3D 및 다중 모달리티 확장
저자들이 언급했듯이, Poly-INR을 **3D-aware 이미지 합성**으로 확장하는 것이 중요한 미래 연구 방향이다. 좌표 기반 표현의 특성상 $(x, y)$를 $(x, y, z)$로 자연스럽게 확장 가능하며, NeRF와의 결합이 유망하다.

#### (c) 판별기(Discriminator) 설계
논문에서 지적된 GAN 아티팩트(다중 머리, 누락 팔다리 등)는 **CNN 기반 판별기가 객체의 부분만을 판별**하고 전체 형상을 고려하지 못하기 때문이다. 전역적 구조를 인식하는 판별기 설계가 필요하다.

#### (d) 확산 모델과의 통합
현재 GAN 프레임워크에서의 결과이므로, **확산 모델(diffusion model)** 프레임워크와 Poly-INR의 결합이 흥미로운 연구 방향이다. 다항 함수 기반 연속 표현이 확산 과정의 점진적 노이즈 제거와 시너지를 낼 가능성이 있다.

#### (e) 이론적 분석 강화
다항식 차수와 표현력 간의 관계, 최적 레벨 수 결정, 다항 함수 공간에서의 수렴 보장 등에 대한 **이론적 기반**이 보강될 필요가 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 INR 기반 생성 모델

| 연구 | 년도 | 핵심 방법 | Poly-INR과의 차이 |
|------|------|----------|-----------------|
| **CIPS** (Anokhin et al.) | 2021 | 푸리에 특징 + 학습 가능 위치 벡터 + StyleGAN 가중치 변조 | Poly-INR은 위치 인코딩 없이 다항 함수로 대체, FFHQ FID 4.38 → 2.72로 개선 |
| **INR-GAN** (Skorokhodov et al.) | 2021 | 하이퍼네트워크가 MLP 파라미터 결정, 멀티스케일 구조 | Poly-INR은 하이퍼네트워크 없이 아핀 변환으로 조건화, 파라미터 72.4M → 13.6M |
| **SIREN** (Sitzmann et al.) | 2020 | 주기적 활성화 함수 ($\sin$) 사용 | Poly-INR은 ReLU 유지하면서 요소별 곱셈으로 고주파 포착, GAN에서 SIREN보다 우수 |

### 5.2 CNN 기반 생성 모델

| 연구 | 년도 | 핵심 방법 | Poly-INR과의 비교 |
|------|------|----------|-----------------|
| **StyleGAN-XL** (Sauer et al.) | 2022 | StyleGAN을 대규모 데이터셋으로 확장, projected discriminator 사용 | Poly-INR이 동일한 discriminator 사용, 3-4x 적은 파라미터로 비견할 만한 성능 |
| **StyleGAN-3** (Karras et al.) | 2021 | 좌표 기반 푸리에 특징 + 등변(equivariant) 필터 | 회전 등변 버전이 ImageNet 확장에 실패한 반면, Poly-INR은 성공 |
| **BigGAN** (Brock et al.) | 2018/2019 | 대규모 배치 학습 + 조건부 생성 | Poly-INR이 모든 해상도에서 FID 기준 능가 |

### 5.3 확산 모델 (Diffusion Models)

| 연구 | 년도 | 핵심 방법 | Poly-INR과의 비교 |
|------|------|----------|-----------------|
| **ADM/ADM-G** (Dhariwal & Nichol) | 2021 | U-Net 기반 확산 모델 + 분류기 가이던스 | Poly-INR이 ADM을 능가, ADM-G와 비교 시 해상도별 차이 |
| **DiT-XL/2-G** (Peebles & Xie) | 2022 | 트랜스포머 기반 확산 모델 | DiT-XL이 FID에서 우수하나 675M 파라미터 사용 (Poly-INR의 14.7배) |
| **CDM** (Ho et al.) | 2022 | 캐스케이드 확산 모델 | 128×128에서 FID 3.52 vs. Poly-INR 2.08으로 Poly-INR이 우수 |

### 5.4 좌표 기반 하이브리드 모델

| 연구 | 년도 | 핵심 방법 | Poly-INR과의 차이 |
|------|------|----------|-----------------|
| **LIIF** (Chen et al.) | 2021 | CNN 백본 + 좌표별 로컬 암시적 함수 | CNN 의존적, Poly-INR은 순수 MLP |
| **Arbitrary-scale synthesis** (Ntavelis et al.) | 2022 | 멀티스케일 합성곱 + 스케일 인식 위치 임베딩 | 합성곱 기반, Poly-INR은 합성곱 미사용 |

### 5.5 비교 분석의 핵심 통찰

1. **파라미터 효율성**: Poly-INR은 현존 모든 경쟁 모델 대비 가장 적은 파라미터로 경쟁력 있는 성능을 달성한다는 점에서 독보적이다.

2. **아키텍처 단순성**: 합성곱, 정규화, 셀프어텐션 없이 Linear+ReLU만으로 구성되는 것은 이론적 분석과 실용적 배포 모두에서 이점을 제공한다.

3. **확장성 한계**: 확산 모델(DiT-XL)이나 StyleGAN-XL 대비 절대적 성능에서는 아직 격차가 있으며, 특히 고해상도와 다양성(recall)에서 개선이 필요하다.

4. **연속 표현의 고유 장점**: 외삽, 임의 해상도 생성, 부드러운 보간 등은 CNN/확산 모델이 제공하지 못하는 Poly-INR만의 차별점이다.

---

## 참고자료

1. **원논문**: Rajhans Singh, Ankita Shukla, Pavan Turaga, "Polynomial Implicit Neural Representations For Large Diverse Datasets," arXiv:2303.11424v1 [cs.CV], March 2023.
2. **CIPS**: Anokhin et al., "Image generators with conditionally-independent pixel synthesis," CVPR 2021.
3. **INR-GAN**: Skorokhodov et al., "Adversarial generation of continuous images," CVPR 2021.
4. **SIREN**: Sitzmann et al., "Implicit neural representations with periodic activation functions," NeurIPS 2020.
5. **StyleGAN-XL**: Sauer et al., "StyleGAN-XL: Scaling StyleGAN to large diverse datasets," SIGGRAPH 2022.
6. **StyleGAN-3**: Karras et al., "Alias-free generative adversarial networks," NeurIPS 2021.
7. **ADM**: Dhariwal and Nichol, "Diffusion models beat GANs on image synthesis," NeurIPS 2021.
8. **DiT**: Peebles and Xie, "Scalable diffusion models with transformers," arXiv:2212.09748, 2022.
9. **NeRF**: Mildenhall et al., "NeRF: Representing scenes as neural radiance fields for view synthesis," Communications of the ACM, 2021.
10. **Fourier Features**: Tancik et al., "Fourier features let networks learn high frequency functions in low dimensional domains," NeurIPS 2020.
11. **LIIF**: Chen et al., "Learning continuous image representation with local implicit image function," CVPR 2021.
12. **BigGAN**: Brock et al., "Large scale GAN training for high fidelity natural image synthesis," ICLR 2019.
13. **CDM**: Ho et al., "Cascaded diffusion models for high fidelity image generation," JMLR, 2022.
14. **StyleGAN2**: Karras et al., "Analyzing and improving the image quality of StyleGAN," CVPR 2020.
15. GitHub 저장소: https://github.com/Rajhans0/Poly_INR
