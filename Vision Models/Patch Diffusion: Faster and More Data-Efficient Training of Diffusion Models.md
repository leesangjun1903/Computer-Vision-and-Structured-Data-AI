
# Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models

## 1. 핵심 주장 및 주요 기여 요약

**Patch Diffusion**은 Wang et al. (2023)이 제안한 혁신적인 훈련 프레임워크입니다. 이 방법의 핵심 주장은 **전체 이미지 대신 작은 패치를 사용하여 훈련하면, 훈련 시간을 절반 이상 단축하면서도 생성 품질을 유지하거나 향상시킬 수 있다**는 것입니다.[1]

**주요 기여**:

- **패치 레벨 훈련 프레임워크**: 확산 모델에 처음으로 적용된 일반화 가능한 패치 기반 훈련 기술으로, U-Net 아키텍처에 플러그 앤 플레이 방식으로 적용 가능[1]

- **조건부 점수 함수 혁신**: 패치 위치와 패치 크기를 조건화하는 새로운 전략으로, 전역 구조 의존성을 효과적으로 인코딩[1]

- **실증적 성과**: CelebA-64×64에서 FID 1.77, AFHQv2-Wild-64×64에서 FID 1.93, ImageNet-256×256에서 FID 2.72 달성하며 **2배 이상의 훈련 시간 단축** 달성[1]

- **소규모 데이터셋 성능 향상**: 5,000개 이미지만으로 훈련한 AFHQv2 데이터셋에서 기존 방법보다 우수한 성능 입증[1]

***

## 2. 해결하고자 하는 문제

### 2.1 근본적인 문제점

확산 모델의 훈련은 **세 가지 중대한 과제**에 직면해 있습니다:[1]

1. **막대한 훈련 비용**: DDPM을 8개의 V100 GPU로 훈련하는 데 LSUN-Bedroom 64×64에서 약 4일, 256×256에서는 2주 이상 소요. 최고 성능 모델은 150~1,000 V100 GPU일 필요[1]

2. **데이터 기아(Data Hunger)**: 최고 성능 모델들(DALL-E 2, Stable Diffusion 등)이 OpenImages와 LAION 같은 10억 단위의 대규모 데이터셋을 필수적으로 요구[1]

3. **자원 불평등**: 막대한 계산 자원과 데이터에 접근 불가능한 광범위한 연구 커뮤니티에 이 기술이 제한되어 있는 상황[1]

논문은 이를 "**확산 모델 훈련의 민주화(democratizing diffusion model training)**"라 표현하며, 제한된 자원을 가진 연구자들도 경쟁력 있는 확산 모델을 훈련할 수 있도록 하는 것을 목표로 설정합니다.[1]

### 2.2 기존 해결 시도의 한계

- **추론 최적화**: DDIM, DPM-Solver, EDM-Sampling 등 기존 방법들은 샘플링 속도 개선에만 집중하여 훈련 비용 문제를 해결하지 못함[1]

- **잠재 공간 확산(LDM)**: Latent Diffusion은 잠재 공간에서 훈련하여 어느 정도 비용을 절감했으나, 고해상도 생성에는 여전히 많은 훈련 시간 필요[1]

- **GAN 기반 패치 훈련**: Coco-GAN의 패치 기반 훈련은 판별자가 여전히 전체 이미지를 처리해야 하므로 메모리 절감 효과가 제한적[1]

***

## 3. 제안하는 방법론 상세 설명

### 3.1 패치 기반 점수 매칭 이론

확산 모델의 핵심은 **점수 기반 생성 모델링** 프레임워크를 따릅니다. 역 SDE(Stochastic Differential Equation)는 다음과 같이 표현됩니다:[1]

$$dx = \left( f(x,t) - g^2(t) \nabla_x \log p_{\sigma_t}(x) \right) dt + g(t) dw$$

여기서 $\(\nabla_{x}\log p_{\sigma _{t}}(x)\)$ 는 **점수 함수(score function)**이며, 신경망 $s_θ(x, σ_t)$ 로 근사됩니다.[1]

**전통적 점수 매칭**은 전체 이미지에 대해 다음을 최소화합니다:[1]

$$\mathbb{E}_{x \sim p(x)} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma_t^2 I)} \| D_\theta(x + \epsilon; \sigma_t) - x \|_2^2$$

**Patch Diffusion 접근법**은 작은 패치들에 대해 조건부 점수를 학습합니다:[1]

$$\mathbb{E}_{x \sim p(x), \epsilon \sim \mathcal{N}(0, \sigma_t^2 I), (i,j,s) \sim \mathcal{U}} \| D_\theta(\tilde{x}_{i,j,s}; \sigma_t, i, j, s) - x_{i,j,s} \|_2^2$$

여기서:
- (i, j): 패치의 좌상단 모서리 좌표
- s: 패치 크기 (예: 16, 32, 64 픽셀)
- $x̃_{i,j,s} = x_{i,j,s} + ε$ : 노이즈가 추가된 패치
- U: 균등 분포

조건부 점수 함수는 다음과 같이 정의됩니다:[1]

$$s_\theta(x, \sigma_t, i, j, s) = (D_\theta(x; \sigma_t, i, j, s) - x) / \sigma_t^2$$

### 3.2 패치 크기 스케줄링 전략

전역 구조 의존성을 학습하기 위해 다양한 크기의 패치를 사용합니다. 패치 크기 확률 분포 p_s는 다음과 같이 정의됩니다:[1]

$$s \sim p_s := \begin{cases} p & \text{when } s = R \\ \frac{3}{5}(1-p) & \text{when } s = R//2 \\ \frac{2}{5}(1-p) & \text{when } s = R//4 \end{cases}$$

여기서 R은 원본 이미지 해상도이고, p는 전체 이미지를 사용하는 비율입니다.[1]

**두 가지 스케줄링 방식**:[1]

1. **확률적 스케줄링(Stochastic)**: 각 미니배치에서 위 확률로 패치 크기를 무작위로 샘플링

2. **점진적 스케줄링(Progressive)**: 훈련을 세 단계로 나누어 작은 패치에서 큰 패치로 점진적 전환

실증 결과, 확률적 스케줄링이 점진적 스케줄링보다 우수합니다:[1]
- CelebA-64×64: 확률적(FID 1.66) vs 점진적(FID 2.05)
- FFHQ-64×64: 확률적(FID 3.11) vs 점진적(FID 3.85)

### 3.3 픽셀 좌표 조건화 메커니즘

패치 위치 정보를 인코딩하기 위해 **정규화된 좌표 시스템**을 구성합니다:[1]

- 좌표 범위: [-1, 1] × [-1, 1]
- 좌상단 모서리: (-1, -1)
- 우하단 모서리: (1, 1)

패치의 좌상단 좌표 (i, j)를 **두 개의 추가 채널**로 인코딩하여 원본 이미지 채널과 연결합니다:[1]

$$\text{입력} = [\text{이미지 채널} | \text{X 좌표 채널} | \text{Y 좌표 채널}]$$

훈련 시에는 재구성된 좌표 채널의 손실을 무시하고 이미지 채널의 손실만 계산합니다. 이는 데이터 증강 효과를 제공합니다. 예를 들어 64×64 이미지에서 16×16 패치 크기로는 (64-16+1)² = 2,401개의 서로 다른 위치의 패치를 생성할 수 있습니다.[1]

### 3.4 샘플링 절차

**Patch Diffusion의 핵심 이점**은 샘플링이 원래 확산 모델과 동일하게 간단하다는 점입니다:[1]

1. 원본 이미지 해상도에 대한 전체 좌표 계산
2. 좌표 채널과 샘플 연결
3. 역 확산 과정 실행
4. 각 반복에서 재구성된 좌표 채널 버림

이는 별도의 패치 병합 절차가 필요 없다는 점에서 기존의 패치 기반 GAN 방법과 근본적으로 다릅니다.[1]

***

## 4. 모델 구조 및 아키텍처

### 4.1 U-Net 기반 아키텍처

Patch Diffusion은 **U-Net 기반 아키텍처**의 완벽한 호환성을 유지하도록 설계되었습니다:[1]

**기본 U-Net 구조**:[1]
- 입력 → ResDown-Block1 → ResDown-Block2 → ResDown-Block3
- 중간: Middle-Block (병목 레이어)
- ResUp-Block3 → ResUp-Block2 → ResUp-Block1 → 출력

**Patch Diffusion의 수정 사항**:[1]
- 입력: 원본 채널 + X 좌표 채널 + Y 좌표 채널 (3개 채널 추가)
- 나머지: 완전히 동일한 U-Net 구조
- **완벽한 플러그 앤 플레이**: 기존 UNet 모델을 수정 없이 사용 가능

이는 U-Net의 완전 합성곱(fully convolutional) 특성 때문에 가능합니다. 합성곱 필터는 입력 해상도에 관계없이 작동하므로, 패치 훈련과 전체 이미지 훈련 간 전환이 자유롭습니다.[1]

### 4.2 구현 세부사항

논문은 두 가지 주요 백본 모델을 사용합니다:[1]

**저해상도 모델(64×64)**:
- 백본: EDM-DDPM++
- GPU: 16개 V100
- 배치 크기: 512
- 훈련 기간: 2억 이미지

**고해상도 모델(256×256)**:
- 백본: EDM-ADM + Stable Diffusion 잠재 인코더/디코더 결합
- 명칭: Latent Patch Diffusion Model (LPDM)
- 특징: 잠재 공간에서 패치 훈련 수행

***

## 5. 성능 향상 및 실험 결과

### 5.1 훈련 효율성 개선

| 데이터셋 | 방법 | FID | 훈련 시간 | 개선율 |
|---------|------|-----|---------|--------|
| CelebA-64×64 | EDM-DDPM++ (기준) | 1.66 | ~48h | - |
| CelebA-64×64 | Patch Diffusion | 1.77 | ~24h | **2배** |
| FFHQ-64×64 | EDM-DDPM++ (기준) | 2.60 | ~48h | - |
| FFHQ-64×64 | Patch Diffusion | 3.11 | ~24h | **2배** |
| LSUN-Bedroom-256×256 | LDM-ADM (기준) | 4.32 | ~8일 | - |
| LSUN-Bedroom-256×256 | LPDM (ours) | 2.75 | ~4일 | **2배** |
| ImageNet-256×256 | ADM (기준) | 10.94 | ~7일 | - |
| ImageNet-256×256 | LPDM (ours) | 7.64 | ~3.5일 | **~2배** |

### 5.2 전체 이미지 비율(p)의 영향 분석

ablation study를 통해 훈련 중 전체 이미지를 사용하는 비율 p의 영향을 분석했습니다:[1]

| p 값 | FID | 훈련 시간(시간) |
|------|-----|--------------|
| 0.0 | 14.51 | 13.6 |
| 0.1 | 3.05 | 20.1 |
| 0.25 | 2.10 | 22.5 |
| 0.5 | 1.77 | 24.6 |
| 0.75 | 1.65 | 42.7 |
| 1.0 | 1.66 | 48.5 |

**주요 발견**:[1]
- p=0.0 (패치만 사용): FID 14.51으로 품질 저하
- p=0.1: 극적인 개선으로 FID 3.05 달성 (전역 정보의 중요성 확인)
- p=0.5: **최적 지점** - 훈련 효율성과 생성 품질의 최고 균형점
- p≥0.75: 한계 효과로 인해 훈련 시간만 증가하고 품질 개선 미미

### 5.3 소규모 데이터셋 성능

**AFHQv2 데이터셋 (각 ~5,000개 이미지)**:[1]

| 데이터셋 | 방법 | FID | 훈련 시간 | 개선 |
|---------|------|-----|---------|------|
| AFHQv2-Cat | EDM-DDPM++ | 4.60 | ~18h | - |
| AFHQv2-Cat | Patch Diffusion | 3.11 | ~9h | **32% FID 개선** |
| AFHQv2-Dog | EDM-DDPM++ | 4.94 | ~18h | - |
| AFHQv2-Dog | Patch Diffusion | 4.80 | ~9h | **3% FID 개선** |
| AFHQv2-Wild | EDM-DDPM++ | 2.59 | ~18h | - |
| AFHQv2-Wild | Patch Diffusion | 1.93 | ~9h | **26% FID 개선** |

**중요한 발견**: 소규모 데이터셋에서 패치 기반 훈련은 단순히 훈련 시간을 줄이는 것을 넘어 생성 품질도 개선합니다. 이는 패치 기반 무작위 크롭이 데이터 증강 효과를 제공하여 과적합을 방지하기 때문입니다.[1]

### 5.4 미세조정(Fine-tuning) 성능

ControlNet 미세조정 실험에서 Patch Diffusion을 적용했을 때:[1]
- Stable Diffusion v1-5 체크포인트에서 20k 스텝으로 미세조정
- HED map-to-image 생성 작업
- **결과**: 생성 품질 저하 없이 **약 2배 훈련 시간 단축**

***

## 6. 모델의 일반화 성능 향상 메커니즘

### 6.1 데이터 증강 효과

Patch Diffusion이 일반화 성능을 향상시키는 핵심 메커니즘은 **내재적 데이터 증강**입니다:[1]

**증강 메커니즘**:
- 64×64 이미지에서 16×16 패치: (64-16+1)² = 2,401개의 서로 다른 패치 생성
- 매 에포크마다 무작위 크롭으로 새로운 패치 샘플 생성
- 픽셀 좌표를 추가 정보로 제공하여 위치 정보 다양성 확보

이는 다음을 가능하게 합니다:
- **오버피팅 방지**: 제한된 데이터로도 높은 다양성을 제공
- **공간 불변성 학습**: 모델이 위치에 무관한 특징을 학습
- **계층적 특징 학습**: 다양한 패치 크기로 다중 스케일 특징 획득

### 6.2 전역 일관성 보장

패치 기반 훈련의 핵심 도전은 **전역 일관성**입니다. Patch Diffusion은 이를 다음과 같이 해결합니다:[1]

1. **다양한 크기의 패치 사용**: 작은 패치(R//4)에서 큰 패치(R//2), 전체 이미지(R)로의 계층적 학습으로 점진적으로 전역 맥락 이해

2. **확률적 스케줄링**: 각 미니배치에서 무작위로 패치 크기 선택하여 모델이 모든 스케일의 패치를 동시에 처리하도록 훈련

3. **전체 이미지 가이드**: p 비율의 훈련 반복에서 전체 이미지를 사용하여 글로벌 점수 함수 학습

### 6.3 일반화 성능 향상의 이론적 근거

논문의 Appendix A에서 제시된 이론적 해석에 따르면:[1]

**핵심 통찰**: 조건부 점수 함수 $s_θ(x, σ_t, i, j, s)$ 는 단순히 로컬 패치의 점수를 학습하는 것이 아니라, 좌표 조건을 통해 위치 정보를 인코딩함으로써 전역 점수 함수로의 수렴을 유도합니다.

이는 다음 수학적 성질에 기반합니다:
- 패치 좌표 (i,j)를 명시적으로 입력하면, 모델이 패치 간 관계를 학습
- 다양한 크기의 패치 조합으로 전체 이미지의 점수 함수를 재구성 가능
- 확률적 샘플링으로 서로 다른 패치 위치 조합을 균등하게 커버

***

## 7. 한계 및 문제점

### 7.1 이론적 한계

1. **수렴성 증명 부족**: 논문이 명시적으로 지적하듯이, 패치 기반 점수 매칭의 일반적인 경우에 대한 엄밀한 수렴 증명이 부족[1]

2. **최적 p 값의 원리**: p=0.5가 최적인 이유에 대한 깊이 있는 이론적 설명 부재. 경험적 발견이지만 일반화 가능성의 이론적 보장 없음[1]

3. **위치 인코딩의 한계**: 현재 간단한 픽셀 좌표 사용은 고주파 정보 인코딩 측면에서 제한적. 논문이 향후 연구로 주기함수(periodic positional embedding) 개선 제안[1]

### 7.2 실험적 한계

1. **고해상도 성능**: ImageNet-256×256에서 ADM 기준(FID 10.94, CFG 미적용)에 비해 LPDM(FID 7.64)이 개선을 보이지만, 이는 Latent Diffusion 기반이므로 순수 픽셀 공간 비교와는 다름[1]

2. **경계 아티팩트**: 패치 경계에서 불연속성이 발생할 수 있으나, 논문에서는 명시적으로 다루지 않음. 동시 연구인 Ding et al. (2023)이 feature collage로 이 문제를 해결[1]

3. **초고해상도 한계**: 512×512 이상 해상도에서의 성능이 명시적으로 평가되지 않음[1]

### 7.3 적용 제약

1. **텍스트-이미지 모델**: DALL-E 2, Stable Diffusion 같은 텍스트 조건 모델에의 직접 적용 평가 부재[1]

2. **비이미지 도메인**: 시간 계열, 3D 데이터, 음성 등에의 확장 가능성이 명확하지 않음

3. **분산 훈련**: 다중 GPU 간 패치 분배 전략에 대한 상세한 논의 부재

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 훈련 효율성 개선 관련 연구

**Latent Diffusion Models (Rombach et al., 2021)**[2]
- **접근법**: 이미지를 VAE로 인코딩하여 잠재 공간에서 확산 훈련
- **성과**: 훈련 비용 및 메모리 대폭 감소
- **Patch Diffusion과의 비교**: LDM은 아키텍처 수준의 변경이 필요한 반면, Patch Diffusion은 모든 U-Net 기반 모델에 플러그 앤 플레이 적용 가능. 또한 Patch Diffusion은 고품질 인코더 부재 시에도 사용 가능[1]

**EDM (Elucidating the Design Space of Diffusion-Based Generative Models, Karras et al., 2022)**[2]
- **접근법**: 노이즈 스케줄, 샘플러, 아키텍처 설계 최적화
- **성과**: 더 효율적인 훈련 및 샘플링
- **관계**: Patch Diffusion의 백본으로 사용되어 상호 보완적 성과 달성[1]

**Scaling Properties of Latent Diffusion (2024)**[3]
- **발견**: 모델 크기와 데이터 크기의 상호작용이 성능에 미치는 영향 분석
- **Patch Diffusion과의 차이**: 매개변수 스케일링에 초점을 맞춘 반면, Patch Diffusion은 훈련 방식의 구조적 개선에 집중[3]

### 8.2 데이터 효율성 개선 관련 연구

**DreamBooth (Ruiz et al., 2022)**[1]
- **접근법**: 4~5개의 특정 객체 이미지로 사전훈련 모델 미세조정
- **성과**: 적은 데이터로 신원 보존 생성 가능
- **Patch Diffusion과의 비교**: DreamBooth는 미세조정에 초점이고, Patch Diffusion은 처음부터 훈련하는 경우에 최적. 또한 Patch Diffusion의 5,000장 성과는 처음부터 훈련한 결과로 더 일반적[1]

**One-shot Diffusion (Zhang et al., 2023)**[1]
- **접근법**: 단일 이미지로 사전훈련 모델 미세조정
- **한계**: 동일 이미지의 변형만 생성 가능 (본질적으로 다른 목표)

**Limited Data Diffusion (LD-Diffusion, 2024)**[4]
- **접근법**: 압축 모델과 혼합 증강(MAFP) 전략 조합
- **성과**: 소규모 데이터에서 상태최고(SOTA) 성능
- **Patch Diffusion과의 비교**: 두 방법 모두 데이터 증강을 통한 효율성 개선이지만, LD-Diffusion은 추가 압축 모듈이 필요한 반면 Patch Diffusion은 더 간단한 구조[4]

### 8.3 패치 기반 생성 모델 관련 연구

**Patched Denoising Diffusion Models (Ding et al., 2023)**[1]
- **접근법**: 동시 개발된 대안으로, feature collage 기법으로 경계 아티팩트 해결
- **특징**: 고해상도 이미지 합성에 특화
- **Patch Diffusion과의 차이**: 윈도우 슬라이딩 방식으로 공간 일관성 강화, Patch Diffusion의 경계 문제 개선[1]

**Patch-based Position-aware Diffusion (PaDIS, 2024)**[5][6]
- **접근법**: Patch Diffusion과 유사한 패치 기반 훈련 + 위치 인코딩
- **응용**: 의료 이미지(CT 재구성, MRI) 및 역문제 해결
- **발견**: 제한된 데이터에서 전체 이미지 사전 훈련 방법보다 우수한 성능[6][5]

**Hierarchical Patch Diffusion Models (2024)**[7]
- **응용**: 고해상도 비디오 합성
- **특징**: 패치 계층적 구조로 비디오 시간 일관성 보장
- **확장성**: Patch Diffusion 개념을 비디오 도메인으로 확장[7]

### 8.4 조건부 확산 모델 관련 연구

**Classifier-Free Guidance (Ho & Salimans, 2021)**[1]
- **개념**: 별도 분류기 없이 조건 정보를 통합
- **Patch Diffusion과의 통합**: 논문에서 ImageNet 조건부 생성에 CFG 적용하여 성능 향상 달성[1]

**Geometrically-Conditioned Point Diffusion (GECCO, 2023)**[8]
- **응용**: 3D 점 구름 생성
- **방식**: 기하학적 조건을 명시적으로 인코딩
- **관련성**: 좌표 기반 조건화 개념의 3D 확장[8]

### 8.5 효율성 중심 확산 모델 서베이

**Efficient Diffusion Models Survey (2024)**[9][10]
- **범위**: 아키텍처, 훈련, 추론, 배포 전 단계의 효율성 개선 기술 종합 분석
- **Patch Diffusion의 위치**: 훈련 효율성 개선 기법의 대표 사례로 수록
- **카테고리**: "훈련 방식 개선(Training Paradigm Improvement)" 분류 아래 패치 기반 훈련 강조[10][9]

**Efficient Diffusion Models for Vision Survey (2022)**[11]
- **초점**: 컴퓨터 비전 특화 효율성 설계
- **포함 기술**: 아키텍처 최적화, 샘플링 가속화, 경량화
- **평가**: Patch Diffusion이 훈련 단계의 혁신으로 인정[11]

***

## 9. 향후 연구 방향 및 고려사항

### 9.1 이론적 개선

1. **수렴성 증명**: 일반적인 경우에 대한 패치 기반 점수 매칭의 수렴성 엄밀한 증명 필요[1]

2. **최적 패치 크기 선택**: 데이터셋 특성에 따른 최적 패치 크기 선택 이론 개발

3. **위치 임베딩 개선**: 논문 저자들이 언급한 **주기함수 위치 임베딩(periodic positional embedding)** 적용으로 고주파 정보 인코딩 강화[1]

### 9.2 기술적 확장

1. **경계 아티팩트 해결**: Ding et al. (2023)의 feature collage 기법과 Patch Diffusion의 통합으로 고품질 경계 생성

2. **초고해상도 확장**: 512×512 이상 해상도에 최적화된 패치 크기 스케줄링 연구

3. **멀티스케일 아키텍처**: 계층적 U-Net 구조와 패치 훈련의 결합으로 더 효율적인 표현 학습

### 9.3 응용 확대

1. **조건부 생성 강화**: 텍스트-이미지, 이미지-이미지 조건부 모델에서 패치 기반 훈련의 효과 체계적 평가

2. **도메인 확장**:
   - **의료 영상**: PaDIS 연구처럼 CT, MRI 등에 특화[5][6]
   - **비디오 생성**: Hierarchical Patch Diffusion 확장[7]
   - **3D 생성**: 점 구름, 메시, 복셀 데이터
   - **시간 계열**: 신계열 예측, 이상 탐지

3. **분산 훈련 최적화**: 다중 GPU/노드 간 패치 분배 전략으로 매우 대규모 모델 훈련

### 9.4 실무 고려사항

1. **프로덕션 배포**: Patch Diffusion의 편의성과 효율성을 활용한 엣지 디바이스 기반 확산 모델 배포 전략

2. **미세조정 표준화**: ControlNet 미세조정 성공 사례처럼 대규모 사전훈련 모델 미세조정의 표준 접근법으로 확립[1]

3. **데이터셋 선택**: 패치 기반 훈련이 특히 효과적인 데이터셋 특성 규정 (해상도, 복잡도, 다양성)

### 9.5 관련 기술과의 시너지

1. **Flow Matching과의 통합**: 최근 부각되는 Flow Matching 프레임워크와 패치 기반 훈련의 결합[12]

2. **Diffusion Transformer와의 조합**: Vision Transformer 기반 확산 모델에 패치 훈련 적용[13]

3. **다중 모달 학습**: CLIP과 같은 비전-언어 모델과 패치 확산의 통합 학습

***

## 10. 결론 및 영향 평가

### 10.1 논문의 핵심 영향

Patch Diffusion은 **확산 모델의 접근성 혁신**이라는 점에서 획기적인 의의를 가집니다:[1]

1. **기술적 혁신**: 간단하면서도 효과적인 방법으로 2배 이상의 훈련 시간 단축과 소규모 데이터 성능 개선을 동시 달성

2. **민주화 기여**: 제한된 자원의 연구자들도 경쟁력 있는 확산 모델을 훈련할 수 있는 경로 제공

3. **일반화 가능성**: U-Net 기반 모든 확산 모델에 플러그 앤 플레이 방식 적용 가능으로 즉시 영향력 발휘

### 10.2 학계 및 산업 영향

**인용 현황** (2023년 발표 이후): 387회 이상 인용으로 빠른 학술 수용[14]

**후속 연구 촉발**: 
- PaDIS의 역문제 해결 응용[6][5]
- Hierarchical Patch Diffusion의 비디오 확장[7]
- 효율성 서베이에 핵심 기술로 수록[9][10]

**실무 적용**: 
- Stable Diffusion 기반 모델들의 미세조정 표준화
- 엣지 기기 배포 가능성 증가

### 10.3 학문적 기여의 한계

1. **이론 부족**: 수렴성 증명 등 깊이 있는 이론적 분석 필요[1]

2. **경계 처리**: 패치 경계의 아티팩트 문제가 동시 연구에 의해 해결되었으나 원 논문에서 부재

3. **초고해상도 미평가**: 1024×1024 이상의 성능이 명확하지 않음

### 10.4 향후 연구자를 위한 권고사항

1. **즉시 활용**: 기존 확산 모델의 훈련 시간 단축 필요 시 적극 도입 권장

2. **개선 영역**:
   - 주기함수 위치 임베딩 도입으로 성능 향상 기대[1]
   - Feature collage 통합으로 경계 품질 개선[1]
   - 도메인별 최적 패치 크기 스케줄 연구

3. **새로운 응용**:
   - 의료 영상, 비디오, 3D 생성 등 미개척 영역 탐색
   - 분산 훈련 프레임워크 개발로 극대규모 모델 가능성 탐색

***

## 참고 문헌

 Wang, Z., Jiang, Y., Zheng, H., Wang, P., He, P., Wang, Z., Chen, W., & Zhou, M. (2023). Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models. NeurIPS 2023.[1]

 Skorokhodov, A., et al. (2024). Hierarchical Patch Diffusion Models for High-Resolution Video Generation.[7]

 Scaling Properties of Latent Diffusion Models. (2024).[3]

 Comparative Profiling: Insights Into Latent Diffusion Model Training. (2024).[2]

 Efficient Diffusion Models: A Comprehensive Survey from Principles to Practices. (2024).[9]

 Efficient Diffusion Models: A Survey. (2025).[10]

 Efficient Diffusion Models for Vision: A Survey. (2022).[11]

 Learning Image Priors through Patch-based Diffusion Models for Solving Inverse Problems. (2024).[5]

 Towards data efficient generative models. (2024).[4]

 Learning Image Priors Through Patch-Based Diffusion Models for Solving Inverse Problems. OpenReview.[6]

 Diffusion Models and Representation Learning: A Survey. (2024).[12]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3722b1bf-688b-4e03-b49a-776595c63f34/2304.12526v2.pdf)
[2](https://dl.acm.org/doi/10.1145/3642970.3655847)
[3](https://arxiv.org/abs/2404.01367)
[4](https://pure.qub.ac.uk/en/studentTheses/towards-data-efficient-generative-models/)
[5](https://neurips.cc/virtual/2024/poster/95843)
[6](https://openreview.net/forum?id=HGnxhHz6ss)
[7](https://openaccess.thecvf.com/content/CVPR2024/papers/Skorokhodov_Hierarchical_Patch_Diffusion_Models_for_High-Resolution_Video_Generation_CVPR_2024_paper.pdf)
[8](https://openaccess.thecvf.com/content/ICCV2023/papers/Tyszkiewicz_GECCO_Geometrically-Conditioned_Point_Diffusion_Models_ICCV_2023_paper.pdf)
[9](http://arxiv.org/pdf/2410.11795.pdf)
[10](https://arxiv.org/html/2502.06805v1)
[11](https://arxiv.org/pdf/2210.09292.pdf)
[12](https://arxiv.org/pdf/2407.00783.pdf)
[13](https://arxiv.org/html/2507.01467v2)
[14](https://arxiv.org/abs/2304.12526)
[15](https://smj.org.sa/lookup/doi/10.15537/smj.2024.45.7.20240032)
[16](https://goodwoodpub.com/index.php/JoMABS/article/view/1910)
[17](https://dergipark.org.tr/en/doi/10.30622/tarr.1526734)
[18](https://www.journals.spu.ac.ke/index.php/amjr/article/view/260)
[19](https://jurnal.umt.ac.id/index.php/IJOEE/article/view/10538)
[20](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[21](http://medrxiv.org/lookup/doi/10.1101/2024.07.22.24310801)
[22](https://journals.sagepub.com/doi/10.1177/02537176241229197)
[23](http://journal.universitaspahlawan.ac.id/index.php/prepotif/article/view/28053)
[24](https://www.mdpi.com/2079-9292/13/3/585)
[25](https://arxiv.org/pdf/2209.00796v8.pdf)
[26](https://arxiv.org/html/2311.01223v2)
[27](https://arxiv.org/pdf/2305.00624.pdf)
[28](http://arxiv.org/pdf/2209.04747v2.pdf)
[29](http://arxiv.org/pdf/2303.07909.pdf)
[30](https://yonsei.elsevierpure.com/en/publications/diffusion-models-a-comprehensive-survey-of-methods-and-applicatio)
[31](https://s-space.snu.ac.kr/handle/10371/209605)
[32](https://openaccess.thecvf.com/content/ICCV2025/papers/Shao_Memory-Efficient_Generative_Models_via_Product_Quantization_ICCV_2025_paper.pdf)
[33](https://www.sciencedirect.com/science/article/pii/S0031320325005941)
[34](https://www.imsi.institute/videos/patch-based-diffusion-models-for-solving-inverse-problem/)
[35](https://aclanthology.org/D19-1048/)
[36](https://arxiv.org/abs/2209.00796)
[37](https://arxiv.org/html/2508.10875v1)
[38](https://arxiv.org/html/2304.12526)
[39](https://arxiv.org/abs/2512.08854)
[40](https://arxiv.org/abs/2504.16081)
[41](https://arxiv.org/abs/2412.08781)
[42](https://arxiv.org/html/2406.02462v3)
[43](https://arxiv.org/abs/2411.15584)
[44](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1606247/full)
