# Restoration based Generative Models

## 1. 핵심 주장 및 주요 기여

**"Restoration-based Generative Models (RGM)"**은 **Denoising Diffusion Models (DDMs)**을 이미지 복원(Image Restoration) 관점에서 재해석합니다.[1]

**핵심 통찰:**
- DDMs는 MMSE(Minimum Mean Square Error) 기반 이미지 복원 모델로 이해 가능[1]
- IR(Image Restoration) 문헌의 **MAP(Maximum A Posteriori) 추정**을 생성 모델에 적용하여 효율성 극대화[1]
- 가우스 노이징 제한을 벗어나 **임의의 선형 변환 사용 가능**[1]

**주요 기여:**
1. 샘플링 단계를 1000+ → 4-7단계로 감소 (100배 이상 빠른 추론)[1]
2. 다양한 정규화 항(KLD, MMD, DSWD) 지원으로 유연성 극대화[1]
3. 다중 스케일 학습으로 성능 향상 (FID 2.47 CIFAR-10에서 달성)[1]
4. 역문제(초해상화, 컬러화, 노이즈 제거) 직접 해결 가능[1]

***

## 2. 해결하는 문제 및 기술적 배경

### 2.1 핵심 문제

**문제 1: 추론 효율성**
- DDMs는 한 이미지 당 수백~수천 개의 네트워크 평가 필요[1]
- 실시간 응용 불가능[1]

**문제 2: Forward Process 제한**
- DDMs는 가우스 노이징만 지원[1]
- 다른 형태의 변환(다운샘플링 등) 활용 불가[1]

**문제 3: Ill-posedness 처리**
- MMSE는 높은 노이즈에서 여러 해의 평균 구성 → 비현실적 복원[1]
- 다양한 복원 가능성 표현 어려움[1]

### 2.2 이미지 복원의 수학적 기초

**표준 역문제:**
$$y = Ax + \xi, \quad \xi \sim \mathcal{N}(0, \Sigma)$$

**MAP 추정:**
$$\arg\min_x \frac{1}{2}\left\|\!\!\left(\Sigma^\dagger\right)^{1/2}(Ax-y)\right\|\!\!_2^2 + \lambda g(x)$$

여기서 $g(x)$는 정규화 항(사전 지식)으로, MMSE의 모드 붕괴 문제를 해결[1]

***

## 3. 제안 방법: RGM의 수학적 공식

### 3.1 기본 손실 함수

$$\mathbb{E}_{x \sim p_{\text{data}}, y \sim \mathcal{N}(x,\sigma^2I), z \sim \mathcal{N}(0,I)} \left[ \frac{1}{2\sigma^2}\|G_\theta(y,z)-y\|_2^2 + \lambda g_\varphi(G_\theta(y,z)) \right]$$

**세 가지 핵심 요소:**[1]
1. **충실도 항** $(G_\theta - y)$: 손상 이미지로부터 복원
2. **정규화 항** $\lambda g_\varphi$: 학습 가능한 사전 지식
3. **보조 변수** $z$: Ill-posedness 완화

### 3.2 일반화된 전방 프로세스

$$\mathbb{E}_{x, y \sim \mathcal{N}(Ax,\Sigma), z} \left[ \lambda g_\varphi(G_\theta(y,z)) + \frac{1}{2}\left\|\!\!\left(\Sigma^\dagger\right)^{1/2}(A \cdot G_\theta(y,z)-y)\right\|\!\!_2^2 \right]$$

**자유도:** $A$ (변환 행렬), $\Sigma$ (잡음 공분산), $g_\varphi$ (정규화 함수)[1]

### 3.3 학습 가능한 정규화 항

**KLD 기반 (판별기 활용):**
$$g_\varphi(x) = \log(1-D_\varphi(x)) - \log D_\varphi(x)$$

최적 판별자에서: $\mathbb{E}[g_\varphi] \to D_{\text{KL}}(p_\theta(x|y) \| p(x|y)) + H(p_\theta)$[1]

**MMD 기반:**
$$g(X,Y) = \frac{1}{\binom{M}{2}} \left( \sum_{i \neq j} k(x_i,x_j) - 2\sum_{i \neq j} k(x_i,y_j) + \sum_{i \neq j} k(y_i,y_j) \right)$$

**DSWD 기반:** Distributed Sliced Wasserstein Distance로 고차원에서 안정적[1]

### 3.4 다중 스케일 RGM

**블록 평균화 전방 프로세스:**

$$A_k = P_k \quad \text{(2×2 블록 평균화 필터)}$$

**효과:**[1]
- 잠재 변수 차원 감소 (256×256 → 128×128 → 64×64)
- 공간 정보 점진적 추출
- 프로그레시브 생성의 장점 활용

**스케줄 최적화:**
- RGM-SR (Naive): 다운샘플링+잡음 동시 (성능 저조)
- **RGM-SR**: 홀수 단계=다운샘플링, 짝수 단계=노이징 (분리 → 성능 향상)[1]

***

## 4. 모델 구조 및 훈련 알고리즘

### 4.1 생성기 구조

**기반:** UNet 아키텍처 (NCSN++ 영감)[1]

**입력:** 손상 이미지 $y_k$, 시간 단계 $k$, 보조 변수 $z$
**출력:** 복원 이미지 $\hat{x}_k$

### 4.2 훈련 절차

**Algorithm 1 (후방 샘플링):** 이론적으로 정확하지만 조건 필요[1]

**Algorithm 2 (일반화):** 모든 전방 프로세스에 적용 가능, 실제 성능 우수[1]

```
생성기 손실: log D_φ(ŷ_{k-1}) + (1/2λ)||(Σ̃_k†)^{1/2}(Ã_k ŷ_{k-1} - y_k)||_2^2
판별기 손실: log D_φ(y_{k-1}) + log(1-D_φ(ŷ_{k-1}))
```

### 4.3 샘플링 알고리즘

```
y_T ~ N(0, Σ_T)
for k = T-1 to 0:
    z ~ N(0, I)
    x̂_k = G_θ(y_{k+1}, k+1, z)
    y_k ~ N(A_k x̂_k, Σ_k)
return x̂_0
```

**효율성:** 4-7단계로 고품질 이미지 생성[1]

***

## 5. 성능 향상 및 실험 결과

### 5.1 CIFAR-10 생성 성능[1]

| 모델 | FID ↓ | IS ↑ | NFE ↓ |
|------|-------|------|--------|
| **RGM-KLD-SR** | **2.47** | **9.68** | **7** |
| Score SDE (VE) | 2.20 | 9.89 | 2000 |
| DDPM | 3.21 | 9.46 | 1000 |
| LSGM | 2.10 | 9.87 | 147 |

**결론:** 최고 수준의 FID와 IS를 4-7단계로 달성 (285배 빠름)[1]

### 5.2 CelebA-HQ-256 성능[1]

| 모델 | FID ↓ | NFE ↓ |
|------|-------|--------|
| **RGM-KLD-D** | **7.15** | **4** |
| Score SDE (VP) | 7.23 | 4000 |
| UDM | 7.16 | 2000 |

### 5.3 역문제 성능[1]

**초해상화 (×4):** PSNR 21.14, SSIM 0.59
**컬러화:** PSNR 23.78, SSIM 0.93 (DDRM 대비 우수)

### 5.4 아블레이션 연구 (Table 3, CIFAR-10)[1]

| 요소 | Multi-step | Fidelity | Z | FID |
|------|:----------:|:--------:|:-:|--------|
| 기준 | ✓ | ✗ | ✗ | 3.87 |
| 최적 | ✓ | ✓ | ✓ | **3.04** |

**결론:**
- 충실도 항 필수 (10배 성능 향상)[1]
- 보조 변수 $z$ 필수 (ill-posedness 완화)[1]
- 다중 단계 훈련 필수[1]

### 5.5 정규화 파라미터 영향[1]

**최적 범위:** $\frac{d}{10} \leq \lambda \leq d$ (이미지 크기 $d$)
- 작은 $\lambda$: 복원 집중 → 분포 학습 방해
- 큰 $\lambda$: 성능 저하
- 실제 최적값: $\lambda \approx \frac{d}{3}$

***

## 6. 일반화 성능 향상 가능성

### 6.1 일반화 메커니즘

**기하학적 적응 조화 표현 이론 (Kadkhodaie et al., ICLR 2024):**

확산 모델의 일반화는 **네트워크 아키텍처의 귀납적 편향**에서 비롯됨.[2][3]
- CNN의 계층적 구조가 자연 이미지 기하학과 일치
- 이 정렬이 작은 데이터셋에서도 우수한 일반화 가능하게 함

### 6.2 RGM의 일반화 장점

**1. 프레임워크 유연성:**
- 다양한 정규화 항 (KLD, MMD, DSWD) 지원[1]
- 각 작업에 최적화된 구성 가능

**2. 다중 스케일 학습:**
- 공간 피라미드 구조 (프로그레시브 트레이닝)[1]
- 각 스케일에서 특성 점진적 학습

**3. 사전 지식 통합:**
- 손상 프로세스에 맞춘 정규화[1]
- 역문제 해결 시 직접적 활용

### 6.3 일반화 성능 분석

**메모리 vs 일반화 트레이드오프 (Table 7):**[1]
- T=1 (직접): FID 21.2 (실패)
- **T=4 (권장):** FID 3.04 (최적)
- T=8: FID 6.50 (과부족)

**해석:** 최적 단계 수에서 분포 복잡도와 계산 효율의 균형[1]

### 6.4 작업 특화 설계의 효과

**초해상화 특화 (RGM-SR):**
$$\text{손상 프로세스} = \text{블록 평균화} \Rightarrow \text{FID 향상 (2.47 vs 3.04)}$$

- 같은 NFE에서도 나은 성능
- 정규화 항이 고주파 성분 보존 유도[1]

***

## 7. 한계 및 개선 필요 영역

### 7.1 이론적 한계

**부족한 이론적 근거:**[1]
- 수렴 보장 부재
- 일반화 경계 미분석
- 최적성 조건 미제시

### 7.2 실무적 한계

**1. 정규화 파라미터 선택 어려움**
- 최적 $\lambda$ 직관적이지 않음
- 휴리스틱 범위만 제시: $\frac{d}{10} \leq \lambda \leq d$

**2. 전방 프로세스 설계 복잡성**
- 새 작업마다 스케줄 재설계 필요
- 최적 스케줄 찾기 어려움

**3. 계산 복잡도**
- DDMs 대비 매우 빠름 (285배)
- StyleGAN2 (1 NFE) 대비 여전히 느림

### 7.3 PnP 알고리즘 한계

**현재:** Douglas-Rachford Splitting 사용[1]
**부족:** 수렴 보장 부재

***

## 8. 관련 최신 연구 (2020년 이후)

### 8.1 확산 기반 역문제 해결

**Cold Diffusion (Bansal et al., 2023):**[4]
- 가우스 노이즈 없이 임의 변환 사용 가능
- 하지만 MMSE 사용 → 비효율적 샘플링 (152.76 FID on CIFAR-10)
- **RGM 대비:** MAP 기반으로 훨씬 우수 (2.47 FID)

**Soft Diffusion (Daras et al., 2024):**[5]
- 일반 선형 손상 프로세스 지원
- FID 3.86 (≤100 NFE)
- RGM과 유사한 아이디어 독립적 개발

### 8.2 생성 모델 일반화 이론 (2023-2025)

**Key Findings:**[3][2]
- 기하학적 적응 조화 표현이 일반화 기반
- 충분한 훈련 데이터로 우수한 일반화 가능 (~10^5 이미지)
- 네트워크 아키텍처의 귀납적 편향 결정적

**RGM 함의:**
- 다중 스케일 학습이 더 강한 귀납적 편향 제공
- 명시적 정규화 항이 추가 기하학적 구조 인코딩

### 8.3 효율적 확산 모델 (2024-2025)

**Consistency Models (Song et al., 2023):**
- 1단계 생성 가능하지만 품질 저하

**Latent Diffusion 확장:**
- 압축 공간에서 500배 속도 향상 가능[6]

**Flow Matching의 부상:**[7]
- Multi-marginal Flow Matching (Lee et al., 2025)
- Hamiltonian 기반 접근
- RGM의 유연한 프로세스와 상보적

### 8.4 응용 확장 (2024-2025)

**비디오 생성:** Diffusion-based Video Generation 서베이 발표 (2025)[8]
- 시간적 일관성 개선
- 다중 프레임 조화 (RGM 아이디어 유사)

**의료 이미지:** Low-dose CT 복원 (MarCoDiff, 2025)[9]
- 저선량 이미징 잡음 제거
- 구조 충실도 유지

***

## 9. 앞으로의 연구 시 고려할 점

### 9.1 이론적 개선

**1. 수렴 분석**
- Algorithm 1, 2의 수렴 속도 분석
- 비볼록 최적화 이론 적용

**2. 일반화 경계**
- 훈련 오류 vs 테스트 오류 분석
- 네트워크 용량 vs 샘플 복잡도

**3. 최적성 조건**
- MAP 목적 함수의 최적성 특성화
- 정규화 파라미터 최적값 유도

### 9.2 방법론 개선

**1. 자동 정규화 파라미터 선택**
- Cross-validation 기반 자동 선택
- 데이터 특성 분석 후 초기화

**2. 적응형 전방 프로세스 스케줄**
- 메타 러닝으로 최적 스케줄 학습
- 각 작업/데이터셋 특화

**3. 하이브리드 접근**
- RGM + Consistency Models 결합
- 1-2 단계 초고속 + 추가 정제

### 9.3 응용 확장

**1. 비디오 생성**
- 공간-시간 손상 프로세스
- 시간적 연속성 정규화:
$$g_\varphi(\{x_t\}) = \|x_t - x_{t-1}\|_2 + \text{광학흐름 제약}$$

**2. 조건부 생성**
- 텍스트-이미지: 텍스트를 손상 프로세스 포함
- 레이아웃 조건: 공간 마스크 활용

**3. 의료 이미지 재구성**
- 저선량 CT/MRI 복원
- 불완전 측정 재구성

### 9.4 도메인 확장

**Generalization to Unseen Domains:**[10]
- 최근 연구가 확산 모델 잠재 공간의 도메인 일반화 장점 발견
- RGM의 다중 표현 선택지가 추가 이점 제공 가능

***

## 10. 최종 평가

### 10.1 주요 성과

✓ **개념적 혁신**: 생성 모델을 이미지 복원으로 재해석[1]
✓ **계산 효율성**: 100배 이상의 속도 향상 (4-7 vs 1000+ 단계)[1]
✓ **프레임워크 유연성**: 다양한 손상/정규화 지원[1]
✓ **실무 가능성**: 역문제 직접 해결 가능[1]

### 10.2 영향력

**직접적 영향:**
- Cold Diffusion (Bansal et al., 2023) 영감
- DDGAN (Xiao et al., 2021) 보완 이론 제공
- Soft Diffusion 등 후속 연구 영감

**간접적 영향:**
- 확산 모델 일반화 이론 발전
- Plug-and-Play 알고리즘 재조명
- 효율적 생성 모델 연구 촉진

### 10.3 종합 평가

**강점:**
- 획기적 개념적 발전
- 매우 높은 계산 효율성
- 유연한 프레임워크
- 다양한 응용 가능

**약점:**
- 부족한 이론적 기초
- 하이퍼파라미터 선택 어려움
- StyleGAN2 수준의 초고속은 아직

**결론:** RGM은 **생성 모델 역사의 이정표적 기여**입니다. 특히 **계산 효율성에서 혁신적**이며, **이미지 복원과 생성 간의 이론적 다리**를 놓았습니다. 2025년 현재 최신 연구들(Flow Matching, 확산 모델 일반화 이론)이 RGM의 아이디어를 발전시키고 있음은 이 프레임워크의 영향력을 증명합니다.[7][2][9]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3f25eb92-6a93-4e37-98a6-cf0e07b13ca1/2303.05456v2.pdf)
[2](https://www.cns.nyu.edu/pub/lcv/kadkhodaie24a.pdf)
[3](https://arxiv.org/abs/2310.02557)
[4](https://proceedings.neurips.cc/paper_files/paper/2023/file/80fe51a7d8d0c73ff7439c2a2554ed53-Paper-Conference.pdf)
[5](https://arxiv.org/html/2308.09388v2)
[6](http://arxiv.org/pdf/2410.11795.pdf)
[7](https://www.emergentmind.com/topics/flow-score-matching)
[8](https://arxiv.org/abs/2504.16081)
[9](https://www.sciencedirect.com/science/article/abs/pii/S0957417425024340)
[10](http://arxiv.org/pdf/2503.06698.pdf)
[11](https://link.springer.com/10.1007/s10489-025-06673-1)
[12](http://medrxiv.org/lookup/doi/10.1101/2025.08.11.25333418)
[13](https://arxiv.org/abs/2410.19429)
[14](https://arxiv.org/abs/2402.07211)
[15](https://ieeexplore.ieee.org/document/10475490/)
[16](https://edu.pubmedia.id/index.php/ptk/article/view/1603)
[17](https://arxiv.org/abs/2503.04696)
[18](https://journal.stemfellowship.org/doi/10.17975/sfj-2024-004)
[19](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[20](https://aclanthology.org/2023.acl-long.248.pdf)
[21](https://arxiv.org/pdf/2208.14699.pdf)
[22](https://arxiv.org/pdf/2305.18455.pdf)
[23](https://arxiv.org/abs/2408.08306)
[24](http://arxiv.org/pdf/2412.17162.pdf)
[25](http://arxiv.org/pdf/2402.17090.pdf)
[26](https://arxiv.org/html/2410.22637)
[27](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)
[28](https://proceedings.neurips.cc/paper_files/paper/2024/file/25869dbf7682272357bc2cbbf860e1c8-Paper-Conference.pdf)
[29](https://www.sciencedirect.com/science/article/abs/pii/S0925231224011445)
[30](https://papers.nips.cc/paper_files/paper/2023/file/3663ae53ec078860bb0b9c6606e092a0-Paper-Conference.pdf)
[31](https://arxiv.org/abs/2209.00796)
[32](https://www.sciencedirect.com/science/article/abs/pii/S0925231225021897)
[33](https://www.paperdigest.org/report/?id=advances-in-flow-matching-insights-from-icml-2025-papers)
[34](https://diffusion.kaist.ac.kr)
[35](http://pm-research.com/lookup/doi/10.3905/jod.2024.1.212)
[36](https://ieeexplore.ieee.org/document/10716806/)
[37](https://arxiv.org/abs/2403.01633)
[38](https://dl.acm.org/doi/10.1145/3707292.3707367)
[39](https://arxiv.org/abs/2310.02279)
[40](https://www.semanticscholar.org/paper/9e73a3beffc299ccabedc98512b3dc234d2b0350)
[41](https://nbpublish.com/library_read_article.php?id=71827)
[42](http://biorxiv.org/lookup/doi/10.1101/2024.10.15.616846)
[43](https://ieeexplore.ieee.org/document/10389779/)
[44](https://arxiv.org/html/2411.19339v2)
[45](https://arxiv.org/pdf/2311.01797.pdf)
[46](https://arxiv.org/html/2410.02667v1)
[47](https://arxiv.org/abs/2407.00503)
[48](https://openaccess.thecvf.com/content/WACV2025/papers/Gutha_Inverse_Problems_with_Diffusion_Models_A_MAP_Estimation_Perspective_WACV_2025_paper.pdf)
[49](https://simons.berkeley.edu/sites/default/files/2024-09/zahra%20kadkhodaie%20MPG24-1%20Slides.pdf)
[50](https://openaccess.thecvf.com/content/CVPR2024/papers/Rout_Beyond_First-Order_Tweedie_Solving_Inverse_Problems_using_Latent_Diffusion_CVPR_2024_paper.pdf)
[51](https://pmc.ncbi.nlm.nih.gov/articles/PMC8044255/)
[52](https://arxiv.org/abs/2407.20784)
[53](https://openreview.net/forum?id=ANvmVS2Yr0)
