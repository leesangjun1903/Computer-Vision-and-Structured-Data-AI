# RePaint: Inpainting using Denoising Diffusion Probabilistic Models

### **1. 핵심 주장 및 주요 기여**

**RePaint**는 사전학습된 무조건부 DDPM을 활용한 **마스크-무관(mask-agnostic) 이미지 인페인팅** 방법을 제안합니다.[1]

**주요 혁신**:
- 특정 마스크 분포에 대한 훈련 없이 **어떤 형태의 마스크에도 일반화 가능**[1]
- 역확산 과정에서만 조건화하여 원본 DDPM 네트워크 수정 불필요[1]
- **리샘플링 전략**: 확산 시간에서 전진-후진을 반복하여 조화로운 이미지 생성[1]
- 의미론적으로 자연스러운 내용 생성으로 기존 GAN과 차별화[1]

***

### **2. 제안 방법의 수학적 기초**

#### **역확산 조건화 (Equation 8)**

$$x_{t-1}^{\text{known}} = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

$$x_{t-1}^{\text{unknown}} \sim \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

$$x_{t-1} = m \odot x_{t-1}^{\text{known}} + (1-m) \odot x_{t-1}^{\text{unknown}}$$

**핵심**: 알려진 픽셀은 원본 이미지에서, 미지의 픽셀은 DDPM에서 각각 샘플링[1]

#### **리샘플링 전략**

기본 방법의 한계:
- 알려진 픽셀이 생성된 부분과 독립적으로 계산되어 불조화 발생

해결책:
- $$x_{t-1}$$ 생성 후 다시 $$x_t$$로 확산: $$x_t \sim \mathcal{N}(\sqrt{1-\beta_{t-1}} x_{t-1}, \beta_{t-1}I)$$
- 이를 통해 생성 정보가 다음 단계로 전파[1]
- 최적 설정: 점프 길이 $$j=10$$, 리샘플 횟수 $$U=10$$[1]

***

### **3. 성능 향상 및 실험 결과**

#### **정량적 성능 (Table 1)**

**CelebA-HQ**:
- **Wide 마스크**: LPIPS 0.059 (참조), 사용자 투표에서 다른 모든 방법 능가[1]
- **Super-Resolution 2×**: 73% LPIPS 향상 (DSI 대비)[1]
- **Alternating Lines**: 사용자 투표 99.3% (압도적 승리)[1]

**ImageNet**:
- **모든 마스크 유형에서 경쟁력 있는 성능**[1]
- 특히 극단적 마스킹(Expand, Half)에서 의미론적 일관성 우수[1]

#### **리샘플링 효과 분석 (Table 3)**

| 점프 길이(j) | 리샘플(r) | LPIPS | 사용자 투표 |
|-----------|---------|-------|-----------|
| 1 | 10 | 0.088 | 42.5 |
| **10** | **10** | **0.068** | **56.2** |

**결론**: 점프 길이 $$j=1$$에서는 흐린 결과, $$j=10$$에서 최적 성능[1]

#### **"느린 확산" vs "리샘플링" (Table 2)**

동일 계산 예산에서:
- 느린 확산: LPIPS 0.167-0.179
- **리샘플링**: LPIPS 0.134 (약 20% 개선)[1]

***

### **4. 모델 일반화 성능 향상**

#### **4.1 마스크-무관 일반화의 원리**

**훈련**: 마스크 정보 없이 무조건부 DDPM 학습
**추론**: 임의의 이진 마스크에 역확산 과정 적용

수학적 근거:
$$q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$$

마스크 $$m$$이 이 분포에 영향을 주지 않으므로, **어떤 마스크도 동일한 프레임워크 적용 가능**[1]

#### **4.2 다양한 마스크에서의 일반화**

6가지 마스크 유형에서 최소 5개에서 SOTA 달성:[1]
- Wide (넓은 지역 제거)
- Narrow (좁은 영역 마스킹)
- Super-Resolve 2× (픽셀 간격 마스킹)
- Alternating Lines (행 단위 마스킹)
- Half (이미지 절반 제거)
- Expand (중앙 64×64만 유지)

#### **4.3 확률적 다양성 유지**

**Diversity Score (Table 6)**:

| 마스크 | LPIPS | Diversity Score |
|------|-------|-----------------|
| Narrow | **0.034** | **23.79** |
| Alternating Lines | **0.011** | **23.00** |

**결론**: 높은 품질 유지하면서 의미있는 다양성 생성[1]

***

### **5. 한계 및 문제점**

#### **5.1 계산 효율성 부족**

- **GAN/Autoregressive**: 수백 ms 추론 시간
- **RePaint (T=250, U=10)**: 수 초 이상 (실시간 응용 부적합)[1]

#### **5.2 극단적 마스킹에서의 평가 어려움**

- 생성 이미지가 Ground Truth와 **의미론적으로 다를 수 있음**
- LPIPS의 한계: 다양한 의미론적 완성들을 처벌[1]

#### **5.3 데이터 편향**

- ImageNet 모델의 개(dog) 과다 생성
- 특정 도메인 DDPM 재훈련 필요[1]

***

### **6. 2020년 이후 최신 관련 연구**

#### **6.1 확산 모델 기반 인페인팅의 발전 (2021-2024)**

| 연도 | 방법 | 특징 |
|------|------|------|
| 2021 | ILVR | 저주파 정보 기반 조건화 (인페인팅 직접 적용 불가) |
| 2021 | SDEdit | 중간 시간부터 시작 (53% 성능 향상) |
| 2022 | Palette | 조건부 DDPM 훈련 (마스킹 영역 필요) |
| 2023-24 | GuidPaint | 분류기 유도 기반 마스크-무관 (추가 분류기 필요) |
| 2024 | 3D-Consistent | 다중 시점 3D 일관성 보장 |
| 2024 | PrefPaint | 인간 선호도 RL 정렬 (51K 이미지 데이터셋) |
| 2024 | BrushNet | 플러그인 기반 마스크 인식 어댑터 |

#### **6.2 이론적 기초 연구 (2023-2024)**

**"On the Generalization Properties of Diffusion Models" (NeurIPS 2023)**:[2]
- 일반화 갭: $$O(n^{-2/5})$$ (표본 크기에 대해 다항식)
- 차원의 저주 회피: 낮은 내재 차원 분포 학습[2]
- 모드 시프트 효과: 모드 거리에 따른 성능 저하[2]

**"Generalization of Diffusion Models" (2024)**:[3]
- 모델 재현성 발견: 동일 초기화에서 다양한 모델이 유사 출력 생성
- 위상 전이: ~10,000 샘플에서 메모라이제이션→일반화 전환[3]

#### **6.3 효율성 개선 방향**

**잠재 확산 모델 (Latent Diffusion)**:
- 압축 잠재 공간에서 확산 수행
- Stable Diffusion으로 상용화 성공
- RePaint의 계산 병목 해결 가능성

**Semi-Implicit DDPM (SIDDM)**:
- 매우 큰 디노이징 점프 가능
- 샘플링 속도 대폭 향상

#### **6.4 의료 영상 및 특화 분야**

**MaskMedPaint (2024)**:[4]
- 의료 영상의 허위 상관관계 완화
- 목표 도메인으로의 일반화 향상

**주요 응용 분야**:
- 의료 영상 완성
- 문화유산 복원
- 원격 감지 (위성 이미지)
- 3D 재구성 (LiDAR)

***

### **7. 앞으로의 연구에 미치는 영향과 고려사항**

#### **7.1 이론적 기여**

- **생성 모델링의 새로운 패러다임**: GAN/자기회귀와 근본적으로 다른 조건화 방식
- **확산 모델 일반화 이론**: 명시적 상한 도출 필요
- **마스크 분포 강건성**: 이론화 및 정량화

#### **7.2 기술 개선 방향**

**우선순위 1: 효율성 (2025년)**
- 목표: 밀리초 수준 추론
- 방법: 잠재 확산 + 리샘플링

**우선순위 2: 멀티모달 조건화 (2025-2026년)**
- 텍스트 프롬프트 + 참조 이미지
- Classifier-free guidance 활용

**우선순위 3: 이론적 기반 (2024-2027년)**
- 일반화 특성 완전 이해
- 확률 모델 재구성

**우선순위 4: 3D 확장 (2025-2028년)**
- 3D 재구성 일관성
- 동적 씬 완성

#### **7.3 도전 과제**

**윤리 및 사회적 문제**:
- 조작 가능성 및 신원 위장
- 데이터 편향 증폭
- 저작권 문제

**신뢰성**:
- 극단적 인페인팅에서 불안정
- 제어된 다양성 필요

***

### **8. 결론**

**RePaint**는 이미지 인페인팅 분야에서 **패러다임 변화**를 주도했습니다:[1]

| 측면 | 기존 방법 | RePaint |
|------|---------|--------|
| 마스크 학습 | 특정 분포 훈련 | 마스크 무관 |
| 의미 일관성 | 텍스처 복사 | 의미론적 생성 |
| 다양성 | 단일 결과 | 확률적 다양성 |
| 극단 마스킹 | 실패 | 의미있는 완성 |

**현재 상태 (2025년)**:
- 효율성: 개선 중 (5-10배 향상)
- 품질: 최고 수준 유지
- 일반화: 계속 확장 중

**앞으로의 방향**:
- 실시간 인페인팅 실현
- 완전 멀티모달 제어
- 이론-실무 통합
- 윤리적 가이드라인 정립

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/92874f09-4d69-43df-a4cb-7bb105d83e63/2201.09865v4.pdf)
[2](https://proceedings.neurips.cc/paper_files/paper/2023/file/06abed94583030dd50abe6767bd643b1-Paper-Conference.pdf)
[3](https://www.siam.org/publications/siam-news/articles/generalization-of-diffusion-models-principles-theory-and-implications/)
[4](http://arxiv.org/pdf/2411.10686.pdf)
[5](https://arxiv.org/abs/2507.20478)
[6](https://arxiv.org/abs/2403.01633)
[7](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[8](https://ieeexplore.ieee.org/document/11147740/)
[9](https://biss.pensoft.net/article/136839/)
[10](http://pubs.rsna.org/doi/10.1148/radiol.240343)
[11](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[12](http://pubs.rsna.org/doi/10.1148/rycan.240287)
[13](https://aacrjournals.org/cancerres/article/85/8_Supplement_1/1830/755809/Abstract-1830-A-chemoradiation-platform-for)
[14](https://academic.oup.com/bjd/article/doi/10.1093/bjd/ljaf085.208/8161783)
[15](https://arxiv.org/abs/2406.04206)
[16](https://arxiv.org/pdf/2312.14091.pdf)
[17](http://arxiv.org/pdf/2403.16016.pdf)
[18](http://arxiv.org/pdf/2312.03771.pdf)
[19](https://arxiv.org/html/2412.05881v1)
[20](https://arxiv.org/pdf/2308.13767.pdf)
[21](http://arxiv.org/pdf/2408.06429.pdf)
[22](https://arxiv.org/pdf/2210.12113.pdf)
[23](https://arxiv.org/html/2507.21627v1)
[24](https://www.emergentmind.com/topics/conditional-denoising-diffusion-probabilistic-models-ddpms)
[25](https://arxiv.org/pdf/2506.23038.pdf)
[26](https://europe.naverlabs.com/research/publications/3d-consistent-image-inpainting-with-diffusion-models/)
[27](https://www.nature.com/articles/s41598-024-61040-3)
[28](https://www.sciencedirect.com/science/article/abs/pii/S0926580525000378)
[29](https://neurips.cc/virtual/2024/poster/94203)
[30](https://openaccess.thecvf.com/content/WACV2025/papers/Mei_Improving_Conditional_Diffusion_Models_through_Re-Noising_from_Unconditional_Diffusion_Priors_WACV_2025_paper.pdf)
[31](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03014.pdf)
[32](https://github.com/AlonzoLeeeooo/awesome-image-inpainting-studies)
[33](https://arxiv.org/html/2312.15540v1)
[34](https://ph.pollub.pl/index.php/acs/article/download/3550/3154)
[35](https://www.mdpi.com/1424-8220/23/16/7094/pdf?version=1691676114)
[36](https://arxiv.org/pdf/2401.10227.pdf)
[37](https://arxiv.org/html/2412.01223v1)
[38](https://arxiv.org/html/2312.05039v1)
[39](https://www.emergentmind.com/topics/masked-diffusion-models)
[40](https://openreview.net/pdf/125e9e5ad28dca85073a0e1a4b870836957f9de9.pdf)
[41](https://blog.outta.ai/201)
[42](https://arxiv.org/html/2502.03491v3)
[43](https://www.sciencedirect.com/science/article/pii/S1077314224000092)
[44](https://arxiv.org/abs/2305.14712)
[45](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_Training-free_Geometric_Image_Editing_on_Diffusion_Models_ICCV_2025_paper.pdf)
[46](https://arxiv.org/html/2411.10686v1)
