# LDMVFI: Video Frame Interpolation with Latent Diffusion Models

### 1. 핵심 주장과 주요 기여

**LDMVFI (Latent Diffusion Model-based Video Frame Interpolation)**는 비디오 프레임 보간 문제를 **생성적 관점**에서 접근한 첫 번째 시도이다. 기존 VFI 방법들이 L1/L2 손실함수 최적화에 의존한 반면, 이 논문은 지각적 품질 중심의 새로운 패러다임을 제시한다.[1]

**핵심 기여:**
- 잠재 확산 모델을 VFI 문제에 처음으로 적용
- VFI 특화 자동인코딩 모델 **VQ-FIGAN** 개발
- 정량적 평가 및 사용자 연구를 통한 지각적 품질 우수성 입증[1]

***

### 2. 해결하고자 하는 문제

**근본적 문제:** 기존 VFI 방법들은 **객관적 지표와 주관적 품질의 불일치** 문제를 안고 있다.[1]

기존 방법의 한계:
- PSNR이 높아도 시각적으로 만족스럽지 못한 결과 생성[1]
- L1/L2 거리가 인간의 지각과 잘 맞지 않음
- 복잡한 움직임과 동적 텍스처(예: 물, 불, 나뭇잎) 처리 부족[1]
- VGG 특징 기반 손실도 지각 품질을 충분히 반영하지 못함

이러한 문제는 VFI 결과물이 **흐릿함(blur)**과 **부자연스러운 움직임(flickering)**을 초래한다.[1]

***

### 3. 제안하는 방법

#### 3.1 전체 아키텍처[1]

LDMVFI는 두 가지 주요 구성요소로 이루어진다:

1. **VQ-FIGAN**: 이미지를 컴팩트 잠재 공간으로 투영하는 VFI 특화 자동인코더
2. **Denoising U-Net**: 잠재 공간에서 조건부 역확산을 수행하는 신경망

#### 3.2 수학적 공식화

**기본 확산 과정:**[1]

$$\mathcal{L}_{DM} = E_{x_0, \epsilon \sim N(0,I), t \sim U(1,T)} \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2 \quad (1)$$

여기서 $x_t$는 정방향 확산 과정에서 샘플링되고, $\epsilon_\theta$는 시간 조건부 신경망이다.[1]

**잠재 공간 확산 손실:**

$$\mathcal{L}_{LDM} = E_{E(x_0), \epsilon \sim N(0,I), t \sim U(1,T)} \left\| \epsilon - \epsilon_\theta(z_t, t) \right\|^2 \quad (2)$$

여기서 $z = E(x)$는 인코더를 통한 잠재 인코딩이다.[1]

**조건부 생성을 위한 VFI 손실:**

$$\mathcal{L} = E_{z^n, z^0, z^1, \epsilon \sim N(0,I), t \sim U(1,T)} \left\| \epsilon - \epsilon_\theta(z_t^n, t, z^0, z^1) \right\|^2 \quad (6)$$

여기서 $z^0, z^1$은 입력 프레임의 잠재 인코딩이다.[1]

#### 3.3 VQ-FIGAN 아키텍처[1]

**핵심 혁신 - 세 가지 개선사항:**

1. **프레임 보조 디코더**: 이웃 프레임의 특징 피라미드 $\phi_0, \phi_1$을 MaxCA (MaxViT Cross-Attention) 블록을 통해 디코더에 통합

2. **효율적 주의 메커니즘**: 전체 자가-주의(O(n²) 복잡도) 대신 MaxViT 블록을 사용하여 선형 복잡도 달성[1]

3. **변형 가능 컨볼루션 기반 합성**: 직접적인 이미지 출력 대신 적응형 변형 가능 컨볼루션 커널 예측[1]

**프레임 합성 과정:**

$$I_\tau^n(h, w) = \sum_{i=1}^{K} \sum_{j=1}^{K} \Omega_\tau^{h,w}(i, j) \cdot P_\tau^{h,w}(i, j) \quad (3)$$

$$P_\tau^{h,w}(i, j) = I_\tau(h + \alpha_\tau^{h,w}(i, j), w + \beta_\tau^{h,w}(i, j)) \quad (4)$$

$$\hat{I}_n = v \cdot I_0^n + (1-v) \cdot I_1^n + \delta \quad (5)$$

여기서:
- $\Omega$: 커널 가중치 ($^{H \times W \times K \times K}$)[1]
- $\alpha, \beta$: 공간 오프셋
- $v$: 가시성 맵 (오클루전 처리)
- $\delta$: 잔차 맵

***

### 4. 성능 향상 및 한계

#### 4.1 정량적 성능[1]

| 데이터셋 | LPIPS ↓ | FloLPIPS ↓ | FID ↓ | 주요 특성 |
|---------|---------|-----------|-------|---------|
| Middlebury | 0.019 | 0.044 | 16.167 | 저해상도, 작은 움직임 |
| UCF-101 | 0.026 | 0.035 | 26.301 | 다양한 액션 |
| DAVIS | 0.107 | 0.153 | 12.554 | 큰 움직임 |
| VFITex | 0.150 | 0.207 | 32.316 | **동적 텍스처 (~20% 향상)** |
| SNU-FILM-Extreme | 0.123 | 0.204 | 47.042 | 극한 조건 |

**주요 성과:**
- VFITex에서 약 **20% 성능 향상**: 동적 텍스처(물, 불, 나뭇잎 등)에서 기존 최고 성능 방법 대비 현저히 우수[1]
- 모든 테스트 셋에서 LPIPS, FloLPIPS, FID 메트릭에서 상위 성능 달성[1]

#### 4.2 사용자 연구[1]

BVI-HFR 데이터셋을 이용한 주관적 평가:
- LDMVFI vs BMBC: **p < 3.8 × 10⁻⁹** (통계적 유의성)
- LDMVFI vs IFRNet: **p = 1.6 × 10⁻³**
- LDMVFI vs ST-MFNet: **p = 1.7 × 10⁻²**
- 모든 비교에서 95% 신뢰도 수준에서 유의미한 우월성 입증[1]

#### 4.3 한계[1]

**계산 효율:**
- 평균 추론 시간: **8.48초/480p 프레임** (기존 방법: 0.01~0.27초)
- 파라미터 수: **439M** (기존: 5~42M)
- 이는 확산 모델의 반복적 역확산 과정의 고유한 한계[1]

**성능 제한:**
- 극단적 큰 움직임에서 만족스럽지 못한 결과[1]
- 일관된 움직임(rigid object)은 잘 처리하지만, 불규칙한 동적 텍스처는 샘플링 결과 변동성 발생 가능[1]

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 현재 일반화 능력[1]

**교차-데이터셋 성능:**
- 다양한 해상도 (225×225 ~ 4K)와 움직임 복잡도의 5개 벤치마크에서 일관된 우수 성능
- 특히 **보이지 않은 패턴의 동적 텍스처**에서 강력한 일반화 능력 입증[1]

**이유:**
1. **생성적 모델의 특성**: 단일 L1 손실 학습의 "평균화" 문제를 피함[1]
2. **VQ-FIGAN의 이웃 정보 활용**: 이웃 프레임의 특징을 활용하여 문맥 의존적 보간[1]
3. **지각적 손실 함수**: LPIPS 기반 훈련으로 인간의 지각과 일치[1]

#### 5.2 향상 가능성[1]

**아키텍처 개선:**
- **모델 경량화**: 지식 증류(Knowledge Distillation) 및 모델 압축 기법 적용 가능[1]
- **효율적 샘플링**: DPM-Solver, Progressive Distillation 등 확산 가속 기법으로 추론 시간 단축 가능[1]

**도메인 적응:**
- 테스트 타임 적응(Test-Time Adaptation): 목표 비디오의 모션 특성에 따른 동적 조정
- 이를 통해 극단적 큰 움직임이나 특수 도메인(의료, 위성 영상) 처리 가능[1]

**데이터 다양성:**
- Vimeo90K + BVI-DVC 결합 훈련으로 움직임 다양성 증가[1]
- 더 많은 도메인의 비디오 포함으로 일반화 강화 가능

**멀티-모달 가이딩:**
- 텍스트, 깊이, 광류 등 추가 지도 신호 활용으로 조건부 생성 개선[1]

***

### 6. 2020년 이후 관련 최신 연구 비교 분석

#### 6.1 VFI 패러다임의 진화[2]

| 방법 분류 | 특징 | 주요 논문 | 한계 |
|----------|------|---------|------|
| **광류 기반** | 광학적 움직임 추정으로 명시적 모션 모델링 | IFRNet(2022), VFIformer(2022) | 움직임 추정 오류에 민감 |
| **커널 기반** | 적응형 변형 가능 컨볼루션 사용 | CDFI(2021), ST-MFNet(2022) | 고해상도에서 계산 비용 증가 |
| **Transformer 기반** | 자가-주의 메커니즘으로 장거리 의존성 포착 | VFIformer(2022), VFIT(2023) | 파라미터 수 및 계산량 증가 |
| **생성 모델 기반** | GAN 또는 확산 모델로 고품질 생성 | LDMVFI(2023), MoG(2025), MiVID(2024) | 추론 속도 느림, 샘플링 불확실성 |

#### 6.2 확산 모델 기반 VFI의 발전[3][4][5]

**LDMVFI (2023)**: 첫 번째 LDM 기반 VFI[1]
- **장점**: 지각적 품질 우수, 특히 동적 텍스처 처리 탁월
- **단점**: 추론 속도, 극단적 큰 움직임

**MoG (Motion-Aware Generative, 2025)**:[6]
- 광류 기반 안정성과 생성적 유연성 결합
- 복잡한 움직임 영역에서 동적 아티팩트 보정
- 실제 및 애니메이션 벤치마크 모두에서 우수 성능

**MiVID (2024)**: 자기-지도 학습 기반 확산 VFI[7]
- 라벨이 없는 데이터로도 학습 가능
- 도메인 적응 향상 가능성

#### 6.3 평가 메트릭의 진화[8]

**전통적 메트릭의 한계:**
- **PSNR/SSIM**: 저수준 픽셀 거리 기반, 지각 품질 반영 부족[9]

**현대적 지각 메트릭:**
- **LPIPS**: 심층 신경망 특징 기반, 인간 판단과 높은 상관성[8]
- **FloLPIPS**: VFI 특화 메트릭, 시간 정보 고려[1]
- **FID**: 분포 수준의 유사성 측정[1]

LDMVFI가 FID를 지각 메트릭으로 도입한 것은 **분포 일치** 관점에서 생성 모델의 우월성을 입증하는 중요한 기여이다.[1]

#### 6.4 도메인 일반화 연구의 트렌드[10][11][12]

**비전 트랜스포머 기반 도메인 적응**: Vision Transformer가 도메인 변화에 대한 강건성을 보임[12][10]
- CLIP 같은 대규모 사전훈련 모델의 적응 가능성[11]

**LDMVFI와의 연결:**
- MaxViT 블록 도입으로 효율적 주의 달성[1]
- 사전훈련된 LDM 활용으로 도메인 이동에 대한 강건성 향상 가능[1]

**테스트 타임 적응:**
최근 연구에서 테스트 타임에 경량 어댑터를 통해 모션 추정 개선 가능성 제시[13]
- LDMVFI에도 적용 가능한 기법

***

### 7. 향후 연구에 미치는 영향 및 고려사항

#### 7.1 패러다임 전환의 의미[4][5][1]

**이전 패러다임 (결정적 예측):**
- 단일 "가장 그럴듯한" 중간 프레임 생성
- L1/L2 손실로 인한 과도한 평균화
- 불확실한 움직임에서 흐릿한 결과

**LDMVFI의 새로운 패러다임 (확률적 생성):**
- 다양한 가능한 프레임 생성 가능
- 불확실성을 명시적으로 모델링[6]
- 동적 텍스처에서 더 선명한 결과[1]

이는 VFI를 "보간"에서 **"조건부 생성"으로 재정의**하는 근본적 전환이다.[2][1]

#### 7.2 후속 연구 방향

**1. 효율성 개선 (필수)**[1]
- Progressive Distillation, DDIM 최적화, 계층적 생성으로 수초에서 밀리초 수준으로 가속화
- 모바일/엣지 디바이스 배포 가능성[1]

**2. 극단적 조건 처리**
- 거대한 움직임: 다단계 보간 또는 광류 초기화 활용
- 폐색(Occlusion): 깊이 정보 또는 인페인팅 기법 통합[1]

**3. 다중 모달 확장**[5][4]
- 텍스트 가이딩: "부드럽게", "선명하게" 같은 지시 반영
- 깊이/세그멘테이션 가이딩: 구조적 일관성 강화
- 광류 프라이어: 기하학적 제약 추가[1]

**4. 적응형 생성**[6]
- 장면의 움직임 복잡도에 따른 샘플링 스텝 동적 조정
- 배경/전경 분리 후 차등 처리

**5. 시간 일관성 보장**[14][4]
- 장시간 비디오에서 시간적 깜박임 방지
- 양방향 샘플링 및 궤적 융합(Trajectory Fusion)[2]

#### 7.3 응용 분야 확대

**현재 응용:**
- 슬로우모션 생성 (영화, 스포츠 방송)
- 프레임 레이트 업컨버전 (30fps → 60fps)

**확대 가능 응용:**
1. **비디오 압축**: 키프레임만 전송, 중간 프레임은 수신단에서 생성[1]
2. **비디오 편집**: 사용자가 원하는 움직임 스타일 지정
3. **의료 영상**: 저 프레임율 촬영 영상의 시간 해상도 개선
4. **원격 감지**: 위성 영상의 시간 보간[1]
5. **인터랙티브 비디오**: 실시간 프레임 레이트 조정 (게임, VR)

#### 7.4 사회적 고려사항[1]

**긍정적 영향:**
- 저대역폭 환경에서 고품질 비디오 전송 가능
- 의료/과학 응용에서 정확한 영상 재구성

**부정적 우려:**
1. **딥페이크 위험**: 생성 모델의 악용 가능성
2. **에너지 소비**: 높은 계산량으로 인한 탄소 발자국[1]
3. **데이터 프라이버시**: 대규모 비디오 데이터 학습 필요[1]

대응책:
- 법제화된 생성물 추적 시스템 개발
- 에너지 효율적 알고리즘 연구
- 합성 데이터 활용으로 프라이버시 보호

***

### 8. 종합 평가 및 시사점

#### 8.1 LDMVFI의 혁신성[1]

✓ **기술적 혁신**: 첫 LDM 기반 VFI로 패러다임 전환
✓ **경험적 우수성**: 특히 동적 텍스처에서 20% 성능 향상
✓ **이론적 기초**: 확률적 생성의 수학적 엄밀성[1]
✓ **포괄적 평가**: 정량/정성 평가, 사용자 연구 포함[1]

#### 8.2 현존 문제와 해결책

| 문제 | 심각도 | 해결 방안 |
|-----|--------|---------|
| 추론 속도 (8.48초) | 높음 | 확산 가속화 기법, 증류학습[1] |
| 모델 크기 (439M) | 중간 | 경량화, 동적 네트워크[1] |
| 극단적 움직임 | 중간 | 광류 초기화, 다단계 보간[1] |
| 일관성 (장시간) | 중간 | 양방향 샘플링[2] |

#### 8.3 학계 및 산업에 미칠 영향

**단기 (1-2년):**
- 지각적 VFI 품질 개선의 새로운 기준 제시
- 확산 모델 적용 분야 확대

**중기 (3-5년):**
- 효율적 확산 모델 개발로 실용화 가속
- 멀티-모달 조건부 생성으로 복합 작업 지원

**장기 (5년 이상):**
- VFI가 표준 비디오 처리 파이프라인의 필수 구성요소화
- 신경 비디오 코덱으로의 통합

***

### 결론

**LDMVFI**는 비디오 프레임 보간을 **생성적 관점**에서 재정의한 획기적 연구이다. 잠재 확산 모델의 고품질 합성 능력과 VFI 특화 설계(VQ-FIGAN)를 결합하여, 특히 **동적 텍스처와 복잡한 움직임**에서 기존 방법을 현저히 능가한다.[1]

주요 제한점인 **계산 효율**은 확산 모델 가속화 기술과 모델 경량화를 통해 단기간에 극복 가능하며, 시간 일관성 개선, 도메인 적응, 다중 모달 확장은 활발한 후속 연구 분야이다.[4][6][2][1]

**향후 확산 모델 기반 영상 처리** 연구의 중심이 될 이 분야는 실용성과 이론성의 균형을 이루며 발전할 것으로 예상된다.

***

### 참고문헌 인덱스

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/24468ea8-d35c-40fb-88a8-8fd5c94c09b6/2303.09508v3.pdf)
[2](https://arxiv.org/html/2506.01061v1)
[3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10606505/)
[4](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
[5](https://arxiv.org/abs/2504.16081)
[6](https://arxiv.org/html/2501.03699v1)
[7](https://arxiv.org/html/2511.06019v1)
[8](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750231.pdf)
[9](https://dl.acm.org/doi/10.1145/3556544)
[10](https://link.springer.com/10.1007/s00521-024-10353-5)
[11](https://arxiv.org/abs/2407.15173)
[12](https://arxiv.org/abs/2404.04452)
[13](https://papers.bmvc2023.org/0179.pdf)
[14](https://arxiv.org/html/2412.18688v2)
[15](https://link.springer.com/10.1007/978-3-030-66823-5_2)
[16](https://www.ijsrp.org/research-paper-0925.php?rp=P16513947)
[17](https://journals.sagepub.com/doi/10.1177/00027642251395845)
[18](https://www.semanticscholar.org/paper/b813e1e4b94caab640307f5cb04d4ee40048c00b)
[19](http://biorxiv.org/lookup/doi/10.1101/2025.10.16.682789)
[20](https://www.tandfonline.com/doi/full/10.1080/02770903.2025.2488000)
[21](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11766/2590958/Multi-frame-super-resolution-for-versatile-video-coding/10.1117/12.2590958.full)
[22](https://www.semanticscholar.org/paper/414836b8de0bfd9291f3c1f9647d7db4e3a39340)
[23](https://www.ijcmph.com/index.php/ijcmph/article/view/8512)
[24](http://arxiv.org/pdf/2404.11108.pdf)
[25](https://arxiv.org/pdf/2105.07673.pdf)
[26](https://arxiv.org/pdf/2206.08572.pdf)
[27](http://arxiv.org/pdf/2211.11309.pdf)
[28](https://arxiv.org/pdf/1708.01692.pdf)
[29](https://arxiv.org/pdf/2402.02892.pdf)
[30](https://arxiv.org/pdf/2106.07286.pdf)
[31](https://learnopencv.com/video-generation-models/)
[32](https://research.nvidia.com/labs/toronto-ai/VideoLDM/)
[33](https://github.com/CMLab-Korea/Awesome-Video-Frame-Interpolation)
[34](https://openaccess.thecvf.com/content/CVPR2023/papers/Blattmann_Align_Your_Latents_High-Resolution_Video_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.pdf)
[35](https://www.emergentmind.com/topics/latent-video-diffusion-model)
[36](https://arxiv.org/abs/2506.01061)
[37](https://arxiv.org/html/2210.00379v7)
[38](https://arxiv.org/abs/2401.12945)
[39](https://www.arxiv.org/pdf/2509.12024v1.pdf)
[40](https://arxiv.org/html/2210.00379v6)
[41](https://arxiv.org/html/2510.00855v1)
[42](https://arxiv.org/html/2505.12705v1)
[43](https://arxiv.org/html/2510.05976v1)
[44](https://arxiv.org/pdf/2504.06328.pdf)
[45](https://github.com/showlab/Awesome-Video-Diffusion)
[46](https://arxiv.org/abs/2304.08818)
[47](https://ieeexplore.ieee.org/document/10675013/)
[48](https://dl.acm.org/doi/10.1145/3654664)
[49](https://ieeexplore.ieee.org/document/10477388/)
[50](https://ieeexplore.ieee.org/document/10660496/)
[51](https://ieeexplore.ieee.org/document/10378205/)
[52](https://ieeexplore.ieee.org/document/10030340/)
[53](http://ieeexplore.ieee.org/document/8099799/)
[54](https://arxiv.org/pdf/2404.04452.pdf)
[55](http://arxiv.org/pdf/2302.12047.pdf)
[56](https://arxiv.org/html/2412.04077v1)
[57](http://arxiv.org/pdf/2007.03511.pdf)
[58](https://arxiv.org/pdf/2012.11807.pdf)
[59](https://arxiv.org/html/2501.18592)
[60](https://pmc.ncbi.nlm.nih.gov/articles/PMC6759585/)
[61](https://arxiv.org/pdf/2106.11344.pdf)
[62](https://openaccess.thecvf.com/content/CVPR2021/papers/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.pdf)
[63](https://www.youtube.com/watch?v=9KpZA-tibrU)
[64](https://arxiv.org/html/2504.05402v2)
[65](https://www.techrxiv.org/users/782254/articles/1267153/master/file/data/main/main.pdf?inline=true)
[66](https://studios.disneyresearch.com/2023/06/04/frame-interpolation-transformer-and-uncertainty-guidance/)
[67](https://arxiv.org/pdf/2503.22375.pdf)
[68](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ldmvfi/)
[69](https://arxiv.org/html/2406.11138v1)
[70](https://openaccess.thecvf.com/content/CVPR2023/papers/Plack_Frame_Interpolation_Transformer_and_Uncertainty_Guidance_CVPR_2023_paper.pdf)
[71](https://arxiv.org/html/2507.16406v1)
[72](https://arxiv.org/html/2504.05402v1)
[73](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_IQ-VFI_Implicit_Quadratic_Motion_Estimation_for_Video_Frame_Interpolation_CVPR_2024_paper.pdf)
[74](https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_Video_Frame_Interpolation_With_Transformer_CVPR_2022_paper.pdf)
[75](https://arxiv.org/html/2509.23402v1)
[76](https://arxiv.org/html/2505.08235v1)
[77](https://wenli-vision.github.io/papers/LRELSSVM_Li_TPAMI2017_preprint.pdf)
[78](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1222815/full)
[79](https://arxiv.org/html/2406.11371)
