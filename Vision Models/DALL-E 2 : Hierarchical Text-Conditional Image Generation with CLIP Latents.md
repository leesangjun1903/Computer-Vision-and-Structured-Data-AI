# unCLIP/DALL-E 2 : Hierarchical Text-Conditional Image Generation with CLIP Latents

### 1. 핵심 주장 및 기여

**논문의 핵심 주장**[1]

본 논문은 CLIP과 같은 대조 모델(contrastive model)이 학습한 견고한 이미지 표현을 활용하여 텍스트 조건부 이미지 생성을 수행할 수 있다는 주요 주장을 제시합니다. 기존의 직접적인 텍스트-투-이미지 생성과 달리, **CLIP 잠재 공간을 매개로** 함으로써 의미론적 다양성과 광학적 사실성을 동시에 달성할 수 있음을 보여줍니다.[1]

**주요 기여**[1]

1. **계층적 생성 구조**: 텍스트 캡션 → CLIP 이미지 임베딩(Prior) → 최종 이미지(Decoder)의 두 단계 구조로, 각 단계의 역할을 명확히 분리
2. **확산 사전(Diffusion Prior)의 효율성**: 자동회귀(AR) 사전 대비 계산량은 줄이면서도 더 높은 품질 달성
3. **이미지 조작 능력**: 인코딩과 디코딩, 보간, 텍스트 기반 변환 등 다양한 영상처리 기능을 CLIP 공간에서 구현
4. **CLIP 잠재 공간 분석**: PCA와 역함수 시각화를 통해 CLIP이 인식하는 정보 구조 규명

***

### 2. 해결하는 문제, 제안 방법 및 모델 구조

**문제 정의**[1]

기존 텍스트-투-이미지 생성 모델(GLIDE, DALL-E)은 다음의 트레이드오프를 겪고 있었습니다:
- **지도 기법(Guidance) 적용 시** 이미지 품질은 향상되지만 의미론적 다양성이 급격히 감소
- **속성 바인딩 문제**: 복수 객체의 속성을 정확하게 연결하기 어려움
- **계산 비용**: 직접 텍스트 조건부 생성은 높은 계산량 필요

**제안 방법**[1]

두 단계의 계층적 생성 모델로 구성:

$$P(x|y) = P(x|z_i, y)P(z_i|y)$$

여기서:
- $x$: 생성 이미지
- $y$: 텍스트 캡션
- $z_i$: CLIP 이미지 임베딩
- $z_t$: CLIP 텍스트 임베딩

**단계 1: Prior 모델** $P(z_i|y)$[1]

두 가지 구현 방식을 비교:

**(1) 자동회귀(AR) Prior**[1]
- CLIP 이미지 임베딩 $z_i$ (1024차원)을 PCA로 차원 축소 (319차원)
- 각 차원을 1,024개 이산 버킷으로 양자화
- Transformer 기반 인과 마스킹(causal masking)으로 순차 예측

텍스트 임베딩 $z_t$와의 내적 $z_i \cdot z_t$ 기반 샘플링으로 품질 제어

**(2) 확산 Prior**[1]
- 연속 벡터 $z_i$를 직접 모델링하는 가우시안 확산 모델
- 손실 함수:

$$L_{\text{prior}} = \mathbb{E}_{t\sim[1,T], z_i^{(t)} \sim q_t} \left\| f_\theta(z_i^{(t)}, t, y) - z_i \right\|_2$$

여기서 $f_\theta$는 뮤토르 역잡음 예측기, $q_t$는 노이즈 프로세스

- 비 $\epsilon$ 예측 대신 직접 임베딩 값 예측 (평균제곱오차 손실)

**단계 2: Decoder 모델** $P(x|z_i, y)$[1]

확산 모델 기반으로 CLIP 임베딩을 조건으로 사용:

1. **조건 통합 방식**:
   - CLIP 임베딩을 timestep 임베딩에 프로젝션 및 덧셈
   - GLIDE 텍스트 인코더 출력에 4개의 추가 토큰으로 CLIP 임베딩 연결

2. **분류기 없는 지도(Classifier-Free Guidance)**[1]
   - 학습 시 CLIP 임베딩 10%, 텍스트 캡션 50% 확률로 제거
   - 추론 시 지도 척도로 생성 제어

3. **고해상도 생성**: 
   - 64×64 기본 이미지 생성
   - 첫 번째 업샘플러: 64×64 → 256×256 (가우시안 블러 기반 손상)
   - 두 번째 업샘플러: 256×256 → 1024×1024 (BSR 손상 기반)

**모델 아키텍처 요약**[1]

| 구성 요소 | 구조 | 파라미터 |
|---------|------|---------|
| CLIP 인코더 | ViT-H/16 | - |
| AR Prior | Transformer (Decoder) | 1B |
| Diffusion Prior | Transformer (Causal) | 1B |
| Decoder | GLIDE 기반 | 3.5B |
| 첫 업샘플러 | ADMNet (conv only) | 700M |
| 두 업샘플러 | ADMNet (conv only) | 300M |

***

### 3. 성능 향상 및 한계

**성능 향상**[1]

1. **MS-COCO 벤치마크 결과**:[1]
   - unCLIP (확산 Prior): FID 10.39 (최고 성능)
   - unCLIP (AR Prior): FID 10.63
   - GLIDE: FID 12.24
   - **37% 상대 개선** (GLIDE 대비)

2. **인간 평가**:[1]
   - 광학적 사실성: 48.9% (GLIDE 추가 선호도)
   - 캡션 유사성: 45.3% 선호도
   - **다양성: 70.5% 우위** (가장 큰 강점)

3. **지도 강도에 따른 효과**[1]

지도 척도 증가 시 성능 비교:
- GLIDE: 의미론적 수렴 발생 (내용 동일화)
- unCLIP: 의미 정보 고정으로 다양성 유지

4. **미학적 품질 평가**[1]
   - AVA 기반 미학 점수에서 GLIDE와 유사한 성능
   - 지도 적용 시에도 Recall 손실 없음

**한계점**[1]

1. **속성 바인딩 문제**[1]
   - 두 객체의 색상 구분 실패: "빨간 정육면체 위에 파란 정육면체"
   - 원인: CLIP 임베딩이 명시적 속성-객체 관계를 인코딩하지 않음
   - 재구성에서 속성과 객체 혼동

$$\text{Binding Error} = \text{Color, Size, Spatial Relation 혼동}$$

2. **텍스트 렌더링 실패**[1]
   - 이미지 내 텍스트 생성 불가: "Deep Learning"이라는 텍스트 표현 불가
   - 원인: BPE 인코딩이 문자 단위 정보를 모호하게 함

3. **복잡한 장면의 세부 부족**[1]
   - 64×64 기본 해상도 후 업샘플링의 정보 손실
   - 해결책: 더 높은 기본 해상도 (계산 비용 증가)

4. **위험성**[1]
   - AI 생성 이미지 탐지 어려움 증가
   - 편향된 학습 데이터로부터의 학습 필요성

***

### 4. 일반화 성능 향상 가능성

**CLIP 임베딩의 일반화 특성**[2]

CLIP 표현은 다음의 일반화 특성을 보유합니다:

1. **분포 이동(Distribution Shift) 견고성**[1]
   - CLIP은 시각-언어 접지(grounding)로 인해 표준 비전 모델보다 강인함
   - 제로샷 성능과 전이 학습 우수

2. **다중모드 정렬의 이점**[2]
   - 텍스트 감독이 시각적 인코더의 의미론적 표현 강화
   - 언어적 개념 구조와의 정렬로 의미론적 일관성 개선

**일반화 개선 방향**[3][4][5]

최신 연구(2024-2025)에서 제시된 개선 방안:

1. **구성적 생성(Compositional Generation) 개선**[4][3]
   - **CompLift**: 리프트 스코어 기반 샘플 필터링으로 복합 프롬프트 만족도 향상
   - **IterComp**: 반복적 구성 인식 피드백으로 다중 객체 배치 개선
   - **SPDiffusion**: 의미론적 보호를 통한 개념 얽힘 해결

2. **속성 바인딩 강화**[5][6]
   - **ColorWave**: RGB 수준의 색상 제어 (학습 없음)
   - **R-Bind**: 엔티티-속성 및 엔티티-관계 바인딩 동시 개선
   - 이중 바인딩 문제에 대한 훈련 무료 해법 제시

3. **잠재 공간 모델링 고도화**[7][8][9]
   - 에너지 기반 모델로 다중 개념 합성
   - 기하학적 불변성 제약 추가
   - 다중 모달 잠재 확산으로 의미론적 표현 개선

$$\text{Generalization} = f(\text{Multi-modal Alignment}, \text{Latent Robustness}, \text{Compositional Binding})$$

***

### 5. 앞으로의 연구 영향 및 고려사항

**학계 및 산업 영향**[10][11]

1. **DALL-E 2의 성공**[11][10]
   - OpenAI 상용화 (DALL-E 2 Preview)
   - 텍스트-투-이미지 생성의 새로운 표준 수립
   - 2022년 이후 다수의 후속 연구 촉발

2. **패러다임 전환**[12][10]
   - 직접 생성에서 중간 표현 활용으로의 전환
   - 잠재 공간 확산의 효율성 입증
   - 계층적 생성 구조의 일반화 방향 제시

3. **학술 기여**[10]
   - SIGGRAPH 2023: 확산 모델 강좌의 핵심 사례 교재
   - 9,000회 이상 인용 (2024 기준)

**향후 연구 방향 및 고려사항**[6][7][3][4]

| 영역 | 현재 문제 | 개선 방향 | 최신 진전 |
|-----|---------|---------|---------|
| **속성 바인딩** | 복수 객체 색상/크기 혼동 | 주의 맵 제어, 에너지 기반 모델 | R-Bind (2025), SPDiffusion (2024) |
| **구성성** | 복잡한 프롬프트 인식 실패 | 리프트 스코어, 거부 샘플링 | CompLift (2025), IterComp (2025) |
| **텍스트 렌더링** | 이미지 내 정확한 텍스트 불가 | 토큰 수준 인코딩 개선 | 진행 중 |
| **일반화** | 분포 이동 시 성능 저하 | 자기 학습, 적응형 미세조정 | SAM 적응 방법 (2024-2025) |
| **계산 효율** | 높은 추론 비용 | 경량 사전, 에너지 효율화 | ECLIPSE (2023), TimeLDM (2024) |

**실무 적용 시 고려사항**[13][14][15][1]

1. **의료 영상 응용**[14][13]
   - DALL-E 2의 X선 이미지 생성 능력 검증 (긍정적)
   - 임상 데이터 증강 활용 가능성
   - 규제 및 윤리 검토 필수

2. **합성 데이터 생성**[9][16][17]
   - 라벨된 데이터 부족 분야에서의 활용 확대
   - 원격 감지, 화학, 단백질 설계 등 다양한 분야 진출
   - 분포 이동 적응 연구의 기초 자료로 활용

3. **안전성 및 신뢰성**[18][1]
   - AI 생성 이미지 탐지 어려움 증가
   - 편향된 학습 데이터로부터의 사회적 영향 평가 필수
   - DALL-E 2 Preview의 시스템 카드 참조

4. **잠재 공간 해석 가능성**[19][20]
   - 리만 기하학적 분석으로 잠재 공간 구조 규명
   - 의미론적 방향 발견 메커니즘 개발
   - 제어 가능성과 안전성의 균형

$$\text{Future Research} = \text{Interpretability} + \text{Compositionality} + \text{Safety} + \text{Efficiency}$$

***

### 결론

unCLIP(DALL-E 2)는 CLIP 임베딩 공간을 매개로 한 계층적 텍스트-투-이미지 생성으로 **다양성-충실성 트레이드오프를 혁신적으로 해결**했습니다. 특히 확산 사전의 효율성 입증과 이미지 조작 능력은 생성 모델 연구에 새로운 패러다임을 제시했습니다.

그러나 속성 바인딩, 텍스트 렌더링, 복잡한 장면 표현 등의 한계는 향후 연구의 중요한 과제입니다. 2024-2025년의 최신 연구들은 **에너지 기반 모델, 의미론적 보호, 구성성 개선** 등을 통해 이러한 한계를 점진적으로 극복하고 있으며, 의료, 환경, 화학 등 다양한 실무 분야로의 확산이 가속화되고 있습니다. 앞으로의 핵심 과제는 생성 모델의 강력한 성능을 안전성, 해석 가능성, 사회적 책임과 조화시키는 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c114407-6f6a-4376-98f5-c40da8f33109/2204.06125v1.pdf)
[2](https://aclanthology.org/2022.repl4nlp-1.4.pdf)
[3](https://arxiv.org/html/2410.07171)
[4](https://arxiv.org/html/2505.13740v1)
[5](https://arxiv.org/abs/2503.09864)
[6](https://aclanthology.org/2025.emnlp-main.349/)
[7](https://ieeexplore.ieee.org/document/11093839/)
[8](https://www.nature.com/articles/s41467-024-54712-1)
[9](https://arxiv.org/abs/2407.04211)
[10](https://dl.acm.org/doi/10.1145/3587423.3595503)
[11](https://3dvar.com/Ramesh2022Hierarchical.pdf)
[12](https://arxiv.org/pdf/2312.04655.pdf)
[13](http://arxiv.org/pdf/2209.13696.pdf)
[14](https://arxiv.org/pdf/2312.14988.pdf)
[15](https://pmc.ncbi.nlm.nih.gov/articles/PMC10131692/)
[16](http://biorxiv.org/lookup/doi/10.1101/2023.08.22.554145)
[17](https://arxiv.org/abs/2404.08892)
[18](https://gmd.copernicus.org/articles/18/2051/2025/)
[19](https://papers.neurips.cc/paper_files/paper/2023/file/4bfcebedf7a2967c410b64670f27f904-Paper-Conference.pdf)
[20](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Improving_the_Generalization_of_Segmentation_Foundation_Model_under_Distribution_Shift_CVPR_2024_paper.pdf)
[21](https://e-journal.unair.ac.id/JESTT/article/view/47782)
[22](https://arxiv.org/pdf/2211.12112.pdf)
[23](https://arxiv.org/vc/arxiv/papers/2204/2204.13807v1.pdf)
[24](https://arxiv.org/pdf/2212.07839.pdf)
[25](https://arxiv.org/pdf/2306.07005.pdf)
[26](https://learnopencv.com/mastering-dall-e-2/)
[27](https://academic.oup.com/bioinformatics/article/41/8/btaf426/8219452)
[28](https://cocoa-t.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-DALL-E-2-Hierarchical-Text-Conditional-Image-Generation-with-CLIP-Latents)
[29](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Segment_Any-Quality_Images_with_Generative_Latent_Space_Enhancement_CVPR_2025_paper.pdf)
[30](https://github.com/lucidrains/DALLE2-pytorch)
[31](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03044.pdf)
[32](https://cdn.openai.com/papers/dall-e-2.pdf)
[33](https://academic.oup.com/bib/article/doi/10.1093/bib/bbae142/7640735)
[34](https://www.semanticscholar.org/paper/e0eac8c64be3313e581c28a495bec192e7e67284)
[35](https://ieeexplore.ieee.org/document/10446349/)
[36](https://arxiv.org/abs/2412.05043)
[37](http://arxiv.org/pdf/2303.11916.pdf)
[38](https://arxiv.org/pdf/2304.12536.pdf)
[39](https://arxiv.org/pdf/2308.10040.pdf)
[40](http://arxiv.org/pdf/2410.01594.pdf)
[41](https://arxiv.org/html/2312.02548)
[42](https://arxiv.org/html/2408.03637v1)
[43](http://arxiv.org/pdf/2112.10752.pdf)
[44](https://arxiv.org/abs/2412.14706)
[45](https://arxiv.org/pdf/2411.05824.pdf)
[46](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_EnergyMoGen_Compositional_Human_Motion_Generation_with_Energy-Based_Diffusion_Model_in_CVPR_2025_paper.pdf)
[47](https://arxiv.org/abs/2409.01327)
[48](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d3b57e06e3fc45f077eb5c9f28156d4-Paper-Conference.pdf)
[49](https://dl.acm.org/doi/10.1145/3680528.3687645)
