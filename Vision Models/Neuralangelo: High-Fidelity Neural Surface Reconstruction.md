# Neuralangelo: High-Fidelity Neural Surface Reconstruction

### 1. 핵심 주장과 주요 기여

Neuralangelo는 **다중 해상도 3D 해시 그리드와 신경 표면 렌더링을 결합**하여 보조 데이터(깊이 맵, 세그멘테이션) 없이 RGB 이미지로부터 고충실도 3D 표면 재구성을 수행하는 프레임워크입니다.[1]

**핵심 기여는 세 가지입니다:**

첫째, **다중 해상도 해시 인코딩을 신경 부호화거리함수(SDF) 표현에 통합**함으로써 기존 방법의 표현력 한계를 극복했습니다. 둘째, **수치 기울기를 통한 고차 도함수 계산**과 **점진적 세부 최적화 전략**이라는 두 가지 핵심 기법을 제시하여 해시 인코딩의 잠재력을 완전히 활용했습니다. 셋째, 광범위한 실험을 통해 기존 신경 표면 재구성 방법 대비 **재구성 정확도와 뷰 합성 품질 모두에서 유의미한 성능 향상**을 입증했습니다.[1]

***

### 2. 해결 문제 및 제안 방법

#### 2.1 기본 문제의 정의

신경 표면 재구성은 다중 시점 이미지로부터 조밀한 3D 기하구조를 복원하는 것을 목표로 합니다. 그러나 기존 신경 표면 재구성 방법들은 **실제 장면의 세부 구조 복원에 어려움**을 겪었습니다. 특히, 다층퍼셉트론(MLP) 기반 방법들은 표현 용량에 따라 충실도가 개선되지 않는 한계가 있었습니다.[1]

#### 2.2 신경 볼륨 렌더링 기반

Neuralangelo는 신경 볼륨 렌더링에 기반하며, 카메라 레이를 따라 샘플링된 점들의 색상과 밀도를 합성합니다. 렌더링된 픽셀 색상은 다음 리만 합으로 계산됩니다:[1]

$$\hat{\text{rgb}}(\text{camera}, \text{direction}) = \sum_{i=1}^N w_i \cdot \text{rgb}_i$$

여기서:
$$w_i = T_i \alpha_i, \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i), \quad T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$$

색상 손실은 입력 이미지와 렌더링된 이미지 간 L1 거리로 정의됩니다:[1]

$$\mathcal{L}_{\text{rgb}} = \|\hat{\text{rgb}} - \text{rgb}\|_1$$

#### 2.3 SDF 기반 볼륨 렌더링

명확한 표면 정의를 위해 SDF 표현을 채택합니다. 3D 점 $$x_i$$에서 SDF 값 $$f(x_i)$$에 대해, 불투명도는 로지스틱 함수로 변환됩니다:[1]

$$\alpha_i = \max\left(\frac{\Phi_s(f(x_i)) - \Phi_s(f(x_{i+1}))}{{\Phi_s(f(x_i))}}, 0\right)$$

여기서 $$\Phi_s$$는 시그모이드 함수입니다.

#### 2.4 다중 해상도 해시 인코딩

다중 해상도 그리드 집합 $$\{V_1, ..., V_L\}$$을 사용하여 공간 위치를 인코딩합니다. 위치 $$x_i$$는 각 그리드 해상도 $$V_l$$에 대해 스케일링됩니다:[1]

$$x_{i,l} = x_i \cdot V_l$$

삼선형 보간을 통해 얻은 특징 벡터는:[1]

$$\gamma_l(x_{i,l}) = \gamma_l(\lfloor x_{i,l} \rfloor) \cdot (1-\beta) + \gamma_l(\lceil x_{i,l} \rceil) \cdot \beta$$

여기서 $$\beta = x_{i,l} - \lfloor x_{i,l} \rfloor$$입니다. 모든 해상도의 특징은 연결되어 MLP로 전달됩니다:[1]

$$\gamma(x_i) = (\gamma_1(x_{i,1}), ..., \gamma_L(x_{i,L}))$$

#### 2.5 수치 기울기를 통한 고차 도함수 계산

**핵심 혁신:** 해시 인코딩의 분석적 기울기는 공간에서 불연속이어서 국소적입니다. 삼선형 보간에서 반올림 연산이 미분 불가능하므로:[1]

$$\frac{\partial \gamma_l(x_{i,l})}{\partial x_i} = \gamma_l(\lfloor x_{i,l}\rfloor) \cdot (-V_l) + \gamma_l(\lceil x_{i,l} \rceil) \cdot V_l$$

이는 eikonal 손실이 국소 그리드 셀에만 역전파되는 문제를 야기합니다.[1]

이를 해결하기 위해 수치 기울기를 도입합니다. 중앙차분 공식으로 표면 법선을 계산하면:[1]

$$\nabla_x f(x_i) = \frac{f(x_i + \epsilon_x) - f(x_i - \epsilon_x)}{2\epsilon}$$

여기서 $$\epsilon_x = [\epsilon, 0, 0]$$입니다. 총 6개의 추가 SDF 샘플이 필요합니다. 단계 크기 $$\epsilon$$가 그리드 크기보다 작으면, 수치 기울기는 분석적 기울계와 동치이고, 그보다 크면 **여러 그리드 셀의 특징들이 동시에 최적화 업데이트를 받게 됩니다**. 이는 연속 표면에 대한 일관된 표면 법선을 보장합니다.[1]

**Eikonal 손실:**[1]

$$\mathcal{L}_{\text{eik}} = \frac{1}{N} \sum_{i=1}^N (\|\nabla f(x_i)\|_2 - 1)^2$$

#### 2.6 점진적 세부 최적화

**단계 크기 감소:** 초기에는 큰 $$\epsilon$$로 시작하여 지수적으로 감소시킵니다. 큰 $$\epsilon$$는 smoothing 효과로 광역적 형태를 캡처하고, 작은 $$\epsilon$$는 세부 구조를 복원합니다.[1]

**해시 그리드 해상도 점진적 활성화:** 초기에는 저해상도 그리드만 활성화하다가, $$\epsilon$$가 각 해상도의 그리드 크기와 같아질 때 새로운 해상도를 점진적으로 활성화합니다. 이는 고해상도 그리드가 저해상도 최적화를 "제학습(unlearn)"하고 다시 학습하는 과정을 피합니다.[1]

#### 2.7 곡률 정규화

표면 부드러움을 강제하기 위해 평균 곡률을 정규화합니다:[1]

$$\mathcal{L}_{\text{curv}} = \frac{1}{N} \sum_{i=1}^N \nabla^2 f(x_i)$$

이차 기울기는 표면 법선 계산에 사용된 샘플로부터 계산됩니다.[1]

#### 2.8 전체 손실 함수

최종 손실은 세 항의 가중 합으로 정의됩니다:[1]

$$\mathcal{L} = \mathcal{L}_{\text{rgb}} + w_{\text{eik}} \mathcal{L}_{\text{eik}} + w_{\text{curv}} \mathcal{L}_{\text{curv}}$$

***

### 3. 모델 구조 및 구현

#### 3.1 신경망 구조

Neuralangelo는 두 개의 MLP로 구성됩니다:[1]

- **SDF MLP:** 1개 계층, 다중 해상도 해시 인코딩 특징을 입력받아 SDF 값 $$f(x)$$를 출력
- **색상 MLP:** 4개 계층, 점 위치, SDF MLP 특징, 보기 방향, 표면 법선을 입력받아 RGB 색상 출력

#### 3.2 해시 인코딩 설정

실험에서는 다음 사양을 사용합니다:[1]

- 해시 인코딩 해상도: $$2^5$$에서 $$2^{11}$$ (16개 레벨)
- 각 해시 엔트리: 채널 크기 8
- 최대 해시 엔트리 수: $$2^{22}$$
- DTU 데이터셋: 초기에 4개 해상도 활성화
- Tanks and Temples 데이터셋: 초기에 8개 해상도 활성화

#### 3.3 최적화 설정

- **총 훈련 반복:** 500,000회
- **학습률:** $$1 \times 10^{-3}$$ (선형 warmup 5,000회)
- **학습률 감쇠:** 300k, 400k에서 10배 감쇠
- **옵티마이저:** AdamW, 가중치 감쇠 $$10^{-2}$$
- **손실 가중치:** $$w_{\text{eik}} = 0.1$$, $$w_{\text{curv}}$$는 동적 스케줄링
- **배치 크기:** DTU는 1, Tanks and Temples는 16

#### 3.4 메시 추출

marching cubes 알고리즘으로 예측된 SDF를 삼각형 메시로 변환합니다:[1]
- DTU: 512 해상도
- Tanks and Temples: 2048 해상도

***

### 4. 성능 향상 및 실험 결과

#### 4.1 DTU 벤치마크

| 방법 | Chamfer 거리 (mm)↓ | PSNR ↑ |
|------|------------------|---------|
| NeRF | 1.90 | 24 |
| VolSDF | 1.14 | 37 |
| NeuS | 1.00 | 40 |
| HF-NeuS | 0.76 | 55 |
| RegSDF† | 0.84 | 63 |
| NeuralWarp† | 0.77 | 65 |
| **Neuralangelo (Ours)** | **0.61** | **69** |

Neuralangelo는 **평균 Chamfer 거리에서 0.61mm로 최저**를 달성하고, **PSNR 69로 최고 값**을 기록했습니다. 특히 보조 입력 없이 이러한 성과를 달성했습니다.[1]

#### 4.2 Tanks and Temples 벤치마크

| 장면 | F1 점수↑ (NeuralWarp) | F1 점수↑ (NeuS) | F1 점수↑ (Ours) | PSNR (Ours) |
|-----|-----|-----|-----|-----|
| Barn | 0.22 | 0.29 | **0.70** | 28.57 |
| Caterpillar | 0.18 | 0.29 | **0.36** | 27.81 |
| Courthouse | 0.08 | 0.17 | **0.28** | 27.23 |
| Ignatius | 0.02 | 0.83 | **0.89** | 23.67 |
| Meetingroom | 0.08 | 0.24 | **0.32** | 30.70 |
| Truck | 0.35 | 0.45 | **0.48** | 25.43 |
| **평균** | **0.15** | **0.38** | **0.50** | **27.24** |

대규모 실외 장면에서도 Neuralangelo는 F1 점수 0.50으로 우수한 성능을 보였습니다.[1]

#### 4.3 주요 절제(Ablation) 결과

**분석적 기울기 (AG) vs. 수치 기울기 (NG):**[1]
- AG: 노이즈가 많은 표면 생성
- AG+P (AG + 점진적 해상도 활성화): 개선되지만 여전히 부정확함
- NG (수치 기울기만): 표면은 부드럽지만 세부 사항 손실
- NG+P (제안 방법): 부드러운 표면과 세부 사항 모두 보존

**곡률 정규화 효과:** $$\mathcal{L}_{\text{curv}}$$을 제거하면 표면에 불원하는 급격한 전환이 발생합니다.[1]

**위상 warmup:** 초기 구를 기반으로 하면 $$\mathcal{L}_{\text{curv}}$$이 오목한 형태 형성을 방해합니다. 짧은 warmup 기간으로 곡률 손실 강도를 선형 증가시켜 이를 해결합니다.[1]

***

### 5. 일반화 성능 및 한계

#### 5.1 일반화 성능 향상 가능성

**현재 강점:**

1. **보조 데이터 불필요:** Neuralangelo는 깊이 맵이나 세그멘테이션 없이 RGB 이미지만으로 재구성합니다. 이는 **더 나은 일반화 적용성**을 의미합니다.[1]

2. **계층적 비구조적 표현:** 해시 인코딩은 전체 공간에서 기울기를 분배하여 **임의의 위상 변화를 처리**할 수 있습니다. 고해상도에서만 표현될 수 있는 세부 구조를 점진적으로 복구합니다.[1]

3. **점진적 최적화의 강건성:** 코스-투-파인 전략이 손실 지형을 더 나은 국소 극값으로 유도하여 **다양한 씬의 일반화 능력을 향상**시킵니다.[1]

4. **사실상의 전이:** 논문의 추가 in-the-wild 결과(NVIDIA HQ Park, Johns Hopkins University)에서 **동일한 하이퍼파라미터 설정으로 다양한 장면을 성공적으로 재구성**하여 일반화 가능성을 입증합니다.[1]

**최신 연구 동향:**

최근 연구는 신경 표면 재구성의 일반화 성능을 다음과 같이 개선하고 있습니다:[2]

- **GenS (Generalizable Neural Surface Reconstruction):** 씬별 최적화 대신 **멀티스케일 부피로 모든 씬을 직접 인코딩**하는 방식으로 새로운 씬에 대한 일반화를 달성합니다.[2]

- **멀티스케일 특징-메트릭 일관성:** 포토메트릭 일관성 실패에 **견고한 판별적 멀티스케일 특징 공간**을 도입합니다.[2]

- **뷰 대조 손실:** 적은 시점으로 촬영된 영역에 대한 기하 사전을 학습하여 **희소 입력에 대한 견고성**을 강화합니다.[2]

#### 5.2 주요 한계

**1. 반사 표면 처리 부족:**

논문에서 명시적으로 인정한 한계는 **고반사 재료(highly reflective materials)**에 대한 성능입니다. Instant NGP(해시 인코딩의 기반)도 반사 장면에서 Fourier 특징 인코딩 + 심층 MLP보다 성능이 낮습니다.[1]

최근 연구들이 이 문제를 다루고 있습니다:[3][4][5]

- **NeuS-HSR:** 고반사 장면에서 반사 분해를 통해 목표 객체 표면을 복원합니다.[5]
- **Ref-NeuS:** 이상 탐지기를 사용하여 반사 점수를 추정하고 반사 표면의 영향을 감소시킵니다.[4]
- **NeRSP:** 편광 이미지를 활용하여 반사 표면의 모호성을 해결합니다.[3]

**2. 수치 기울계 계산 오버헤드:**

분석적 기울계 대비 약 **1.2배 훈련 속도 저하**가 발생합니다. 그러나 1계층 MLP 사용으로 여전히 8계층 MLP를 사용하는 NeuS보다 빠릅니다.[1]

**3. 훈련 시간 효율성:**

논문은 "임의의 픽셀 샘플링 전략 없이 랜덤 샘플링을 사용하므로 긴 훈련 반복이 필요하다"고 명시합니다. 이는 **실시간 응용**에 제약이 됩니다.[1]

**4. 특정 재료의 제한:**

- 간단한 텍스처와 저대비 영역에서 개선 폭 감소
- 높은 반사율 객체(버튼, 눈 구조)의 세부 사항 손실

**5. 일반화 성능의 제약:**

- **씬별 최적화 필요:** Neuralangelo는 여전히 각 씬마다 별도 네트워크 훈련이 필요합니다. 이는 새로운 씬으로의 **직접 전이 불가능**을 의미합니다.[1]
- 기하학적 구조가 매우 다른 씬으로의 외삽 능력 부족

***

### 6. 앞으로의 연구에 미치는 영향 및 고려사항

#### 6.1 연구 영향

**1. 신경 표현의 새로운 패러다임:**

Neuralangelo는 **해시 기반 인코딩이 신경 표면 재구성에 효과적**임을 입증하여, 후속 연구에서 이 표현을 채용하도록 영감을 주었습니다.[6][7][8]

**2. 수치 기울기의 가치:**

분석적 기울기의 국소성 문제를 수치 기울기로 우아하게 해결한 접근이 **다른 암묵적 함수 학습 문제에 적용** 가능함을 보여줍니다.[1]

**3. 점진적 최적화 전략의 일반성:**

코스-투-파인 최적화가 **고충실도 신경 재구성의 필수 요소**임을 확립했습니다. 최근 HF-NeuS, NeuRodin 등의 후속 방법들이 이 전략을 채용하고 있습니다.[9][10]

#### 6.2 향후 연구 시 고려사항

**1. 일반화 능력 강화:**

- **씬별 최적화 제거:** GenS 같은 generalizable 접근을 Neuralangelo의 고충실도 표현과 결합하여 새로운 씬에 직접 적용 가능한 모델 개발
- **도메인 적응:** 다양한 카메라, 조명, 재료 조건에 대한 견고성 향상

**2. 반사 장면 처리:**

- **물리 기반 분해:** 반사, 굴절, 산란을 명시적으로 모델링
- **멀티모달 입력 활용:** 편광 이미지, 라이다, 깊이 센서 등 통합

**3. 효율성 개선:**

- **지능형 샘플링:** 임의 샘플링 대신 오류 기반 또는 불확실성 기반 샘플링으로 훈련 시간 단축
- **실시간 처리:** 점진적 메시 추출, 적응형 해상도 선택으로 실시간 응용 가능성 탐색

**4. 노이즈 입력 견고성:**

- **카메라 포즈 오류 처리:** SG-NeRF, Robust SG-NeRF 같은 선행 연구에서 outlier 카메라 포즈 처리 메커니즘 적용[11][12]
- **노이즈 깊이/세그멘테이션 활용:** 부정확한 보조 정보의 효과적 통합

**5. 동적 장면 확장:**

- **시공간 해시 인코딩:** Masked Space-Time Hash Encoding 같은 기법으로 비디오 시퀀스 실시간 재구성[8]
- **동적 그래프 구조:** 장시간 변형되는 객체의 추적 및 재구성

**6. 물리적 정확성:**

- **측도 논(metrological) 정확성:** 산업 검사, 의료 영상 등 정밀도가 중요한 응용에 대한 정량화
- **불확실성 추정:** 재구성 신뢰도 맵 생성으로 신뢰도 평가

**7. 교차 모달 학습:**

- **합성 데이터 활용:** 사전학습을 통해 실제 데이터로의 미세조정 필요성 감소
- **자기감독 학습:** 레이블 없는 데이터에서 기하 선행 학습

#### 6.3 최신 트렌드 (2024-2025)

**다중 시점 재구성의 진화:**

최근 3D 재구성 분야는 다음 방향으로 진화하고 있습니다:[13]

- **자기감독과 멀티뷰 일관성:** 대량 주석 데이터 의존성 감소
- **희소 표현:** Neuralangelo의 해시 인코딩을 더욱 희소화하여 $$512^3$$ 해상도 달성[7]
- **3D Gaussian Splatting (3DGS):** 암묵적 표현 대신 명시적 가우시안 기반 렌더링으로 실시간 성능 달성

**신경 표면 재구성의 새로운 도전:**

- **개방형 표면:** 폐쇄형 표면만 처리하는 SDF 대신 부호 없는 거리함수(UDF) 채택으로 개방형 구조 지원
- **고반사 장면:** 편광 이미징, 광학 흐름, 스페큘러 분해를 통한 견고성 강화
- **대규모 장면:** 언바운드 장면과 무한 평면 처리의 표준화

***

## 결론

Neuralangelo는 **수치 기울기 기반의 고차 도함수 계산**과 **점진적 해시 그리드 최적화**라는 두 가지 간단하면서도 효과적인 기법을 통해 신경 표면 재구성의 충실도 한계를 돌파했습니다. DTU 및 Tanks and Temples 벤치마크에서 지표적 성과를 기록하면서 **보조 데이터 없는 실용적 3D 재구성**의 가능성을 증명했습니다.[1]

다만, 반사 표면 처리, 씬별 최적화 필요성, 훈련 효율성 등의 한계는 **향후 연구의 중요한 과제**입니다. 최근 GenS, Sparse 표현 방법, 편광 기반 재구성 등의 후속 연구들이 이러한 제약을 체계적으로 해결하고 있으며, 신경 표면 재구성 분야는 **일반화 능력, 다양한 장면 종류 지원, 실시간 성능**을 중심으로 빠르게 진화하고 있습니다.[7][13][3][2]

***

### 참고 자료 인덱싱

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6ea124f5-2219-4b47-82b2-7c32cc20e671/2306.03092v2.pdf)
[2](https://arxiv.org/abs/2406.02495)
[3](https://arxiv.org/abs/2406.07111)
[4](https://arxiv.org/pdf/2303.10840.pdf)
[5](https://openaccess.thecvf.com/content/CVPR2023/papers/Qiu_Looking_Through_the_Glass_Neural_Surface_Reconstruction_Against_High_Specular_CVPR_2023_paper.pdf)
[6](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08870.pdf)
[7](https://openreview.net/forum?id=m6W5SfQXrT)
[8](https://proceedings.neurips.cc/paper_files/paper/2023/file/df31126302921ca9351fab73923a172f-Paper-Conference.pdf)
[9](https://arxiv.org/html/2408.10178)
[10](https://proceedings.nips.cc/paper_files/paper/2022/file/0ce8e3434c7b486bbddff9745b2a1722-Paper-Conference.pdf)
[11](https://arxiv.org/abs/2411.13620)
[12](https://arxiv.org/pdf/2308.07868.pdf)
[13](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473764/)
[14](https://arxiv.org/html/2306.03092)
[15](https://www.mdpi.com/2306-5354/11/5/487/pdf?version=1715669531)
[16](https://pmc.ncbi.nlm.nih.gov/articles/PMC4468747/)
[17](https://pmc.ncbi.nlm.nih.gov/articles/PMC11468770/)
[18](https://academic.oup.com/bib/article/doi/10.1093/bib/bbae393/7731493)
[19](https://pmc.ncbi.nlm.nih.gov/articles/PMC11118929/)
[20](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/adma.202401750)
[21](https://pmc.ncbi.nlm.nih.gov/articles/PMC4652241/)
[22](https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf)
[23](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Neuralangelo_High-Fidelity_Neural_Surface_Reconstruction_CVPR_2023_paper.pdf)
[24](https://arxiv.org/html/2506.19491v1)
[25](https://research.nvidia.com/labs/dir/neuralangelo/paper.pdf)
[26](http://ieeevis.org/year/2024/program/paper_v-short-1049.html)
[27](https://www.scribd.com/document/666185431/High-Fidelity-Neural-Surface-Reconstruction)
[28](https://arxiv.org/abs/2106.10689)
[29](https://arxiv.org/pdf/2401.12751.pdf)
[30](https://arxiv.org/html/2406.02495v1)
[31](https://pure.korea.ac.kr/en/publications/unified-domain-generalization-and-adaptation-for-multi-view-3d-ob/)
[32](https://proceedings.neurips.cc/paper_files/paper/2021/file/e41e164f7485ec4a28741a2d0ea41c74-Paper.pdf)
[33](https://cvpr.thecvf.com/virtual/2023/poster/23244)
[34](https://arxiv.org/html/2501.09460v1)
[35](https://dl.acm.org/doi/fullHtml/10.1145/3641519.3657403)
[36](https://arxiv.org/pdf/2501.11020.pdf)
