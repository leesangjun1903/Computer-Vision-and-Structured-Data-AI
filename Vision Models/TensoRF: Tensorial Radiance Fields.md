# TensoRF: Tensorial Radiance Fields

### 1. 핵심 주장과 주요 기여

**TensoRF**의 중심 개념은 **4D 텐서로 모델링된 방사 필드(Radiance Field)를 저차(Low-rank) 텐서 성분으로 분해**하는 것입니다. NeRF의 순수 MLP 기반 접근과 달리, TensoRF는 3D 장면을 명시적 복셀 격자(Voxel Grid)로 표현합니다.[1]

**주요 기여:**

- **CP 분해(CANDECOMP/PARAFAC)의 적용**: 순수 벡터 외적을 통해 저차 성분으로 텐서를 분해하여 NeRF보다 빠른 학습과 더 작은 모델 크기(< 4MB)를 달성[1]

- **신규 VM 분해(Vector-Matrix Decomposition) 제안**: 두 모드의 제약을 완화하여 벡터와 행렬의 혼합으로 분해. 동일한 표현 용량으로 필요한 성분 수를 감소시켜 더 빠른 재구성(< 10분)과 우수한 렌더링 품질 달성[1]

- **메모리 효율성**: 공간 복잡도를 O(N³)에서 CP의 경우 O(N), VM의 경우 O(N²)로 감소[1]

***

### 2. 해결 문제 및 제안 방법

**문제점:**
NeRF는 MLP 기반이라 학습 시간이 과도하게 오래 걸리고(수십 시간~수일), 복셀 격자 기반 방법들은 메모리 사용량이 매우 많습니다.[1]

**제안 방법:**

#### **수학적 기반**

**CP 분해(3D 텐서 예시):**

$$T = \sum_{r=1}^{R} \mathbf{v}^1_r \circ \mathbf{v}^2_r \circ \mathbf{v}^3_r$$

여기서 $$\mathbf{v}^m_r$$는 m번째 모드의 r번째 벡터이고, $$\circ$$는 외적입니다.[1]

각 텐서 원소는:

$$T_{ijk} = \sum_{r=1}^{R} v^1_{r,i} \cdot v^2_{r,j} \cdot v^3_{r,k}$$

**VM 분해 (제안한 새로운 방법):**

$$\text{Tensor} = \sum_{r=1}^{R} \left( \mathbf{v}^X_r \circ \mathbf{M}^{YZ}_r + \mathbf{v}^Y_r \circ \mathbf{M}^{XZ}_r + \mathbf{v}^Z_r \circ \mathbf{M}^{XY}_r \right)$$

여기서 $$\mathbf{M}^{YZ}_r$$는 YZ 평면의 행렬(J×K)입니다.[1]

텐서 원소 계산:

$$T_{ijk} = \sum_{r=1}^{R} \sum_{m \in \{X,Y,Z\}} A^m_{r,ijk}$$

여기서 $$A^X_{r,ijk} = v^X_{r,i} \cdot M^{YZ}_{r,jk}$$ 등입니다.[1]

#### **모델 구조**

**밀도와 외관 그리드 분해:**

밀도 그리드($$\mathcal{G}_\sigma$$, 3D 텐서)의 VM 분해:

$$\text{Grid}_\sigma = \sum_{r=1}^{R} \left( \mathbf{v}^X_{\sigma,r} \circ \mathbf{M}^{YZ}_{\sigma,r} + \mathbf{v}^Y_{\sigma,r} \circ \mathbf{M}^{XZ}_{\sigma,r} + \mathbf{v}^Z_{\sigma,r} \circ \mathbf{M}^{XY}_{\sigma,r} \right)$$

외관 그리드($$\mathcal{G}_c$$, 4D 텐서, P개 채널)의 분해는 특성 차원에 벡터만 사용:

$$\text{Grid}_{rad} = \sum_{r=1}^{R_c} \left[ \mathbf{v}^X_{c,r} \circ \mathbf{M}^{YZ}_{c,r} \circ \mathbf{b}_r + \cdots \right]$$

행렬 $$\mathbf{B}$$는 모든 $$\mathbf{b}_r$$을 열로 스택하여 구성되는 글로벌 외관 사전입니다.[1]

**효율적 특성 계산:**

직접 계산:

$$\sigma_{ijk} = \sum_{r=1}^{R} \sum_{m \in \{X,Y,Z\}} A^m_{\sigma,r,ijk}$$

외관 특성(전체 P-채널 벡터):

$$\mathbf{d}_{c,ijk} = \mathbf{B} \cdot \mathbf{A}^c_{ijk}$$

여기서 $$\mathbf{A}^c_{ijk}$$는 모든 성분의 특성을 연결한 벡터입니다.[1]

**삼선형 보간:**

삼선형 보간의 핵심 장점:

$$A^X_r(x) = \mathbf{v}^X_r(x) \otimes \mathbf{M}^{YZ}_r(y,z)$$

여기서 $$\mathbf{v}^X_r(x)$$는 선형 보간, $$\mathbf{M}^{YZ}_r(y,z)$$는 이중선형 보간입니다. 이는 8개 텐서 값의 보간 대신 직접 보간된 값을 계산하므로 계산량이 8배 감소합니다.[1]

**볼륨 렌더링 및 재구성:**

연속 방사 필드:

$$C(\mathbf{r}) = \sum_{q=1}^{Q} T_q (1 - p_q) \left(1 - \exp\left(-\sigma(\mathbf{x}_q) \delta_q\right)\right)$$

렌더링 손실함수:

$$\mathcal{L}_{\text{total}} = \|\mathbf{C} - \tilde{\mathbf{C}}\|^2_2 + \lambda_1 \mathcal{L}_{L1} + \lambda_2 \mathcal{L}_{TV}$$

- $$\mathcal{L}_{L1}$$: L1 정규화(희소성 유도)
- $$\mathcal{L}_{TV}$$: 총변위 정규화(실제 데이터셋 사용 시)[1]

***

### 3. 성능 향상 및 한계

#### **성능 향상**

| 평가 항목 | NeRF | TensoRF-CP | TensoRF-VM(192) |
|---------|------|-----------|-----------------|
| **PSNR (Synthetic-NeRF)** | 31.01 | 31.56 | **33.14** |
| **SSIM** | 0.947 | 0.949 | **0.963** |
| **학습 시간** | 35시간 | 25.2분 | **8.1-17.4분** |
| **모델 크기** | 5.0MB | 3.9MB | 71.8MB |
| **속도 향상** | 1x | ~70x | ~100x |

Synthetic-NSVF 데이터셋:
- TensoRF-VM(192): PSNR **36.52** (DVGO 35.08 vs.)[1]

Tanks & Temples(실제 데이터):
- TensoRF-VM(192): PSNR **28.56** (NeRF 25.78 vs.)[1]

#### **주요 개선 사항**

1. **계산 효율성**: CP/VM 분해로 인한 저차 정규화가 자연스럽게 과적합을 방지[1]

2. **점진적 재구성**: 저해상도($$128^3$$)에서 시작해 선형 및 이중선형 업샘플링으로 단계적 해상도 증가 (2000, 3000, 4000, 5500, 7000 스텝에서)[1]

3. **이중 그리드 구조**: 밀도와 외관 그리드를 분리하여 각각 최적화[1]

#### **한계**

1. **제한된 장면 타입**: 단일 경계 박스를 가진 장면만 지원. **무한 장면(unbounded scenes)은 처리 불가**. 배경과 전경이 함께 있는 야외 장면 적용 불가[1]

2. **일반화 성능의 한계**: 논문은 **장면별 최적화(per-scene optimization)만 고려**하며, 일반화 가능한 모델로의 확장을 미래 연구로 제시[1]

3. **카메라 포즈 의존성**: 알려진 카메라 포즈를 요구하며, 자동 포즈 추정은 미지원[1]

***

### 4. 일반화 성능 향상 가능성[1]

#### **현재 제약:**

- TensoRF는 각 장면에 대해 별도로 최적화되어야 하므로 새로운 장면에 대해 재학습 필요
- 저차 텐서 분해의 정규화는 단일 장면 최적화에는 효과적이지만, 다중 장면 학습에는 제약 존재

#### **개선 방향:**

최신 연구(2023-2024)에서 제시된 해결책들:

1. **크로스 장면 일반화 방법들**: 여러 장면에서 학습하여 새로운 장면에 직접 적용 가능한 모델 개발 진행 중[2][3][4]

2. **저차 정규화의 활용**: 논문의 저차 텐서 분해 개념이 여러 후속 연구에 채택되어 모델 압축과 일반화 개선에 활용[5][6][7]

3. **동적 장면으로의 확장**: D-TensoRF는 시간 축을 추가한 5D 텐서로 동적 장면 처리 가능[8]

4. **역렌더링(Inverse Rendering) 확장**: TensoIR은 TensoRF를 기반으로 기하, 반사율, 조명 추정이 가능한 물리 기반 모델 개발[6]

***

### 5. 앞으로의 연구 시 고려할 점 (최신 연구 기반)[9][7][10][5][8][6]

#### **1단계: 기본 문제 해결**

- **무한 장면 처리**: Cubemap 기반 접근 또는 배경 표현 개선으로 야외 장면 지원[11]
- **일반화 성능**: 다중 장면 데이터에서의 저차 분해 효과 검증 필요. MRVM-NeRF 등 최신 방법에서 마스크 기반 모델링으로 일반화 개선 중[12]

#### **2단계: 응용 분야 확장**

- **의미론적 정보 통합**: NeRF-IS 같은 방법에서 TensoRF의 텐서 분해를 의미적 필드 모델링에 활용 가능[5]
- **얼굴 및 인간 재구성**: TIFace에서 TensoRF 기반 개선으로 정밀 재구성 달성[13]
- **멀티모달 응용**: 열화상, 적외선 등 다중 센서 데이터와의 융합[14]

#### **3단계: 성능 최적화**

- **하드웨어 가속**: Gen-NeRF 같은 알고리즘-하드웨어 공동설계로 실시간 일반화 NeRF 실현[2]
- **스파스 입력 강화**: 적은 입력 이미지 하에서의 강건성 개선. Simple-RF의 정규화 프레임워크 적용[9]

#### **4단계: 혁신적 확장**

- **생성 모델 통합**: 사전학습 확산 모델(Diffusion Model)을 활용한 더 나은 3D 표현 학습[15]
- **대규모 장면 모델링**: 계층적 로컬 필드 또는 멀티 스페이스 표현으로 대규모/복잡 장면 처리[16][17]
- **작업 특화 최적화**: 점유도 예측, 의미 분할 등 특정 비전 작업에 최적화된 텐서 분해 방법 개발[18]

#### **5단계: 근본적 이론 발전**

- **텐서 분해 이론 개선**: 블록 항 분해(BTD) 또는 새로운 분해 방식의 3D 비전 응용[19]
- **저차 정규화의 명시적 모델링**: 장면 복잡도에 따른 최적 텐서 순위 자동 결정[1]

#### **구체적 권장사항**

1. **멀티 스케일 텐서 분해** 도입: 거친 스케일에서 세밀한 스케일까지 계층적 표현[7]

2. **적응형 정규화**: 장면 특성에 따라 L1과 TV 손실의 가중치를 자동 조정[1]

3. **크로스 도메인 사전 학습**: 대규모 장면 데이터셋에서 사전학습된 기저 성분(basis components)을 활용한 전이 학습

4. **불확실성 정량화**: 테스트 시 모델 신뢰도를 추정하여 스파스 입력이나 out-of-distribution 장면에서의 견고성 향상

***

## 결론

TensoRF는 **텐서 분해를 3D 장면 재구성에 처음 적용**하여 NeRF 대비 100배 이상의 속도 향상과 우수한 화질을 동시에 달성했습니다. 특히 VM 분해의 도입으로 모델 표현력과 계산 효율의 균형을 효과적으로 해결했습니다. 앞으로는 **무한 장면 처리, 크로스 장면 일반화, 다양한 비전 작업으로의 확장**이 핵심 과제이며, 최신 연구에서 이들 방향의 개선이 활발히 진행 중입니다.[8][6][7][12][9][5][2][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/243369e2-00e9-4057-a292-14e8c4249985/2203.09517v2.pdf)
[2](https://dl.acm.org/doi/10.1145/3579371.3589109)
[3](https://ieeexplore.ieee.org/document/10658469/)
[4](https://arxiv.org/abs/2312.09095)
[5](https://dl.acm.org/doi/10.1145/3595916.3626379)
[6](https://arxiv.org/html/2304.12461v2)
[7](https://arxiv.org/abs/2303.03808)
[8](https://arxiv.org/abs/2212.02375)
[9](https://arxiv.org/abs/2404.19015)
[10](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/tensorf/)
[11](https://openaccess.thecvf.com/content/WACV2024/papers/Chang_Fast_Sun-Aligned_Outdoor_Scene_Relighting_Based_on_TensoRF_WACV_2024_paper.pdf)
[12](https://proceedings.iclr.cc/paper_files/paper/2024/file/8882d370cdafec9885b918a8cfac642e-Paper-Conference.pdf)
[13](https://arxiv.org/html/2312.09527v1)
[14](https://arxiv.org/abs/2403.11865)
[15](https://www.sciencedirect.com/science/article/abs/pii/S095741742402935X)
[16](https://arxiv.org/abs/2305.04268)
[17](https://arxiv.org/html/2403.12839v2)
[18](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_LowRankOcc_Tensor_Decomposition_and_Low-Rank_Recovery_for_Vision-based_3D_Semantic_CVPR_2024_paper.pdf)
[19](https://dl.acm.org/doi/abs/10.1007/s10489-024-05476-0)
[20](http://www.dpi-journals.com/index.php/dtssehs/article/view/27372)
[21](https://www.semanticscholar.org/paper/d964a74722b87dbcc5e4ecd843e62be39b78eb85)
[22](https://www.semanticscholar.org/paper/eba6b0d9caef19aad9c476d114ad311c6ade9ca7)
[23](https://www.semanticscholar.org/paper/2c9da07d55729addab5c3682042de0e34bcea79a)
[24](https://ems.press/doi/10.4171/owr/2018/46)
[25](https://www.semanticscholar.org/paper/06325c8d7b81fc955ddb55a8e4fe90a79c1b8343)
[26](https://www.semanticscholar.org/paper/1364123ffc9d4421532ee19692ef6b9039530221)
[27](https://teseo.unitn.it/xy-rivista/article/view/2116)
[28](https://arxiv.org/abs/2203.09517)
[29](https://arxiv.org/pdf/2205.06407.pdf)
[30](http://arxiv.org/pdf/2402.16638.pdf)
[31](http://arxiv.org/pdf/1910.09499.pdf)
[32](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920332.pdf)
[33](https://diglib.eg.org/bitstream/handle/10.1111/cgf15062/v43i2star_03_15062.pdf)
[34](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GP-NeRF_Generalized_Perception_NeRF_for_Context-Aware_3D_Scene_Understanding_CVPR_2024_paper.pdf)
[35](https://ar5iv.labs.arxiv.org/html/2203.09517)
[36](https://arxiv.org/abs/1807.10027)
[37](https://arxiv.org/abs/2304.11842)
[38](https://www.nature.com/articles/s40494-025-01695-x)
[39](https://arxiv.org/abs/2306.06359)
[40](https://ieeexplore.ieee.org/document/10205429/)
[41](https://link.springer.com/10.1007/s11390-024-4157-6)
[42](https://www.semanticscholar.org/paper/ebbdffb6dde100f70d8f5f3f67b8baffb467ea2b)
[43](https://ieeexplore.ieee.org/document/10204079/)
[44](https://arxiv.org/abs/2406.15707)
[45](http://arxiv.org/pdf/2304.11842.pdf)
[46](https://arxiv.org/html/2402.04632)
[47](https://arxiv.org/pdf/2208.04717.pdf)
[48](https://arxiv.org/html/2501.02807v1)
[49](https://arxiv.org/abs/2109.07448)
[50](http://arxiv.org/pdf/2309.05028.pdf)
[51](https://openaccess.thecvf.com/content/CVPR2024/papers/Min_Entangled_View-Epipolar_Information_Aggregation_for_Generalizable_Neural_Radiance_Fields_CVPR_2024_paper.pdf)
[52](https://vclab.kaist.ac.kr/cvpr2023p3/2831_progressively_optimized_local_-Camera-ready%20PDF.pdf)
[53](https://www.sciencedirect.com/science/article/abs/pii/S0926580524000785)
[54](https://dl.acm.org/doi/10.1145/3581783.3612246)
