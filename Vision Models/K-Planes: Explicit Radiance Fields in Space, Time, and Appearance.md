# K-Planes: Explicit Radiance Fields in Space, Time, and Appearance

### 핵심 주장과 주요 기여[1]

K-Planes는 **임의의 차원(arbitrary dimensions)에서 방사성 장(radiance fields)을 표현하는 화이트박스 모델**로 제안됩니다. 이 모델의 핵심 기여는 다음과 같습니다.

#### 1. 평면 인수분해(Planar Factorization) 전략[1]

K-Planes는 $$d$$-차원 장면을 $$\binom{d}{2}$$개의 2차원 평면으로 분해합니다. 구체적으로:

- **정적 3D 장면(d=3)**: 3개의 평면(**트라이플레인**)으로 표현 - xy, xz, yz 평면
- **동적 4D 장면(d=4)**: 6개의 평면(**헥스플레인**)으로 표현 - 공간 평면 3개(xy, xz, yz) + 시공간 평면 3개(xt, yt, zt)

이 설계는 **메모리 효율성**과 **해석 가능성**이라는 두 가지 장점을 제공합니다.

#### 2. 곱셈 기반 특징 결합[1]

K-Planes의 중요한 개선 사항은 **Hadamard 곱(elementwise multiplication)**을 통해 평면들의 특징을 결합한다는 점입니다. 4D 점 $$\mathbf{q}=(i,j,k,\tau)$$의 특징은 다음과 같이 계산됩니다:

$$
f(\mathbf{q}) = \prod_{c \in C} f(\mathbf{q})_c
$$

여기서 $$f(\mathbf{q})_c$$는 각 평면 $$c$$에 대한 이중선형 보간(bilinear interpolation) 결과입니다. **덧셈 대신 곱셈을 사용하는 이유**는 공간적으로 국소화된 신호를 생성하는 능력에 있습니다. Table 2에서 곱셈은 명시적 모델에서 PSNR 35.29 대비 덧셈은 28.78로 큰 성능 차이를 보입니다.[1]

#### 3. 화이트박스 설계[1]

K-Planes는 **해석 가능한 모델**을 추구하며, 이는 두 가지 선택으로 구현됩니다:

1. **선형 특징 디코더**: 구면 조화 함수(SH) 대신 **학습된 색상 기저(learned color basis)**를 사용하여 뷰 의존 색상을 모델링합니다. MLP가 뷰-의존 색상과 공간 구조 결정을 모두 수행하는 것을 완화합니다.

2. **명시적 정적-동적 분해**: 공간 전용 평면과 시공간 평면의 분리로 인해 동적 영역이 명확하게 시각화됩니다.

### 해결하는 문제 및 제안 방법[1]

#### 문제 정의

기존 접근 방식의 한계:

- **NeRF**: 느린 최적화(수 시간 ~ 수일), 블랙박스 MLP 사용
- **Plenoxels, DVGO**: 3D 그리드는 차원 증가에 따라 지수적으로 메모리 증가
- **Tensor4D**: 9개 평면 사용으로 중복성 존재 (yt 평면 2개)

#### 제안 방법의 수식

**투영(Projection)**:

$$
f(\mathbf{q})_c = \psi\left(\mathbf{P}_c, \pi_c(\mathbf{q})\right) \quad \text{(식 1)}
$$

**특징 결합**:

$$
f(\mathbf{q}) = \prod_{c \in C} f(\mathbf{q})_c \quad \text{(식 2)}
$$

**색상 디코딩**:

$$
\mathbf{c}(\mathbf{q}, \mathbf{d}) = \bigcup_{i \in \{R,G,B\}} f(\mathbf{q}) \cdot \mathbf{b}_i(\mathbf{d}) \quad \text{(식 6)}
$$

**밀도 디코딩**:

$$
\sigma(\mathbf{q}) = f(\mathbf{q}) \cdot \mathbf{b}_{\sigma} \quad \text{(식 7)}
$$

### 정규화 항들[1]

모델은 다음 정규화항들로 학습됩니다:

**공간 전체변분(Total Variation)**:

$$
\mathcal{L}_{TV}(\mathbf{P}) = \frac{1}{|C|n^2} \sum_{c,i,j} \left(\left\|\mathbf{P}_c^{i,j} - \mathbf{P}_c^{i-1,j}\right\|_2^2 + \left\|\mathbf{P}_c^{i,j} - \mathbf{P}_c^{i,j-1}\right\|_2^2\right) \quad \text{(식 3)}
$$

**시간 평활성(Temporal Smoothness)**:

$$
\mathcal{L}_{smooth}(\mathbf{P}) = \frac{1}{|C|n^2} \sum_{c,i,t} \left\|\mathbf{P}_c^{i,t-1} - 2\mathbf{P}_c^{i,t} + \mathbf{P}_c^{i,t+1}\right\|_2^2 \quad \text{(식 4)}
$$

**희소 일시적 변화(Sparse Transients)**:

$$
\mathcal{L}_{sep}(\mathbf{P}) = \sum_c \left\|\mathbf{1} - \mathbf{P}_c\right\|_1, \quad c \in \{xt, yt, zt\} \quad \text{(식 5)}
$$

### 모델 구조[1]

K-Planes는 **다중 스케일 평면(Multiscale Planes)** 구조를 채택합니다:

- 공간 해상도: 64, 128, 256, 512 (실험에서 사용)
- 각 스케일의 특징 길이: M=32 (기본값)
- 서로 다른 스케일의 M-차원 특징 벡터는 **연결(concatenation)**되어 디코더에 전달됩니다.

#### 두 가지 디코더 버전

1. **명시적 버전**: 선형 디코더 + 학습된 색상 기저 (MLP 기반 기저)
2. **하이브리드 버전**: 두 개의 작은 MLP (밀도용 $$g_{\sigma}$$, 색상용 $$g_{RGB}$$)

### 성능 향상[1]

#### 정량적 결과 (Table 3)

| 데이터셋 | 메트릭 | K-Planes (명시) | K-Planes (하이브리드) | 최고 기존 방법 |
|---------|---------|-----------------|----------------|---------|
| NeRF (합성, 정적) | PSNR | 32.21 | 32.36 | TensoRF: 33.14 |
| LLFF (실제, 정적) | PSNR | 26.78 | 26.92 | TensoRF: 26.73 |
| D-NeRF (합성, 동적) | PSNR | 31.05 | 31.61 | V4D: 33.72 |
| DyNeRF (실제, 동적) | PSNR | 30.88 | 31.63 | Mix Voxels: 30.80 |

#### 주요 성과

1. **메모리 효율성**: 4D 그리드 대비 **1000배 압축** (300GB → 200MB)[1]
2. **최적화 속도**: DyNeRF 대비 **약 370배 빠름** (1344시간 → 3.7시간)[1]
3. **해석 가능성**: 동적-정적 성분 자동 분해 가능[1]

### 모델의 한계[1]

1. **성능 격차**: 일부 데이터셋에서 V4D, TiNeuVox 등 최신 방법에 비해 낮은 성능
2. **Phototourism 성능**: NeRF-W (PSNR 27.00) 대비 K-Planes (PSNR 22.92) 성능 차이[1]
3. **고주파 세부사항**: 명시적 선형 디코더는 복잡한 반사 특성 모델링에 제한
4. **모노큘러 동영상 한계**: 중요 샘플링(importance sampling)을 사용할 수 없음[1]

### 모델의 일반화 성능[1]

K-Planes의 일반화 성능과 관련된 핵심 특징:

#### 1. 장면 특정 최적화(Scene-Specific Optimization)

K-Planes는 다른 대부분의 방사성 장 방법처럼 **각 장면별로 개별 최적화**를 수행합니다. 이는:
- 각 장면마다 새로운 모델 매개변수를 학습해야 함
- 기존 학습 없이 전이 학습(transfer learning) 불가능

#### 2. 일반화 성능 개선 가능성

논문은 직접적으로 다루지 않지만, **구조적 이점**으로 인한 개선 가능성이 있습니다:

**다중 스케일 표현**: 64, 128, 256, 512 해상도의 계층적 구조는 **저주파에서 고주파까지 점진적 학습**을 가능하게 하여, 적은 관찰 데이터에서도 견고한 기하학 학습을 촉진합니다.

**명시적 정규화 제약**: 시간 평활성, 공간 TV 정규화, 희소 일시적 변화 제약은 **과적합 방지**에 도움이 됩니다.

### 앞으로의 연구 영향과 고려사항

#### 1. 현재 연구 트렌드[2][3][4][5][6][7]

**일반화 가능한 NeRF(Generalizable NeRF)**: 2024-2025년 연구는 **모든 장면에 적용되는 단일 모델** 개발에 집중합니다:

- **MRVM-NeRF** (2024): 마스크 기반 모델링으로 장면 간 일반화 개선
- **GSNeRF** (2024): 의미론적 정보와 함께 일반화
- **ID-NeRF** (2025): 사전 훈련된 확산 모델 기반 가이던스로 제한된 데이터 환경에서 일반화 향상

#### 2. 소수 샷 학습(Few-Shot Learning) 분야[3][4][5][6][2]

K-Planes의 **계층적 다중 스케일 구조**는 소수 샷 설정에 이상적입니다:

- **DWTNeRF** (2025): Instant-NGP 기반으로 해시 인코딩과 이산 웨이블릿 손실 결합 (3샷 LLFF에서 PSNR 15.07% 개선)
- **SANeRF** (2024): 공간 어닐링으로 다중 스케일 표현 최적화
- **FrugalNeRF** (2024): 가중치 공유 복셀과 교차 스케일 기하학 적응

이러한 방법들은 K-Planes의 다중 스케일 개념을 활용하여 **제한된 관찰 데이터에서 수렴 가속화**를 달성합니다.

#### 3. 명시적 표현의 부상[8][9]

**RefinedFields** (2024): K-Planes를 사전 훈련된 모델로 정제하여 **약한 감독(weakly supervised)** 환경에서 성능 향상

**TK-Planes** (2024): K-Planes를 동적 UAV 장면으로 확장하여 고도가 높은 비디오 캡처에서 동적 객체 추적 개선

**X-NeRF** (2023): 다중 장면 360° 부족 뷰(insufficient view) 문제에 명시적 완성 매핑 적용

#### 4. 고려할 연구 방향

**크로스 스케일 기하학 적응**: K-Planes의 다중 스케일 특성을 활용하여 각 스케일에서 자동으로 주파수 조정[10]

**사전 훈련 및 미세조정**: 대규모 장면 데이터셋에서 하이퍼네트워크로 일반 맵 학습 후 빠른 미세조정[11]

**깊이 정규화**: 단안 깊이 추정 모델의 사전 지식을 활용한 규칙화로 희소 뷰 성능 개선[12][13]

**상호정보 이론**: K-Planes의 다중 평면 구조를 통해 여러 뷰 간 상호정보 최대화로 일반화 강화[14]

### 결론

K-Planes는 **간단하고 효율적이면서도 해석 가능한** 평면 인수분해 접근법으로 3D 정적 장면, 4D 동적 장면, 변하는 외형의 장면을 모두 다룰 수 있습니다. 특히 **Hadamard 곱을 통한 공간적 국소화**와 **다중 스케일 표현**은 이후 연구에서 소수 샷 학습과 일반화 개선의 기초가 되었습니다.

다만 K-Planes는 장면별 최적화를 요구하므로, **앞으로의 연구 방향**은 다음을 포함합니다:

1. **크로스 장면 일반화**: 사전 훈련된 인코더나 하이퍼네트워크를 통한 장면 간 지식 전이
2. **약한 감독 학습**: 사전 훈련된 확산 모델이나 깊이 추정기를 통한 감독 신호 강화
3. **효율적 미세조정**: 사전 훈련된 K-Planes 매개변수에서 빠른 적응
4. **다중 모달 입력**: RGB-D 입력이나 의미론적 정보 결합

이러한 방향들은 K-Planes의 명시적 표현과 해석 가능성의 이점을 유지하면서, 현실 세계의 제한된 데이터 환경에서 더 나은 성능을 달성하는 것을 목표로 합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ecb80336-6035-49c2-aebb-c2154d89eb7a/2301.10241v2.pdf)
[2](https://arxiv.org/html/2501.12637v2)
[3](https://arxiv.org/abs/2301.10941)
[4](https://arxiv.org/html/2402.14586v1)
[5](http://arxiv.org/pdf/2404.00992.pdf)
[6](http://arxiv.org/pdf/2406.07828.pdf)
[7](https://arxiv.org/html/2408.04803v1)
[8](https://arxiv.org/html/2312.00639v3)
[9](https://openaccess.thecvf.com/content/WACV2023/papers/Zhu_X-NeRF_Explicit_Neural_Radiance_Field_for_Multi-Scene_360deg_Insufficient_RGB-D_WACV_2023_paper.pdf)
[10](https://linjohnss.github.io/frugalnerf/)
[11](https://arxiv.org/html/2310.17075)
[12](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Li_DNGaussian_Optimizing_Sparse-View_CVPR_2024_supplemental.pdf)
[13](https://arxiv.org/html/2403.06912v1)
[14](https://openreview.net/forum?id=5RPpwW82vs)
[15](https://arxiv.org/pdf/2301.10241.pdf)
[16](http://arxiv.org/pdf/2311.18159.pdf)
[17](http://arxiv.org/pdf/2405.02762.pdf)
[18](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2023MS003932)
[19](https://arxiv.org/html/2405.07857v3)
[20](https://arxiv.org/html/2407.13185)
[21](http://arxiv.org/pdf/2112.01523.pdf)
[22](https://openaccess.thecvf.com/content/CVPR2023/papers/Fridovich-Keil_K-Planes_Explicit_Radiance_Fields_in_Space_Time_and_Appearance_CVPR_2023_paper.pdf)
[23](https://openaccess.thecvf.com/content/CVPR2024/papers/Chou_GSNeRF_Generalizable_Semantic_Neural_Radiance_Fields_with_Enhanced_3D_Scene_CVPR_2024_paper.pdf)
[24](https://openaccess.thecvf.com/content/WACV2021/papers/Bautista_On_the_Generalization_of_Learning-Based_3D_Reconstruction_WACV_2021_paper.pdf)
[25](https://arxiv.org/abs/2301.10241)
[26](https://www.sciencedirect.com/science/article/abs/pii/S095741742402935X)
[27](https://arxiv.org/html/2404.03421v1)
[28](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/k-planes/)
[29](https://arxiv.org/html/2210.00379v6)
[30](https://oasisyang.github.io/neural-prior/)
[31](https://jseobyun.tistory.com/361)
[32](http://arxiv.org/pdf/2212.02280.pdf)
[33](https://eccv.ecva.net/virtual/2024/poster/2027)
[34](https://proceedings.iclr.cc/paper_files/paper/2024/file/8882d370cdafec9885b918a8cfac642e-Paper-Conference.pdf)
[35](https://arxiv.org/html/2501.12637v1)
[36](https://proceedings.mlr.press/v202/fu23g/fu23g.pdf)
[37](https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZeroRF_Fast_Sparse_View_360deg_Reconstruction_with_Zero_Pretraining_CVPR_2024_paper.pdf)
