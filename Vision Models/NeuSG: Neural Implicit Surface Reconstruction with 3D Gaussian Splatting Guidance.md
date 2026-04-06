# NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

NeuSG는 **신경 암묵적 표면 재구성(Neural Implicit Surface Reconstruction)**과 **3D 가우시안 스플래팅(3D Gaussian Splatting, 3DGS)**을 **상호 보완적으로 공동 최적화(joint optimization)**함으로써, 기존 방법들이 가졌던 **과도한 표면 평활화(over-smoothing)** 문제와 **세부 디테일 손실** 문제를 동시에 해결할 수 있다고 주장합니다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **공동 최적화 프레임워크** | NeuS와 3DGS를 동시에 최적화하는 파이프라인 제안 |
| **스케일 정규화 (Scale Regularization)** | 3D 가우시안을 극도로 납작하게 만들어 중심점을 표면에 근접시킴 |
| **법선 정규화 (Normal Regularization)** | NeuS가 예측한 법선을 이용해 가우시안 방향을 정렬, 노이즈 감소 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

기존 신경 암묵적 표면 재구성 방법들의 한계:

1. **텍스처가 없는 영역(textureless regions)** 에서 표면 재구성 실패
2. **MVS(다중 뷰 스테레오)** 기반 점군은 균일 분포되어 있어 **세부 영역에서 불완전**
3. 깊이 맵(depth map)이나 점군을 사전 정보로 사용해도 **과도하게 매끄러운(over-smoothed) 표면** 생성
4. 기존 3DGS의 가우시안 중심점은 **표면 위에 위치하지 않고 내부에 존재**하여 직접 활용 불가

### 2.2 제안하는 방법 (수식 포함)

#### (a) NeuS 기반 신경 암묵적 표면 재구성

NeuS는 SDF(Signed Distance Function)를 기반으로 볼륨 렌더링을 수행합니다.

SDF를 불투명도(opacity)로 변환:

$$\alpha_i = \max\left(\frac{\Phi_s(f(\mathbf{x}_i)) - \Phi_s(f(\mathbf{x}_{i+1}))}{\Phi_s(f(\mathbf{x}_i))}, 0\right) $$

여기서 $\Phi_s$는 시그모이드 함수, $f(\mathbf{x}_i)$는 점 $\mathbf{x}_i$에서의 SDF 값입니다.

픽셀의 색상 렌더링:

$$\hat{\mathbf{C}}(\mathbf{o}, \mathbf{d}) = \sum_{i=1}^{N} w_i \mathbf{c}_i, \quad \text{where } w_i = T_i \alpha_i $$

$T_i$는 누적 투과율(transmittance), $N$은 레이 위의 샘플 수입니다.

#### (b) 3D 가우시안 스플래팅

각 3D 가우시안은 공분산 행렬 $\mathbf{\Sigma}$와 중심점 $\mathbf{p}$로 정의:

```math
G(\mathbf{x}) = \exp\left\{-\frac{1}{2}(\mathbf{x} - \mathbf{p})^\top \mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{p})\right\}
```

공분산 행렬은 스케일 행렬 $\mathbf{S}$와 회전 행렬 $\mathbf{R}$의 곱:

$$\mathbf{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top $$

2D 투영을 위한 변환:

$$\mathbf{\Sigma}' = \mathbf{J}\mathbf{W}\mathbf{\Sigma}\mathbf{W}^\top\mathbf{J}^\top $$

색상 블렌딩:

$$\mathbf{C} = \sum_{i \in N} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j) $$

#### (c) 스케일 정규화 (Scale Regularization)

가우시안 타원체를 극도로 납작하게 만들어 중심점을 표면에 접근시킴:

$$\mathcal{L}_s = \|\min(s_1, s_2, s_3)\|_1 $$

스케일 팩터 $\mathbf{s} = (s_1, s_2, s_3)^\top \in \mathbb{R}^3$에서 가장 작은 성분을 0으로 최소화합니다.

#### (d) 법선 정규화 (Normal Regularization)

카메라 좌표계에서 법선 정의:

$$\mathbf{n}_c = \text{OneHot}\left(\arg\min(s_1, s_2, s_3)\right) \in \mathbb{R}^3 $$

월드 좌표계로 변환:

$$\mathbf{n}_w = \mathbf{R} \times \mathbf{n}_c $$

NeuS가 예측한 표면 법선 $\nabla f(\mathbf{p}_i)$와 정렬:

$$\mathcal{L}_\text{align} = \left\|1 - \left|\mathbf{n}_w^\top \cdot \nabla f(\mathbf{p}_i)\right|\right\|_1 $$

절댓값을 사용해 방향 무관하게 정렬합니다.

#### (e) 공동 최적화 손실 함수

**NeuS 학습을 위한 전체 손실:**

색상 손실:

$$\mathcal{L}_\text{RGB} = \|\mathbf{C} - \hat{\mathbf{C}}\|_1 $$

에이코날 정규화(Eikonal regularization):

$$\mathcal{L}_\text{eik} = \frac{1}{N}\sum_{i=1}^{N}\left(\|\nabla f(\mathbf{x}_i)\|_2 - 1\right)^2 $$

점군 SDF 제약:

$$\mathcal{L}_\text{pt} = |f(\mathbf{p}_i)|_1 $$

NeuS 전체 손실:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{RGB} + \lambda_1 \mathcal{L}_\text{eik} + \lambda_2 \mathcal{L}_\text{pt} $$

**3DGS 학습을 위한 전체 손실:**

$$\mathcal{L}_\text{Gaussian} = \mathcal{L}_\text{RGB} + \lambda_3 \mathcal{L}_s + \lambda_4 \mathcal{L}_\text{align} $$

논문에서 사용된 하이퍼파라미터: $\lambda_1=0.1,\ \lambda_2=1,\ \lambda_3=100,\ \lambda_4=1$

### 2.3 모델 구조

```
입력: 다중 뷰 이미지 (263~1,107장)
         │
         ▼
┌─────────────────────────────────────────┐
│           NeuSG Framework               │
│                                         │
│  ┌─────────────┐    ┌────────────────┐  │
│  │   NeuS      │◄──►│ 3D Gaussian    │  │
│  │  (Hash Enc) │    │  Splatting     │  │
│  │  SDF f(x)   │    │  {p, n}        │  │
│  │  ∇f(x)      │───►│  Scale Reg.    │  │
│  └──────┬──────┘    └───────┬────────┘  │
│         │                   │           │
│    L_total              L_Gaussian      │
│    (Eq.14)              (Eq.15)         │
└─────────────────────────────────────────┘
         │
         ▼
출력: 완전하고 세밀한 3D 표면 메시
```

- **NeuS 부분**: Hash Encoding (InstantNGP 스타일, $2^{19}$ 해시 엔트리) + MLP
- **NeuralAngelo** 기법(multi-scale, 수치 미분) 차용
- **NeRF++** 방식의 듀얼 네트워크(외부 환경용 NeRF + 내부 형상용 NeuS)
- 500k 이터레이션(NeuS), 매 100k마다 30k 이터레이션(3DGS)
- 하드웨어: RTX 4090 24GB, 약 16시간 학습

---

## 3. 일반화 성능 향상 가능성

### 3.1 일반화에 유리한 요소

#### (1) 밀도 높은 점군으로 사전 정보 품질 향상
기존 MVS 기반 점군은 균일 분포 + 불완전한 특성을 가지지만, NeuSG의 3DGS 기반 점군은 **임의 구조의 장면에서도 세밀한 기하학적 구조를 밀집하게 포착**합니다. 이는 다양한 도메인의 장면에 적용 시 더 강건한 사전 정보를 제공합니다.

#### (2) 상호 정제(Mutual Refinement)의 자기 교정 효과
NeuS의 법선 예측 → 3DGS 정제 → 더 정확한 점군 → NeuS 향상이라는 **긍정적 피드백 루프**는 특정 장면에 과도하게 의존하지 않고 **자기 교정적(self-correcting)** 특성을 가집니다.

#### (3) 해시 인코딩의 스케일 적응성
InstantNGP 방식의 멀티 해상도 해시 인코딩은 **다양한 스케일의 장면**에서 적응적으로 표현이 가능하여, 대형 야외 장면(Tanks and Temples)뿐 아니라 소규모 실내 장면에도 확장 가능합니다.

#### (4) Vis-MVSNet 점군 보완
3DGS 점군과 Vis-MVSNet 점군을 함께 사용하는 **보완적 점군 전략**은 어느 한 방법이 실패하는 영역을 다른 방법이 보완하여 전반적 강건성을 높입니다.

### 3.2 일반화를 제한하는 요소 (한계점과 연결)

| 제한 요소 | 설명 |
|-----------|------|
| **다중 뷰 의존성** | 충분한 뷰 커버리지 없이는 성능 저하 |
| **희소/불균일 이미지 분포** | 관측이 적은 영역에서 재구성 품질 저하 |
| **단일 장면 최적화** | 장면별 개별 최적화로 새로운 장면에 즉각 적용 불가(zero-shot 불가) |
| **동적 장면 미지원** | 정적 장면 가정 |
| **계산 비용** | 16시간/장면으로 실시간 적용 어려움 |

### 3.3 일반화 성능 향상을 위한 잠재적 방향

1. **Few-shot 학습 결합**: RegNeRF, DietNeRF 등의 희소 뷰 정규화 기법과 결합 시 적은 이미지로도 일반화 가능
2. **사전 학습된 기하 선험 활용**: Monocular depth/normal estimation 모델(예: OmniData, Depth Anything)을 외부 사전으로 추가 활용
3. **Generalizable NeRF와의 통합**: pixelNeRF, IBRNet 등 cross-scene generalization 모델과의 통합으로 새로운 장면에 대한 즉각 적용 가능성 확대

---

## 4. 성능 향상 및 한계

### 4.1 정량적 성능 (Tanks and Temples, F1 Score)

| 방법 | Barn | Caterpillar | Courthouse | Ignatius | Meetingroom | Truck | **Mean** | GPU 시간 |
|------|------|-------------|------------|----------|-------------|-------|---------|---------|
| NeuS | 0.29 | 0.29 | 0.17 | 0.83 | 0.24 | 0.45 | 0.38 | - |
| MonoSDF | 0.49 | 0.31 | 0.12 | 0.78 | 0.23 | 0.42 | 0.39 | 18h |
| NAngelo-19 | 0.61 | 0.34 | 0.13 | 0.82 | 0.22 | 0.45 | 0.43 | 15h |
| NAngelo-22 | **0.70** | **0.36** | **0.28** | **0.89** | **0.32** | **0.48** | **0.50** | 128h |
| **NeuSG** | **0.73** | **0.37** | 0.22 | 0.83 | **0.35** | 0.46 | **0.49** | **16h** |

- NAngelo-22 대비 **동등한 성능을 1/8의 시간**으로 달성
- MVS 기반 점군만 사용 시 F1=0.63 → 3DGS 점군(Scale+Normal 정규화) 추가 시 F1=0.73 (+0.10)

### 4.2 에블레이션 스터디 결과

| MVS | Original 3DGS | Scale Reg | Normal Reg | F1 Score |
|-----|---------------|-----------|------------|----------|
| ✗ | ✗ | ✗ | ✗ | 0.61 |
| ✓ | ✗ | ✗ | ✗ | 0.63 (+0.02) |
| ✓ | ✓ | ✗ | ✗ | 0.59 (-0.02) |
| ✗ | ✗ | ✓ | ✗ | 0.69 (+0.08) |
| ✗ | ✗ | ✓ | ✓ | **0.73 (+0.12)** |

> 핵심 발견: 정규화 없이 원본 3DGS 점군만 사용하면 **오히려 성능이 저하**(-0.02)되며, 스케일+법선 정규화가 모두 필요함.

### 4.3 한계점

1. **다중 뷰 데이터 의존성**: 충분한 뷰 수 확보 필요
2. **희소/불균일 이미지 분포 취약**: 특정 영역 미관측 시 재구성 품질 저하
3. **동적 객체 처리 불가**: 정적 장면 가정
4. **단일 장면 최적화**: 장면 간 일반화 없음
5. **Courthouse 장면에서 NAngelo-22 대비 낮은 성능** (0.22 vs 0.28)

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 방법론 계보

```
NeRF (2020) ──► NeuS (2021) ──► NeuralAngelo (2023)
                    │                    │
              Geo-NeuS (2022)       NeuSG (2023/2025)
              MonoSDF (2022)
              RegSDF (2022)
              
3DGS (2023) ──► NeuSG (2023/2025)
```

### 5.2 상세 비교

| 방법 | 연도 | 기하 사전 | 세부 표현 | 속도 | 주요 특징 |
|------|------|-----------|-----------|------|-----------|
| **NeRF** [28] | 2020 | 없음(밀도 필드) | 낮음 | 느림 | 뷰 합성 최초 신경 렌더링 |
| **NeuS** [47] | 2021 | SDF | 중간 | 느림 | SDF 기반 볼륨 렌더링 |
| **InstantNGP** [29] | 2022 | 해시 인코딩 | 중간 | 빠름 | 멀티해상도 해시 그리드 |
| **Geo-NeuS** [9] | 2022 | MVS 희소 점군 | 중간 | 중간 | 점군으로 SDF 감독 |
| **MonoSDF** [57] | 2022 | 단안 깊이/법선 | 완전하나 평활 | 중간 | 단안 기하 단서 활용 |
| **RegSDF** [60] | 2022 | SfM 희소 점군 | 중간 | 중간 | 야생 환경 정규화 |
| **NeuralAngelo** [23] | 2023 | 해시 + 고차 미분 | 높음 | 매우 느림 | 고품질, 고비용(128h) |
| **3DGS** [18] | 2023 | 명시적 가우시안 | 렌더링 특화 | 매우 빠름 | 실시간 렌더링 |
| **NeuSG** | 2023/2025 | 3DGS 점군 | **높음** | **효율적(16h)** | NeuS + 3DGS 공동 최적화 |

### 5.3 NeuSG의 차별점

- **Geo-NeuS/RegSDF 대비**: MVS 점군 대신 3DGS 점군 사용으로 밀집하고 세밀한 기하 구조 포착 → F1 0.35 vs **0.49**
- **MonoSDF 대비**: 외부 단안 추정기 의존 없이 내부 NeuS 법선 활용 → 과도한 평활화 방지
- **NeuralAngelo-22 대비**: 동등한 F1(0.49 vs 0.50)을 **1/8 시간**으로 달성
- **순수 3DGS 대비**: 표면 재구성을 위한 메시 추출 가능, SDF 기반의 명확한 표면 정의

---

## 6. 앞으로의 연구에 미치는 영향과 고려 사항

### 6.1 앞으로의 연구에 미치는 영향

#### (1) 하이브리드 표현의 새로운 패러다임 제시
NeuSG는 **암묵적(implicit) 표현과 명시적(explicit) 표현의 상호 보완**이 효과적임을 증명했습니다. 이는 두 표현 방식을 단순히 병렬로 사용하는 것이 아니라, **서로의 약점을 보완하는 공동 최적화**가 핵심임을 보여줍니다. 향후 연구들이 이러한 하이브리드 접근을 더욱 발전시키는 계기가 될 것입니다.

#### (2) 3DGS의 기하 재구성 활용 가능성 확장
기존 3DGS는 주로 **렌더링 품질**에 초점이 맞춰져 있었으나, NeuSG는 3DGS를 **기하 재구성 사전**으로 활용할 수 있음을 보여주었습니다. 이는 3DGS 기반 SLAM, 3D 편집, 의료 영상 재구성 등 다양한 응용 분야에 영향을 미칠 것입니다.

#### (3) 점군 정규화의 새로운 기준 제시
스케일 정규화 + 법선 정규화 조합은 **노이즈 있는 점군을 자동으로 정제**하는 새로운 접근으로, 향후 점군 기반 정규화 연구의 기준점이 될 수 있습니다.

#### (4) 계산 효율성과 품질의 균형 지점 탐색
NAngelo-22의 1/8 시간으로 동등한 성능 달성은 **해시 인코딩 크기, 최적화 스케줄, 보조 표현의 활용** 간의 트레이드오프 탐색에 중요한 데이터 포인트를 제공합니다.

### 6.2 앞으로 연구 시 고려할 점

#### (1) 희소 뷰 일반화 (Few-shot Generalization)
```
현재 한계: 충분한 다중 뷰 필요
연구 방향: 
  - DUSt3R, MASt3R 등 기반 모델과 결합
  - 단안 기하 추정기(Depth Anything V2 등)와 통합
  - 크로스씬 학습을 통한 zero-shot 재구성
```

#### (2) 동적 장면 확장
3DGS는 이미 동적 장면으로의 확장 연구(Dynamic 3DGS 등)가 진행 중이며, NeuSG 프레임워크도 **시간 축을 포함한 4D 재구성**으로 확장 가능성이 있습니다.

#### (3) 의미론적 정보 통합
현재 NeuSG는 순수 기하 정보만을 다루지만, **의미론적 분할(semantic segmentation)** 정보를 추가로 활용하면 텍스처 없는 영역에서의 재구성 성능을 더욱 향상시킬 수 있습니다.

#### (4) 3DGS 초기화 품질의 중요성
논문에서 3DGS 점군이 없을 때(-0.02) 성능이 저하됨을 보여주었는데, 이는 **3DGS 초기화 품질이 전체 성능에 결정적 영향**을 미친다는 점을 시사합니다. 향후 연구에서는 더 나은 3DGS 초기화 전략이 필요합니다.

#### (5) 대용량 장면 확장성
현재 단일 GPU 16시간의 학습 비용은 도시 규모의 대형 장면에는 여전히 부담이 됩니다. **Block-NeRF, Mega-NeRF** 등 분산 처리 전략과의 결합이 필요합니다.

#### (6) 손실 함수 가중치 민감도 분석
$\lambda_1, \lambda_2, \lambda_3, \lambda_4$에 대한 **체계적인 민감도 분석(sensitivity analysis)**이 부족하며, 장면 유형에 따른 자동 가중치 조정 메커니즘이 필요합니다.

#### (7) 다른 벤치마크로의 검증 확대
현재 Tanks and Temples에만 집중되어 있어 **DTU, BlendedMVS, Replica** 등 다양한 벤치마크에서의 검증이 필요하며, 특히 실내 소규모 장면과 반사/투명 물체에 대한 성능 평가가 필요합니다.

---

## 참고 자료

### 논문 원본
- **Chen, H., Li, C., Wang, Y., & Lee, G. H. (2023/2025). "NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance." arXiv:2312.00846v2**

### 논문 내 참조 문헌 (주요)
- [47] Wang et al., "NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction," arXiv:2106.10689, 2021
- [18] Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," ACM TOG, 2023
- [23] Li et al., "Neuralangelo: High-Fidelity Neural Surface Reconstruction," CVPR, 2023
- [29] Müller et al., "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding," ACM TOG, 2022
- [28] Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis," ECCV, 2020
- [57] Yu et al., "MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction," NeurIPS, 2022
- [9] Fu et al., "Geo-NeuS: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction," NeurIPS, 2022
- [60] Zhang et al., "Critical Regularizations for Neural Surface Reconstruction in the Wild," CVPR, 2022
- [20] Knapitsch et al., "Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction," ACM TOG, 2017
- [59] Zhang et al., "Visibility-Aware Multi-View Stereo Network," BMVC, 2020
- [11] Gropp et al., "Implicit Geometric Regularization for Learning Shapes," arXiv:2002.10099, 2020
