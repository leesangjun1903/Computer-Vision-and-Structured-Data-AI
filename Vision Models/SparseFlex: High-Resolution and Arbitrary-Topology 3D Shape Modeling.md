
# SparseFlex: High-Resolution and Arbitrary-Topology 3D Shape Modeling

> **논문 정보**
> - **제목**: SparseFlex: High-Resolution and Arbitrary-Topology 3D Shape Modeling
> - **저자**: Xianglong He, Zi-Xin Zou, Chia-Hao Chen, Yuan-Chen Guo, Ding Liang, Chun Yuan, Wanli Ouyang, Yan-Pei Cao, Yangguang Li
> - **소속**: Tsinghua University, VAST, The Chinese University of Hong Kong
> - **발표**: arXiv:2503.21732 (2025.03.27), **ICCV 2025** 채택 (pp. 14822–14833)
> - **프로젝트 페이지**: https://xianglonghe.github.io/TripoSF
> - **GitHub**: https://github.com/VAST-AI-Research/TripoSF

---

## 1. 핵심 주장 및 주요 기여 요약

고품질 3D 메시를 임의의 토폴로지(열린 표면 및 복잡한 내부 구조 포함)로 생성하는 것은 여전히 중요한 과제이며, 기존 암묵적 필드(implicit field) 방법들은 비용이 많이 들고 디테일이 손상되는 방수(watertight) 변환을 요구하거나, 고해상도에서의 처리에 어려움을 겪는다.

이에 대한 핵심 주장과 기여는 다음 세 가지로 요약됩니다:

### ① 새로운 표현 방식: SparseFlex

SparseFlex는 렌더링 손실(rendering loss)로부터 직접 $1024^3$ 해상도까지의 미분 가능한 메시 재구성을 가능하게 하는 새로운 희소 구조(sparse-structured) 등값면(isosurface) 표현 방식이다.

SparseFlex는 Flexicubes의 정확성과 희소 복셀(sparse voxel) 구조를 결합하여, 표면 인접 영역에만 연산을 집중하고 열린 표면(open surface)을 효율적으로 처리한다. 핵심 기여로, 렌더링 시 관련 복셀만을 활성화하는 **frustum-aware sectional voxel training** 전략을 도입하여 메모리 소비를 획기적으로 줄이고 고해상도 학습을 가능하게 한다.

### ② 최초의 렌더링 감독만을 이용한 메시 내부 재구성

이 방법은 **렌더링 감독(rendering supervision)만을 사용하여 메시 내부(interior)를 재구성하는 것을 처음으로 가능하게 한다.**

### ③ 완전한 생성 파이프라인 구축

이를 기반으로 VAE와 Rectified Flow Transformer를 훈련하여 고품질 3D 형상 생성 파이프라인을 시연하며, 실험 결과 이전 방법들 대비 Chamfer Distance ~82% 감소, F-score ~88% 증가라는 최첨단 재구성 정확도를 달성하였다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

이 논문은 임의의 토폴로지(열린 표면 및 복잡한 내부 구조 포함)를 가진 고충실도 3D 메시 생성의 도전을 다룬다. 기존 방법들은 암묵적 방법이 요구하는 방수(watertight) 변환에 의한 디테일 손실 또는 고해상도에서의 높은 메모리 소비에 어려움을 겪고 있다. SparseFlex는 이러한 한계를 극복하기 위한 새로운 3D 형상 표현 방식과 학습 전략을 도입한다.

구체적으로는 다음 세 가지 문제가 있었습니다:

1. **방수(watertight) 변환 문제**: SparseFlex는 Flexicubes의 미분 가능성을 계승하여, 렌더링 손실을 이용한 엔드-투-엔드(end-to-end) 최적화를 가능하게 하며, 이는 방수 메시 전처리의 필요성을 없애고 미세 디테일을 보존한다.
2. **고해상도 학습에서의 메모리 문제**: 희소성의 핵심 이유는 (1) 메모리 소비를 획기적으로 줄여 고해상도 모델링을 가능하게 하고, (2) 열린 경계 근처의 복셀을 효과적으로 제거하여 열린 표면의 표현을 가능하게 하는 것이다.
3. **열린 표면(open surface) 및 내부 구조 모델링 불가 문제**: UDF 기반의 형상 모델링은 SDF보다 더 어렵고, 신경망 경사 추정의 부정확성으로 인해 표면 추출이 불안정해지기 쉬우며, 결과적으로 고품질 열린 표면 메시의 달성이 어려운 상태이다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) SparseFlex 표현

SparseFlex는 Flexicubes를 기반으로 구축되어, 정확하고 미분 가능한 등값면 추출을 제공한다. 핵심 설계는 기존의 조밀한 그리드(dense grid) 대신 희소 복셀(sparse voxel) 구조를 사용하는 것이다.

SparseFlex에서 $\mathcal{F}_c$는 코너 그리드(corner grids)에서의 SDF 값과 변형(deformations)을 포함하고, $\mathcal{F}_v$는 각 복셀의 보간 가중치(interpolation weights)를 포함한다.

이를 수식으로 정리하면:

**SparseFlex 복셀 표현:**
$$\mathcal{V}_{\text{sparse}} = \{(k, \mathcal{F}_c^{(k)}, \mathcal{F}_v^{(k)}) \mid k \in \mathcal{S}\}$$

여기서 $\mathcal{S}$는 표면 인접 복셀의 인덱스 집합, $\mathcal{F}_c^{(k)}$는 SDF 값과 변형, $\mathcal{F}_v^{(k)}$는 보간 가중치.

**Flexicubes 방식의 메시 추출 (SDF 기반):**

각 복셀 코너에서의 SDF 값 $s_i$를 이용하여 등값면을 추출:

$$\mathcal{M} = \text{FlexiCubes}(\{s_i\}, \{\mathbf{d}_i\}, \{w_v\})$$

여기서 $\mathbf{d}_i$는 꼭짓점 위치의 변형량, $w_v$는 복셀별 보간 가중치.

희소 구조는 SDF의 연속적이고 변형 가능한 특성과 결합되어, 고품질 열린 표면 메시의 정확하고 효율적인 표현을 가능하게 한다.

---

#### (B) Frustum-Aware Sectional Voxel Training (핵심 전략)

SparseFlex의 역량을 최대한 활용하기 위해 **frustum-aware sectional voxel training**을 제안하며, 실시간 렌더링 기술에서 영감을 받아, 각 학습 반복(iteration)에서 카메라의 시야 절두체(view frustum) 내에 있는 SparseFlex 복셀의 부분 집합만을 활성화한다.

**렌더링 손실(Rendering Loss):**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{render}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_{\text{flex}} \mathcal{L}_{\text{flex}}$$

여기서:
- $\mathcal{L}_{\text{render}}$: RGB 및 알파 렌더링 손실 (미분 가능 래스터화 기반)
- $\mathcal{L}_{\text{KL}}$: VAE의 KL 발산 정규화 항
- $\mathcal{L}\_{\text{flex}}$: $\mathcal{L}_{\text{flex}}$는 Flexicubes로부터의 정규화 항으로, 매끄러운 SDF 값을 장려하기 위한 것이다.

**기존 방식 vs. Frustum-Aware 방식의 메모리 비교:**

기존 조밀 그리드: $O(N^3)$ 복셀 필요

Frustum-Aware 방식: $O(N^2 \cdot d)$ (절두체 내 복셀만 활성화, $d \ll N$)

기존 메시 기반 렌더링 학습 전략(좌)은 렌더링 시 단 몇 개의 복셀만 필요함에도 메시 표면을 추출하기 위해 전체 조밀 그리드를 활성화해야 하는 반면, 제안 방식(우)은 관련 복셀을 적응적으로 활성화하고 렌더링 감독만으로 메시 내부까지 재구성할 수 있다.

---

#### (C) VAE 학습 목적 함수

VAE의 전체 손실 함수는 다음과 같이 구성됩니다:

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{render}}(\hat{\mathcal{M}}, \mathcal{M}_{\text{gt}}) + \lambda_{\text{KL}} D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) + \lambda_{\text{flex}} \mathcal{L}_{\text{flex}}$$

여기서:
- $\hat{\mathcal{M}}$: 재구성된 메시
- $\mathcal{M}_{\text{gt}}$: 정답 메시
- $\mathbf{z}$: 잠재 변수(latent variable)
- $q(\mathbf{z}|\mathbf{x})$: 인코더 분포
- $p(\mathbf{z})$: 표준 정규 사전 분포

---

### 2.3 모델 구조

**TripoSF VAE Architecture**는 다음과 같은 구조를 가진다:
- **입력**: 원본 기하 디테일을 보존하는 포인트 클라우드(Point Clouds)
- **인코더**: 효율적인 기하 인코딩을 위한 희소 트랜스포머(Sparse Transformer)
- **디코더**: 희소성을 유지하는 자가 가지치기 업샘플링(Self-Pruning Upsampling) 모듈
- **출력**: 메시 추출을 위한 고해상도 SparseFlex 파라미터

TripoSF VAE는 메시에서 샘플링된 포인트 클라우드를 입력으로 받아, 이를 복셀화하고 각 복셀에 특징을 집계한다. 희소 트랜스포머 인코더-디코더가 구조화된 특징을 더 compact한 잠재 공간으로 압축하고, 이어서 고해상도를 위한 자가 가지치기 업샘플링이 수행된다. 마지막으로 구조화된 특징이 선형 레이어를 통해 SparseFlex 표현으로 디코딩되며, frustum-aware section voxel training 전략을 사용해 렌더링 손실로 파이프라인 전체를 더욱 효율적으로 학습한다.

**생성 파이프라인 (Image-to-3D)**:

두 가지 주요 구성 요소로 이루어진다: **Structure Flow Model**과 **Structured Latent Flow Model**. 먼저 별도의 단순한 3D 완전 합성곱 구조 VAE를 사용하여 3D 형상을 나타내는 조밀 복셀을 저해상도($\frac{1}{4}$ 스케일) 공간으로 압축한다. 이어서 이미지 조건 특징이 DINOv2를 이용해 추출되고 교차 어텐션(cross-attention)을 통해 트랜스포머 모델에 주입되며, 이 저해상도 공간에서 Rectified Flow 모델이 학습된다.

---

### 2.4 성능 향상

실험 결과 이전 방법들 대비 Chamfer Distance **82% 감소**, F-score **88% 증가**로 최첨단 재구성 정확도를 달성하였다.

SparseFlex VAE는 복잡한 기하 구조(complex geometries), 열린 표면(open surfaces), 심지어 내부 구조(interior structures)에서도 최첨단 성능을 발휘하며, 임의의 토폴로지를 가진 고품질 image-to-3D 생성을 촉진한다.

기하 인코딩 덕분에 TRELLIS와 동일 해상도에서도 더 나은 기하 재구성을 달성하며, 효율적인 학습을 통해 해상도가 증가할수록 탱크 트랙 같은 복잡한 구조의 더 많은 디테일이 드러난다.

TripoSF는 최초로 물체의 '후면'뿐만 아니라 '내부 구조'(버스 좌석 및 운전석 예시)까지 생성할 수 있으며, 기존 작업들이 의류나 꽃잎을 과도하게 두꺼운 기하 구조로 생성하는 경향이 있었던 반면, TripoSF는 열린 표면 자산을 탁월하게 처리한다.

**데이터 품질 전략**:
200만 개의 고품질 "image-SDF" 학습 쌍으로 구성된 데이터셋을 구축하였으며, 애블레이션 연구(ablation study)는 이 정제된 데이터셋으로 학습된 모델이 더 크지만 필터링되지 않은 원시 데이터셋으로 학습된 모델보다 유의미하게 우수함을 명확히 보여준다.

---

### 2.5 한계

논문에서 명시적으로 언급된 한계는 다음과 같습니다:

1. **카메라 중심 의존성**: frustum-aware sectional voxel training 접근 방식은 비카메라 중심 시나리오(non-camera-centric scenarios)에 적용될 때 잠재적인 한계가 있다.

2. **동적 형상 적용 한계**: 실시간 응용을 위한 동적(dynamic) 또는 변형 가능한(deformable) 3D 형상을 지원하도록 SparseFlex를 확장하는 것이 과제로 남아 있다.

3. **도메인 일반화의 불확실성**: 렌더링 손실의 통합이 의료 영상(medical imaging) 대 엔터테인먼트와 같이 서로 다른 도메인에서 SparseFlex의 일반화 능력에 어떤 영향을 미치는지가 연구 과제이다.

4. **스케일링 한계**: G-Shell은 방수 삼각 메시로부터 비방수 메시를 추출하는 새로운 3D 표현을 제안하고 이를 기반으로 확산 모델을 훈련하지만, 그 조밀한 그리드 구조는 복잡한 형상 처리 시 고해상도에서의 제약이 있다. SparseFlex 역시 매우 희소한 표면(extremely thin features)에서의 복셀 pruning 정확도 문제가 존재합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 부분은 SparseFlex 논문이 일반화 성능 측면에서 특히 중요한 기여를 합니다.

### 3.1 포인트 클라우드 입력을 통한 일반화

SparseFlex VAE는 희소 구조적 미분 가능 등값면 표현과 효율적인 frustum-aware sectional voxel training 전략 덕분에, 복잡한 기하(좌), 열린 표면(우상), 그리고 내부 구조(우하)에서 최첨단 성능을 보여주며, 임의의 토폴로지를 가진 고품질 image-to-3D 생성을 가능하게 한다.

이 논문은 임의의 토폴로지를 보존하며 원시 3D 형상의 디테일을 보존한 채로 3D 형상을 잠재 공간에 인코딩하고 재구성하는 기반(foundational) VAE를 개발하는 것을 목표로 한다.

### 3.2 Sparse 구조의 일반화 기여

희소 구조의 핵심 이점은 다음과 같다:
- **메모리 사용 대폭 감소**: TripoSF가 $1024^3$ 고해상도로 학습 및 추론 가능
- **임의 토폴로지 기본 지원**: 빈 영역의 복셀을 생략함으로써 열린 표면(천, 잎사귀 등)을 자연스럽게 표현하고 내부 구조를 효과적으로 처리

### 3.3 데이터 스케일을 통한 일반화

수천만 개의 고품질 네이티브 3D 자산으로 훈련된 Tripo 시리즈는 생성 속도, 모델 정확도, 전반적인 성공 면에서 지속적으로 새로운 기록을 세웠으며, 뛰어난 기하학적 정밀도로 3D 모델 생성의 프론티어를 재정의하였다.

### 3.4 DINOv2 기반 이미지 조건화를 통한 일반화

이미지 조건 특징이 DINOv2를 이용해 추출되고 교차 어텐션을 통해 트랜스포머 모델에 주입됨으로써, 다양한 도메인의 단일 이미지로부터 다양한 토폴로지의 3D 형상을 생성할 수 있는 강한 일반화 능력을 제공합니다.

### 3.5 일반화 성능 향상을 위한 추가 연구 방향

- **텍스처와 기하 통합 인코딩**: 일부 방법들은 멀티뷰 이미지 특징을 잠재 공간에 인코딩할 때 기하와 텍스처를 모두 디코딩하는 것을 목표로 하며, SparseFlex의 구조에 텍스처 정보를 통합하면 일반화 성능을 더욱 향상시킬 수 있습니다.
- **고품질 필터링 데이터셋**: 이 두 단계 프로세스는 VAE의 재구성 품질을 이후의 생성 성능에 있어 중요하게 만들며, 일부 연구들은 VAE의 형상 인코딩-디코딩 역량을 향상시켜 생성 품질을 개선하였다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### ① 3D 생성 모델의 새로운 기반 표현으로의 자리매김

SparseFlex 모델은 효율성을 희생하지 않고 해상도와 위상적 복잡성을 처리하면서 3D 형상 모델링 분야에서 중요한 진전을 이루었으며, 이 논문의 접근법과 결과는 고충실도 3D 형상 생성의 향후 발전을 위한 촉매제 역할을 할 수 있다.

#### ② 열린 표면(Open Surface) 연구 활성화

SparseFlex의 접근법은 복잡한 형상, 열린 표면, 심지어 내부 구조 재구성에서 우수한 성능을 보여주며, 의류, 식물, 기계 내부 구조와 같은 실세계 자산 생성 연구를 크게 활성화할 것입니다.

#### ③ 산업 적용 가능성

TripoSF는 SparseFlex 표현을 기반으로 VAST가 개발한 기초 3D 모델이며, 테스트 결과 기존의 모든 오픈소스 및 클로즈드소스 작업을 능가하였다. 사전 훈련된 VAE 모델과 관련 추론 코드가 오픈소스로 공개되었으며, Tripo 3.0에서 전체 버전이 공개될 예정이다.

#### ④ 희소 표현 + 생성 모델 결합 연구 방향 제시

SparseFlex와 관련된 후속 연구로는 Hyper3D, ShapeShifter, Pandora3D, DeepMesh, SuperCarver, MARS 등 다양한 고해상도 3D 형상 생성 연구들이 등장하고 있다.

---

### 4.2 앞으로 연구 시 고려할 점

#### ① 동적 형상 및 실시간 응용

실시간 응용을 위한 동적 또는 변형 가능한 3D 형상을 지원하도록 SparseFlex를 확장하는 방법에 대한 연구가 필요하다.

#### ② 확장성(Scalability)과 효율성의 균형

SparseFlex의 희소 복셀 기반 구조가 확장성과 효율성 면에서 다른 희소 표현 방법들과 어떻게 비교되는지에 대한 심층 연구가 필요하다.

#### ③ 다중 도메인 일반화 검증

렌더링 손실의 통합이 의료 영상 대 엔터테인먼트 같이 서로 다른 도메인에서 SparseFlex의 일반화 능력에 어떤 영향을 미치는지 연구할 필요가 있다.

#### ④ 비카메라 중심 시나리오 대응

frustum-aware sectional voxel training 방식을 비카메라 중심 시나리오에 적용할 때의 잠재적 한계를 극복하는 방법에 대한 연구가 요구된다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 표현 방식 | 해상도 | 임의 위상 | 핵심 특징 |
|------|------|----------|--------|-----------|-----------|
| **NeRF** | 2020 | 암묵 신경 복사장 | - | △ | 렌더링 기반, 메시 추출 불안정 |
| **Marching Cubes + SDF** | (고전) | 조밀 SDF 격자 | 저~중 | ✗ | 방수 변환 필요, 고해상도 메모리 부담 |
| **3PSDF** | 2022 | 삼극 부호 거리 | 중 | △ | 표면 인접 영역 구별을 위한 삼극 부호 도입, 그러나 이진 점유 격자 기반 표면 추출로 불연속성과 아티팩트 발생 |
| **G-Shell** | ~2023 | 방수-비방수 변환 | 중 | △ | 비방수 메시 추출을 위한 새 3D 표현을 제안하나 조밀한 그리드 구조가 복잡한 형상의 고해상도 처리를 제한 |
| **TRELLIS** | 2024 | 구조화 3D 잠재 | 256³ | △ | 멀티모달 3D 생성, 기준점으로 비교됨 |
| **Flexicubes** | 2024 | SDF + 변형 가능 메시 | 중 | △ | 미분 가능 아이소서페이스 추출, 조밀 그리드 한계 |
| **UDF 기반 (Surf-D 등)** | 2024 | Unsigned Distance Field | 중 | ✓ | UDF 기반 확산 모델로 임의 위상 메시 생성 가능하나, 신경망 경사 추정의 부정확성으로 인해 표면 추출이 불안정 |
| **TripoSG** | 2025 | Rectified Flow + MoE | 고 | △ | TripoSG: 대규모 Rectified Flow 모델을 활용한 고충실도 3D 형상 합성 |
| **SparseFlex (본 논문)** | 2025 | 희소 Flexicubes | **1024³** | **✓** | 렌더링 손실 직접 최적화, 열린 표면 + 내부 재구성, ICCV 2025 |
| **Sparc3D** | 2025 | 희소 표현 | 고 | ✓ | SparseFlex 이후 후속 연구 방향 |

SparseFlex는 이러한 한계들을 해결하고 렌더링 감독을 활용하는 고해상도 미분 가능 메시 재구성 및 생성을 실현하는 새로운 희소 구조 등값면 표현이다.

---

## 📚 참고 자료 및 출처

| # | 출처 | 유형 |
|---|------|------|
| 1 | **arXiv:2503.21732** – He et al., "SparseFlex: High-Resolution and Arbitrary-Topology 3D Shape Modeling" (2025) | 원본 논문 |
| 2 | **ICCV 2025 Open Access** – openaccess.thecvf.com/content/ICCV2025/papers/He_SparseFlex... | 학회 논문 |
| 3 | **프로젝트 페이지** – https://xianglonghe.github.io/TripoSF | 공식 프로젝트 |
| 4 | **GitHub Repository** – https://github.com/VAST-AI-Research/TripoSF | 코드 저장소 |
| 5 | **Hugging Face Model** – https://huggingface.co/VAST-AI/TripoSF | 모델 허브 |
| 6 | **Hugging Face Paper** – https://huggingface.co/papers/2503.21732 | 논문 요약 |
| 7 | **ResearchGate** – https://www.researchgate.net/publication/390247938 | 논문 DB |
| 8 | **VAST 공식 블로그** – https://www.tripo3d.ai/blog/vast-open-source-month | 기술 블로그 |
| 9 | **EmergentMind** – https://www.emergentmind.com/papers/2503.21732 | 논문 분석 |
| 10 | **Moonlight Literature Review** – https://www.themoonlight.io/en/review/sparseflex-... | 문헌 리뷰 |
| 11 | **Awesome 3D Generation** – https://awesome3dgen.com | 관련 연구 목록 |
| 12 | **Sparc3D** (후속 연구) – arXiv:2505.14521v2 | 후속 연구 |

> ⚠️ **정확도 참고**: 논문 내 세부 수식(특히 손실 함수 전체 구성, 정확한 파라미터 수치)은 공개된 arXiv HTML 버전 및 프로젝트 페이지를 기반으로 재구성한 것으로, 논문 원문 PDF의 정확한 표기와 일부 다를 수 있습니다. 가장 정확한 수식 확인을 위해서는 arXiv 원문 또는 ICCV 2025 proceedings 직접 확인을 권장합니다.
