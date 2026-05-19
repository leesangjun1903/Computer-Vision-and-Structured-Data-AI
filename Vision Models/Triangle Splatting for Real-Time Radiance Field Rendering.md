
# Triangle Splatting for Real-Time Radiance Field Rendering 

> **논문 정보**
> - **제목:** Triangle Splatting for Real-Time Radiance Field Rendering
> - **저자:** Jan Held, Renaud Vandeghen, Adrien Deliege, Abdullah Hamdi, Anthony Cioppa, Silvio Giancola, Andrea Vedaldi, Bernard Ghanem, Andrea Tagliasacchi, Marc Van Droogenbroeck
> - **arXiv:** [2505.19175](https://arxiv.org/abs/2505.19175) (2025)
> - **게재:** Proceedings of 3DV (IEEE, 2026)
> - **공식 페이지:** https://trianglesplatting.github.io/
> - **GitHub:** https://github.com/trianglesplatting/triangle-splatting

---

## 1. 핵심 주장 및 주요 기여 요약

컴퓨터 그래픽스 분야는 NeRF(Neural Radiance Fields) 및 3D Gaussian Splatting과 같은 모델들로 혁신이 이루어졌으며, 이로 인해 삼각형(triangle)은 포토그래메트리의 지배적인 표현 방식에서 밀려났다. 이 논문은 삼각형의 복귀(triangle comeback)를 주장한다.

이 논문은 비구조적(unstructured) 3D 삼각형 프리미티브("triangle soup")를 직접 최적화하는 새로운 미분 가능 렌더링(differentiable rendering) 접근법을 제안하여 사실적인 새로운 뷰 합성(novel view synthesis)을 달성한다.

### 주요 기여 4가지


**(i)** 비구조적 삼각형을 직접 최적화하는 Triangle Splatting을 제안하여 전통적인 컴퓨터 그래픽스와 방사휘도 필드(radiance fields)를 연결한다.
**(ii)** 소프트 삼각형 경계를 위한 미분 가능 윈도우 함수(differentiable window function)를 도입하여 효과적인 그래디언트 흐름(gradient flow)을 가능하게 한다.
**(iii)** Triangle Splatting이 시각적 품질 및 렌더링 속도에서 동시대의 경쟁 방법들을 능가하고, 실내 장면에서 Zip-NeRF 대비 우수한 지각적 품질을 달성함을 정성적, 정량적으로 입증한다.
**(iv)** 최적화된 삼각형이 표준 메시(mesh) 기반 렌더러와 직접 호환되어 전통적인 그래픽스 파이프라인에 원활하게 통합된다.


---

## 2. 논문의 상세 분석

### 2-1. 해결하고자 하는 문제

Neural Radiance Fields(NeRF)은 연속적 볼류메트릭(volumetric) 방사휘도 필드로 높은 품질의 이미지 합성을 달성했으나, 긴 학습·렌더링 시간이 실용성을 제한했다. 3D Gaussian Splatting(3DGS)은 수백만 개의 가우시안으로 장면을 표현하여 실시간 렌더링과 빠른 최적화를 가능하게 했지만, 가우시안 프리미티브는 VR 헤드셋이나 실시간 그래픽스 애플리케이션에서 사용되는 메시 기반 파이프라인과 기본적으로 호환되지 않는다. 기존 솔루션들은 후처리(post-processing) 또는 2단계 파이프라인을 통해 가우시안을 메시로 변환하려 했으나, 복잡성이 증가하고 시각적 품질이 저하되는 문제가 있었다.

즉, 논문이 해결하려는 핵심 문제는:
1. **NeRF의 느린 렌더링 속도** 문제
2. **3DGS의 전통적인 그래픽스 파이프라인 비호환성** 문제
3. **삼각형 프리미티브의 미분 불가능성** 문제 (삼각형 경계가 불연속적이어서 그래디언트 기반 최적화가 어려움)

---

### 2-2. 제안하는 방법 (수식 포함)

#### (a) 기본 표현 방식 (Triangle Primitive)

렌더링 파이프라인은 3D 삼각형을 프리미티브로 사용하며, 각 삼각형은 학습 가능한(learnable) 3D 정점(vertices) 3개, 색상(color), 불투명도(opacity), 스무스니스 파라미터 $\sigma$로 정의된다. 삼각형들은 알려진 내부 파라미터(intrinsics)와 외부 파라미터(extrinsics)를 가진 표준 핀홀 카메라 모델(pinhole camera model)을 사용하여 이미지 평면에 투영된다.

각 삼각형의 파라미터를 수식으로 표현하면:

$$T = \{v_1, v_2, v_3 \in \mathbb{R}^3,\; c \in \mathbb{R}^3,\; o \in [0, 1],\; \sigma > 0\}$$

#### (b) 투영 (Projection)

미분 가능 렌더링 과정은 표준 카메라 내부 파라미터 $K$ 및 포즈 $(R, t)$를 사용하여 각 3D 삼각형을 2D 이미지 평면에 투영하여 투영된 삼각형 $T_{2D}$를 생성한다.

$$T_{2D} = \pi(v_i;\; K, R, t), \quad i \in \{1, 2, 3\}$$

#### (c) 핵심: 미분 가능 윈도우 함수 (Differentiable Window Function)

삼각형을 이진(binary) 불투명 형태로 렌더링하는 대신, 각 픽셀 $p$에서의 영향이 부드러운 윈도우 함수 $I(p) \in [0, 1]$로 가중치가 매겨지는 미분 가능 스플랫(splat)으로 렌더링된다. 이 윈도우 함수의 핵심 설계는 2D 투영 삼각형의 부호 있는 거리 필드(SDF, Signed Distance Field) $\phi(p)$를 기반으로 하며, 다음과 같이 정의된다:

$$\phi(p) = \max_{i \in \{1,2,3\}} L_i(p)$$

여기서 $L_i(p) = n_i \cdot p + d_i$는 외향 단위 법선 $n_i$를 가진 세 개의 에지에 대한 부호 있는 거리 함수이다. SDF는 삼각형 내부에서 음수, 경계에서 0, 외부에서 양수가 된다.

이를 이용한 윈도우 함수 $I(p)$:

$$I(p) = \text{sigmoid}\!\left(-\frac{\phi(p)}{\sigma}\right)$$

$\sigma$가 증가하면 식의 지지(support)가 삼각형 푸트프린트(footprint)를 초과하여 래스터화(rasterization) 작업에 부적합해진다. 극단적인 경우, 식은 전역적(globally) 지지를 가지게 되어 모든 삼각형이 이미지의 모든 픽셀의 색상에 기여하게 된다.

#### (d) 색상 합성 (Alpha-Compositing)

깊이 정렬 후 알파-블렌딩(alpha-blending) 방식으로 최종 픽셀 색상 $\hat{C}(p)$를 계산:

$$\hat{C}(p) = \sum_{i=1}^{N} c_i \cdot o_i \cdot I_i(p) \cdot \prod_{j < i} \left(1 - o_j \cdot I_j(p)\right)$$

#### (e) 학습 손실 함수

파라미터들은 학습 이미지에 대한 렌더링 손실(rendering loss)을 최소화하기 위해 경사하강법(gradient descent)을 이용하여 공동 최적화(jointly optimized)된다.

$$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$$

---

### 2-3. 모델 구조

Triangle Splatting은 비구조적 3D 삼각형 프리미티브("triangle soup")를 직접 최적화하는 새로운 미분 가능 렌더링 방식을 제안한다. 핵심 아이디어는 삼각형의 계산 효율성 및 하드웨어 호환성을 Gaussian Splatting(3DGS)과 같은 프리미티브 기반 신경 렌더링 방법의 적응적 밀도(adaptive density) 및 미분 가능 속성과 결합하는 것이다. 이 방법은 고시각적 품질과 실시간 렌더링 성능을 달성하면서 전통적인 그래픽스 파이프라인과 직접 호환되는 표현을 생성하는 것을 목표로 한다.

모델의 전체 파이프라인은 아래와 같이 구성된다:

```
[입력: 다시점 이미지 + SfM 포인트 클라우드]
         ↓
[삼각형 초기화 (triangle soup)]
         ↓
[3D → 2D 투영 (핀홀 카메라 모델)]
         ↓
[미분 가능 윈도우 함수 I(p) 적용]
         ↓
[깊이 정렬 + Alpha-Compositing]
         ↓
[렌더링 손실(L1 + D-SSIM) 역전파]
         ↓
[삼각형 파라미터 업데이트 + 적응적 밀도 제어]
         ↓
[출력: 최적화된 Triangle Soup]
```

가우시안 스플래팅이나 컨벡스 스플래팅이 볼류메트릭(예: 가우시안, 복셀) 또는 솔리드(예: 컨벡스, 사면체) 프리미티브를 탐색한 것과 달리, Triangle Splatting은 표면(surface) 프리미티브를 제안하여 실제 세계 장면에서 가장 일반적으로 발견되는 고체 객체의 표면과 정렬된다.

Triangle Splatting에서 영향 범위는 정확히 세 정점의 투영으로 경계가 정해진다. 타일 커버리지(tile coverage)를 결정하기 위한 초기 추정값은 투영된 정점의 최소·최대 x, y 좌표를 계산하여 구한다. 이 방법은 계산이 간단하지만 보수적(conservative)이어서, 삼각형의 정점을 벗어난 픽셀에 대한 불필요한 연산이 발생할 수 있다.

---

### 2-4. 성능 향상

2D 및 3D Gaussian Splatting 방법 대비 더 높은 시각적 품질, 더 빠른 수렴, 향상된 렌더링 처리량을 달성한다. Mip-NeRF360 데이터셋에서 동시대의 비볼류메트릭(non-volumetric) 프리미티브보다 시각적 품질이 뛰어나고, 실내 장면에서 최첨단 Zip-NeRF보다 더 높은 지각적 품질을 달성한다.

표준 GPU 하드웨어 및 그래픽스 스택과 호환되어 *Garden* 장면에서 시판 메시 렌더러를 사용하여 1280×720 해상도에서 **2,400 FPS 이상**을 달성한다.

| 지표 | Triangle Splatting | 3D Gaussian Splatting | Zip-NeRF |
|---|---|---|---|
| 시각적 품질 (PSNR) | ✅ 우수 | 비교 기준 | ✅ 실내 열세 |
| 렌더링 속도 (FPS) | ✅ 2,400+ FPS | ~100–200 FPS | 느림 |
| 수렴 속도 | ✅ 빠름 | 보통 | 느림 |
| 파이프라인 호환성 | ✅ 직접 호환 | ❌ 변환 필요 | ❌ |

### 2-5. 한계점

Triangle Splatting은 고전적인 렌더링 파이프라인과 미분 가능 방사휘도 필드 프레임워크를 연결하는 초기 단계에 불과하다. 학습이 불투명한(opaque) 삼각형을 강제하지 않고 소프트(soft)하고 반투명한(semi-transparent) 삼각형에 의존하기 때문에, 게임 엔진에서 triangle soup을 렌더링할 때 눈에 띄는 시각적 품질 저하가 발생한다. 또한 모든 삼각형이 고립되어 있어, 선택한 파라미터화로 인해 정점이 공간적으로 일치하거나 유사한 색상 분포를 공유하더라도 이웃 삼각형 간에 연결성(connectivity)을 확보할 수 없다.

현재 비주얼은 셰이더(shader) 없이 렌더링되었으며 게임 엔진 충실도를 위해 특별히 학습되거나 최적화되지 않았고, 이것이 제한된 시각적 품질의 원인이다. 그럼에도 불구하고, 이는 방사휘도 필드를 인터랙티브 3D 환경에 직접 통합하기 위한 중요한 첫 번째 단계를 보여준다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 삼각형 표현의 일반화 이점

이 연구는 3D 삼각형을 렌더링 프리미티브로 도입함으로써 방사휘도 필드 렌더링에서 중요한 발전을 이룬다. 고전적인 메시 표현에서 사용되는 동일한 프리미티브를 활용함으로써 신경 렌더링과 전통적인 그래픽스 파이프라인 사이의 간극을 좁힌다. Triangle Splatting은 볼류메트릭 및 암시적(implicit) 방법의 대안을 제시하며, 더 빠른 렌더링 성능으로 높은 시각적 품질을 달성한다.

"triangle soup" (즉, 비구조적이고 연결되지 않은 삼각형들)을 경사 기반(gradient-based) 방법을 통해 최적화하는 것은 템플릿 없는(template-free) 메시 최적화 목표를 향한 중요한 단계를 나타낼 수 있다. 이러한 접근법은 수십 년간의 GPU 가속 삼각형 처리 및 성숙한 메시 처리 문헌을 활용하여 이러한 기술들을 통합하기 쉽게 한다.

### 3-2. 다양한 도메인으로의 일반화 가능성

이는 방사휘도 필드를 인터랙티브 3D 환경에 직접 통합하기 위한 중요한 첫 번째 단계를 보여준다. 향후 연구는 메시 기반 렌더러에서 시각적 충실도를 최대화하기 위해 특별히 조정된 학습 전략을 탐색할 수 있으며, AR/VR 또는 인터랙티브 시뮬레이션과 같은 실시간 애플리케이션을 위한 표준 게임 엔진에 재구성된 장면을 원활하게 통합하는 길을 열 수 있다.

DARB-splatting은 마할라노비스 거리의 일반적인 감쇠 이방성 방사 기저 함수(DARBFs)로 지수 계열 가우시안 커널을 대체하며, 이러한 커널들은 순수 가우시안 대비 최대 34% 빠른 수렴과 상당한 메모리 절약을 지원한다. 이는 Triangle Splatting도 더 나은 충실도나 효율성을 위한 최적 윈도우 함수 선택으로 혜택을 받을 수 있음을 시사한다.

### 3-3. 후속 연구 Triangle Splatting+의 일반화 개선

Triangle Splatting+는 공유 정점 집합에서 시작하여 삼각형들이 공통 정점을 통해 자연스럽게 연결될 수 있도록 삼각형 파라미터화를 재정의하여 연결성을 활성화한다. 또한 불투명한 삼각형을 강제하는 학습 전략을 설계하여 최종 표현이 메시 기반 렌더링 파이프라인에 원활하게 통합되도록 한다. 결과 메시가 반연결(semi-connected)에 불과하더라도, 물리 기반 시뮬레이션이나 인터랙티브 워크스루 등 광범위한 다운스트림 애플리케이션에 충분하다. Triangle Splatting+는 VR/AR 환경과 현대 비디오 게임에 즉시 통합할 수 있는 길을 열어준다.

---

## 4. 향후 연구에 미치는 영향과 연구 시 고려할 점

### 4-1. 향후 연구에 미치는 영향

#### 📌 전통적 그래픽스와 신경 렌더링의 통합

Triangle Splatting은 볼류메트릭 및 암시적 방법에 대한 설득력 있는 대안을 제공하며, 더 빠른 렌더링 성능으로 높은 시각적 품질을 달성한다. 이 결과들은 Triangle Splatting을 메시 인식 신경 렌더링(mesh-aware neural rendering)을 향한 유망한 단계로 확립하며, 수십 년의 GPU 가속 그래픽스를 현대적인 미분 가능 프레임워크와 통합한다.

#### 📌 게임엔진·AR/VR 산업에 대한 영향

Gaussian Splatting이나 NeRF 기반 접근법과 달리, 후처리 없이 실시간·게임엔진 준비(game-engine-ready) 메시를 제공하여 Unity 또는 Unreal과 같은 엔진과의 즉각적인 호환성을 가능하게 한다. 이 방법은 리에주 대학교, 사이먼 프레이저 대학교, 메릴랜드 대학교, 브리티시 컬럼비아 대학교, 토론토 대학교 및 어도비 리서치 연구자들이 개발하였다.

#### 📌 후속 연구 방향 촉발

Triangle Splatting은 비구조적 삼각형 집합이 미분 가능 스플래팅 프레임워크 내에서 엔드-투-엔드로 최적화될 수 있음을 입증했다. 그러나 고전적인 렌더링 파이프라인과 미분 가능 방사휘도 필드 프레임워크를 연결하는 초기 단계에 불과하다.

Triangle Splatting+는 방사휘도 필드와 그래픽스 프리미티브를 통합하여 VR, AR 및 게임엔진 파이프라인에 신경 렌더링을 원활하게 통합할 수 있게 한다. 삼각형을 미분 가능하고 학습 가능한 빌딩 블록으로 전환함으로써, 물리 기반 시뮬레이션, 실시간 디지털 트윈(digital twin) 등 인터랙티브 3D 환경으로의 문을 열어준다.

### 4-2. 향후 연구 시 고려할 점

| 고려사항 | 세부 내용 |
|---|---|
| **불투명도 제어** | 소프트 반투명 삼각형 → 완전 불투명 삼각형으로의 전환 학습 전략 필요 (Triangle Splatting+에서 일부 해결) |
| **삼각형 연결성** | 고립된 triangle soup → 메시 위상(topology) 구조로 발전시키는 연구 필요 |
| **셰이더 통합** | 게임 엔진 내 셰이더 최적화 없이 학습되어 시각 품질 저하 발생; 셰이더-aware 학습 전략 탐색 필요 |
| **윈도우 함수 일반화** | $\sigma$ 스케줄링 전략 및 적응적 커널 선택 |
| **대규모 장면** | 현재 Mip-NeRF360 같은 소-중형 장면 위주; 도시 스케일·드론 장면 일반화 검증 필요 |
| **동적 장면** | 현재 정적 장면만을 다루며, 동적 객체나 시간 변화 장면에 대한 확장 연구 필요 |
| **초기화 민감도** | SfM 포인트 클라우드 기반 초기화에 의존하므로 희소 입력 뷰에서의 강인성 검토 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 프리미티브 | 렌더링 속도 | 메시 호환성 | 특징 |
|---|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | 암시적(MLP) | 매우 느림 | ❌ | 고품질, 높은 계산 비용 |
| **TensoRF** | 2022 | 텐서 분해 | 보통 | ❌ | 빠른 학습 속도 |
| **Plenoxels** | 2022 | 복셀(Voxel) | 빠름 | ❌ | 신경망 없이 방사휘도 필드 |
| **MobileNeRF** | 2023 | 폴리곤+텍스처 | 빠름 | 부분적 ✅ | 모바일 기기 대상 |
| **3DGS** (Kerbl et al.) | 2023 | 3D 가우시안 | 매우 빠름 | ❌ | 현 상태 최고 속도 기준 |
| **2DGS** | 2024 | 2D 가우시안 | 빠름 | 부분적 ✅ | 표면 재구성 강화 |
| **3D Convex Splatting** | 2024 | 3D 컨벡스 | 빠름 | ❌ | 기하학적 의미의 프리미티브 |
| **Triangle Splatting** | 2025 | 3D 삼각형 | **2,400+ FPS** | ✅ | 직접 메시 호환, 본 논문 |
| **Triangle Splatting+** | 2025 | 불투명 삼각형 | 빠름 | ✅✅ | 연결성 + 불투명도 강화 |

3D Convex Splatting(3DCS)은 다중 시점 이미지로부터 기하학적으로 의미 있는 방사휘도 필드를 모델링하기 위해 3D 스무스 컨벡스를 프리미티브로 활용하는 새로운 방법으로, 고품질 장면 재구성과 새로운 뷰 합성을 위한 새로운 표준이 될 잠재력을 보여준다.

GauRast는 기존 GPU 삼각형 래스터라이저에 지수(exponentiation) 및 적당한 산술 논리를 추가함으로써 3D Gaussian Splatting(및 확장적으로 Triangle Splatting)을 최소한의 하드웨어 오버헤드로 에지 디바이스에서 실시간으로 20배 이상 가속화할 수 있음을 보여준다. 이는 Triangle Splatting과 고전적인 삼각형 기반 렌더링 파이프라인 간의 깊은 정렬을 활용한다.

---

## 📚 참고 자료 및 출처

1. **arXiv 논문 (원문):** Held et al., "Triangle Splatting for Real-Time Radiance Field Rendering," arXiv:2505.19175, 2025. https://arxiv.org/abs/2505.19175
2. **공식 프로젝트 페이지:** https://trianglesplatting.github.io/
3. **공식 GitHub 구현:** https://github.com/trianglesplatting/triangle-splatting
4. **Oxford 연구 아카이브 (ORA):** https://ora.ox.ac.uk/objects/uuid:be4d300d-71cb-425a-a00b-d9eae9bb5c4b
5. **Oxford 공식 PDF:** https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/held26triangle.pdf
6. **ResearchGate:** https://www.researchgate.net/publication/392104998
7. **Moonlight Literature Review:** https://www.themoonlight.io/en/review/triangle-splatting-for-real-time-radiance-field-rendering
8. **Emergent Mind (비교 분석):** https://www.emergentmind.com/topics/triangle-splatting
9. **후속 연구 Triangle Splatting+ (arXiv:2509.25122):** https://arxiv.org/abs/2509.25122
10. **Triangle Splatting+ 프로젝트 페이지:** https://trianglesplatting2.github.io/trianglesplatting2/
11. **OpenCV 블로그 (Triangle Splatting+):** https://opencv.org/triangle-splatting/
12. **Semantic Scholar (3DGS 원본):** https://www.semanticscholar.org/paper/3D-Gaussian-Splatting...

> ⚠️ **주의 사항:** 일부 수식(알파-컴포지팅, 손실 함수)은 논문의 표준적인 방사휘도 필드 렌더링 관행과 공개된 구현 코드에 기반하여 정리하였으며, 논문 원문의 세부 수식 표기와 일부 차이가 있을 수 있습니다. 정확한 수식은 arXiv 원문 PDF를 참조하시기 바랍니다.
