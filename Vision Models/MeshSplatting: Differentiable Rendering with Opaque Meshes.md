
# MeshSplatting: Differentiable Rendering with Opaque Meshes

> **논문 정보**
> - **제목**: MeshSplatting: Differentiable Rendering with Opaque Meshes
> - **저자**: Jan Held, Sanghyun Son, Renaud Vandeghen, Daniel Rebain, Matheus Gadelha, Yi Zhou, Anthony Cioppa, Ming C. Lin, Marc Van Droogenbroeck, Andrea Tagliasacchi
> - **arXiv**: [2512.06818](https://arxiv.org/abs/2512.06818) (2025년 12월 7일)
> - **프로젝트 페이지**: https://meshsplatting.github.io/
> - **공식 코드**: https://github.com/meshsplatting/mesh-splatting

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

3D Gaussian Splatting(3DGS)과 같은 Primitive 기반 Splatting 방법들은 실시간 렌더링으로 Novel View Synthesis 분야에 혁신을 가져왔지만, 이들의 포인트 기반 표현 방식은 AR/VR과 게임 엔진을 구동하는 메시 기반 파이프라인과 호환되지 않는다는 문제를 제기합니다.

이에 대한 핵심 해결책으로, MeshSplatting은 대규모 실세계 메시를 end-to-end로 재구성하는 최초의 방법으로, 후처리 없이 연결된(connected), 불투명한(opaque), 색상이 입혀진(colored) 삼각형 메시를 직접 생성합니다.

### 📋 주요 기여 (4가지)

MeshSplatting은 다음 네 가지 핵심 한계를 해결합니다:
① 시각적 품질을 유지하면서 현재 최신 기법보다 2배 빠르게 훈련하는 메시 기반 장면 표현의 end-to-end 최적화,
② 분리된 삼각형 수프(triangle soup) 대신 Restricted Delaunay Triangulation의 정점 위치를 정제하여 연결된 메시 생성,
③ 정점에 저장된 양(quantities)이 각 삼각형 전체에 부드럽게 보간되도록 삼각형을 자연스럽게 연결,
④ 삼각형이 불투명해야 함을 인식하는 최적화를 통해 표준 게임 엔진에서 직접 고품질 렌더링 지원(깊이 버퍼, 오클루전 컬링 등 고전적 기술 활용 가능)

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능

### 🚨 해결하고자 하는 문제

3DGS 같은 기존 Primitive 기반 Splatting 방법들은 실시간 Novel View Synthesis를 가능하게 하지만, AR/VR 및 게임 엔진에서 사용되는 메시 기반 파이프라인과 호환되지 않는다. Gaussian Radiance Field를 메시로 변환하는 과정은 복잡하고 미분 불가능한 후처리에 의존하며, 이는 시각적 품질 손실과 물리 기반 시뮬레이션에 부적합한 분리된 삼각형 수프(disconnected triangle soup)를 생성한다.

따라서 본 논문의 목표는 표준 그래픽스 파이프라인과 호환되는 고품질 Novel View Synthesis를 위해 불투명하고 연결된 폴리곤 메시를 직접 최적화하는 end-to-end 미분 가능 렌더링 접근법을 개발하는 것이다.

---

### ⚙️ 제안하는 방법 (수식 포함)

#### 2.1 Triangle Splatting 기반 렌더링 (Stage 1: Triangle Soup 최적화)

MeshSplatting은 본 연구에서 미분 가능 렌더링을 위한 Volume-Renderable Primitive로 Triangle Splatting을 활용합니다.

렌더링 파이프라인은 3D 삼각형을 Primitive로 사용하며, 각 삼각형은 세 개의 학습 가능한 3D 정점, 색상, 불투명도, 부드러움 파라미터 $\sigma$로 정의됩니다.

각 픽셀의 색상은 투영된 Primitive를 합성하여 계산합니다. 표준 볼륨 렌더링 공식에서 출발하면:

$$
C = \sum_{i=1}^{N} c_i \cdot \alpha_i \cdot \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

여기서:
- $C$: 렌더링된 픽셀 색상
- $c_i$: $i$번째 삼각형의 색상
- $\alpha_i$: $i$번째 삼각형의 불투명도 기여도

Triangle Splatting에서 각 삼각형의 불투명도 기여는 소프트 윈도우 함수를 통해 계산됩니다:

$$
\alpha_i(p) = o_i \cdot w(d(p, T_i), \sigma_i)
$$

- $o_i$: 학습 가능한 불투명도 파라미터
- $d(p, T_i)$: 픽셀 $p$에서 삼각형 $T_i$의 경계까지의 거리
- $\sigma_i$: 부드러움(smoothness) 파라미터

MeshSplatting은 안정적인 그래디언트 흐름을 보장하기 위해 불투명도 파라미터 스케줄링과 윈도우 파라미터 스케줄링을 적용하여, 초기 단계에서는 반투명 상태를 허용하고 최종적으로는 완전히 불투명하고 날카로운 삼각형으로 수렴합니다.

즉, 훈련 초기에는 $o_i \in (0, 1)$이지만, 스케줄링을 통해 점진적으로 $o_i \to 1$로 수렴하도록 유도합니다.

#### 2.2 Restricted Delaunay Triangulation을 통한 연결 메시 생성 (Stage 2)

Stage 2에서는 초기 최적화 후 최적화된 정점 집합에 대해 Restricted Delaunay Triangulation을 수행하여 연결성을 확립한다. 이 작업은 정점의 위치를 수정하거나 새로운 정점을 도입하지 않고 기존 정점을 재사용하여 학습된 공간적 정확도와 외관을 보존한다. 결과로 생성된 연결 메시는 이후 정제(refinement) 단계를 거치며, 공유 정점이 인접 삼각형 전반에 걸쳐 일관된 업데이트를 보장한다.

Restricted Delaunay Triangulation을 적용하면 전역적 연결성은 복원되지만, 정점 색상이 더 이상 기하학과 정확히 정렬되지 않아 기하학적 아티팩트와 시각적 품질 손실이 발생할 수 있다. 최종 미세 조정(fine-tuning) 단계에서 연결된 메시를 정제하여 부드러운 표면과 정확한 기하학을 달성하고 삼각형 분할 과정에서 손실된 시각적 충실도를 복원한다.

전체 손실 함수는 기본적으로 다음과 같이 구성됩니다:

$$
\mathcal{L} = \mathcal{L}_\text{rgb} + \lambda_\text{normal} \mathcal{L}_\text{normal} + \lambda_\text{opacity} \mathcal{L}_\text{opacity}
$$

- $\mathcal{L}_\text{rgb}$: 렌더링된 이미지와 GT 이미지 간의 포토메트릭 손실
- $\mathcal{L}_\text{normal}$: 법선 일관성(normal consistency) 손실 — 인접 삼각형 간 표면 일관성 유지
- $\mathcal{L}_\text{opacity}$: 삼각형이 완전 불투명에 수렴하도록 유도하는 손실

---

### 🏗️ 모델 구조 (전체 파이프라인 요약)

MeshSplatting은 2단계 최적화(two-stage optimization)를 채택하여, 초기 삼각형 수프(triangle soup)를 구조화된 메시로 점진적으로 변환한다. 연결성을 부과하기 전에 빠른 장면 커버리지를 위한 비제약(unconstrained) 최적화부터 시작한다.

```
[전체 파이프라인]

Stage 1: Triangle Soup 최적화
  ├── 포인트 클라우드(SfM)로부터 초기 삼각형 배치
  ├── 로컬 밀도 기반 삼각형 크기 스케일링
  ├── 미분 가능 렌더링으로 독립 삼각형 최적화
  └── Opacity/Window Parameter Scheduling 적용

Stage 2: Mesh 생성 및 정제
  ├── Restricted Delaunay Triangulation 수행 (1회, <2분)
  ├── 연결 메시 생성 (정점 위치 보존)
  └── Fine-tuning: 법선 일관성 + 포토메트릭 손실로 정제

[출력]
  └── 연결된 불투명 삼각형 메시
       ├── 게임 엔진 직접 임포트 지원
       ├── 물리 시뮬레이션 지원
       └── 레이 트레이싱 지원
```

이 표현은 표준 게임 엔진과 호환되어 투명도를 위한 사후 변환이나 커스텀 렌더링 루틴이 필요 없으며, 물리적 상호작용, 인터랙티브 워크스루, 레이 트레이싱을 네이티브로 지원한다. MeshSplatting은 간단한 오브젝트 추출을 가능하게 하여 장면 요소를 게임 엔진에서 직접 내보내고 가져올 수 있다.

---

### 📊 성능 향상

Mip-NeRF360에서 MeshSplatting은 메시 기반 Novel View Synthesis의 현재 SOTA인 MiLo보다 PSNR +0.69 dB 향상을 달성하면서 2배 빠른 훈련 속도와 2배 적은 메모리 사용량을 실현하여 Neural Rendering과 인터랙티브 3D 그래픽스를 연결한다.

MeshSplatting은 평균 48분 훈련으로 동시 메시 생성 방법들 대비 35~55% 속도 향상을 달성한다. 최적화된 Restricted Delaunay Triangulation은 2분 이내에 실행되어 전체 훈련 시간에 거의 영향을 주지 않는다. MiLo는 매 이터레이션마다 Delaunay Triangulation을 수행하는 반면, 본 방법은 단 한 번만 실행한다. 이로 인해 MiLo의 훈련 시간은 106분인 반면 MeshSplatting은 48분에 불과하다.

MeshSplatting이 생성하는 최종 메시 표현은 100 MB에 불과하여 동시 방법들 대비 약 15배 감소한 매우 컴팩트한 크기를 보인다.

MeshSplatting은 유사하거나 더 적은 수의 정점을 사용하면서 Mip-NeRF360과 Tanks & Temples 데이터셋 모두에서 더 높은 PSNR, SSIM, 낮은 LPIPS를 달성하여 동시대 메시 기반 Novel View Synthesis 방법들을 시각적 품질에서 크게 능가한다.

| 지표 | MiLo (SOTA) | MeshSplatting | 향상 |
|------|:-----------:|:-------------:|:----:|
| PSNR (Mip-NeRF360) | 기준 | +0.69 dB ↑ | ✅ |
| 훈련 시간 | 106 min | 48 min | 2× 빠름 |
| 메모리 사용량 | 기준 | 2× 감소 | ✅ |
| 메시 크기 | 기준 | 100 MB (~15× 감소) | ✅ |

---

### ⚠️ 한계 (Limitations)

문서화된 한계로는 희소하거나 관측되지 않은 영역에서의 불완전한 커버리지, 불투명 메시 스플랫의 진정한 반투명(translucency) 표현 불가, 그리고 워터타이트(watertight)하거나 매니폴드(manifold) 보장의 잠재적 부재가 포함된다.

추가적으로:
- DMesh++가 단일 객체 재구성에 초점을 맞추듯이, 특정 씬 유형(실내 vs. 실외 복잡 장면)에 따른 성능 편차 가능성이 있다.
- Restricted Delaunay Triangulation 적용 시 전역 연결성은 복원되지만, 정점 색상이 기하학과 더 이상 정확하게 정렬되지 않아 기하학적 아티팩트와 시각적 품질 손실이 발생할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

삼각형 수에 따른 효과적인 스케일링 능력 — 특히 Mip-NeRF360의 야외 장면에서 정점 수가 증가할수록 시각적 품질(특히 LPIPS)의 일관된 향상을 보임 — 은 강건한 설계를 보여준다. 이는 MeshSplatting이 증가된 기하학적 복잡성을 활용하여 시각적 충실도를 더욱 향상시킬 수 있음을 나타내며, 다양한 씬 복잡도에 대한 확장 가능한 솔루션임을 의미한다.

### 일반화 성능 향상을 위한 핵심 요소

**① 스케일러블 표현 (삼각형 수 조절)**

삼각형 수 $N$을 조절함으로써 단순/복잡 장면 모두에 유연하게 대응 가능합니다:

$$
N_\text{triangles} \uparrow \Rightarrow \text{LPIPS} \downarrow, \quad \text{PSNR} \uparrow \quad \text{(특히 야외 복잡 장면)}
$$

**② 표준 그래픽스 파이프라인 호환성**

본 논문은 Restricted Delaunay Triangulation을 사용하여 연결된 불투명 삼각형 메시를 직접 최적화하는 완전한 미분 가능 파이프라인을 도입한다. 이전 메시 기반 View Synthesis 방법 대비 PSNR, 메모리 사용량, 훈련 시간 측면에서 현저한 향상을 달성한다. 게임 엔진과의 원활한 통합을 통해 실시간 Novel View Synthesis, 효율적인 객체 분리, 그리고 인터랙티브 애플리케이션의 즉각적 배포를 가능하게 한다.

**③ 연결 메시 구조의 일반화 이점**

MeshSplatting은 Restricted Delaunay Triangulation을 통해 정점 위치를 정제하여 연결된 메시를 생성함으로써 정점 전반에 걸쳐 자연스러운 삼각형 연결성과 부드럽게 보간된 양을 보장한다. 삼각형이 완전히 불투명한 표현을 생성하는 최적화를 통해 표준 게임 엔진에서 직접 고품질 렌더링을 지원하고, 깊이 버퍼(depth buffer)와 오클루전 컬링(occlusion culling) 같은 고전적 기술도 지원한다.

**④ 미래 일반화 확장 방향**

미래 연구는 신경 텍스처(neural textures), 위상학적 정제(topological refinement), 혼합 메시/볼류메트릭 표현(hybrid mesh/volumetric representations)을 통합하는 방향을 지향하며, 동적 해상도, 스플랫 감소, 표면 정렬 정규화를 통한 효율성 향상이 계속 이루어질 것이다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 🌐 앞으로의 연구에 미치는 영향

**① Neural Rendering ↔ 게임 엔진 브리지 역할**

Gaussian 기반 접근법과 달리 결과 표현은 게임 엔진과 즉시 호환되어 리라이팅(relighting), 물리 시뮬레이션, 인터랙티브 워크스루 같은 다운스트림 애플리케이션을 가능하게 한다. Triangle Splatting+는 Radiance Field 최적화와 전통적인 그래픽스 파이프라인을 연결하여 인터랙티브 VR 애플리케이션, 게임 엔진, 시뮬레이션 프레임워크에 Radiance Field 표현을 실용적으로 통합하는 길을 열어준다.

**② Triangle Splatting 계열 연구의 발전 기반**

Triangle Splatting은 Gaussian을 삼각형으로 대체하는 MeshSplatting의 가장 직접적인 선행 연구이다. MeshSplatting은 삼각형에 대한 볼류메트릭 렌더링 공식을 직접 계승하면서 핵심 한계인 '연결된 메시 대신 삼각형 수프' 문제와 '게임 엔진에서 직접 사용 가능한 불투명 삼각형'을 해결한다.

**③ 새로운 비교 기준선 제시**

MeshSplatting은 대규모 실세계 메시를 end-to-end로 재구성하는 최초의 방법으로, 후처리 없이 연결된, 불투명한, 색상 부여된 삼각형 메시를 직접 생성함으로써 앞으로의 메시 기반 Novel View Synthesis 연구의 새로운 기준점(baseline)이 됩니다.

---

### 🔬 앞으로의 연구 시 고려할 점

| 연구 방향 | 구체적 고려 사항 |
|-----------|----------------|
| **동적 장면** | 정적 장면에 최적화된 현재 방법을 동적 객체/움직임이 있는 장면으로 확장 필요 |
| **반투명 재질 표현** | 불투명 삼각형의 한계 극복: 유리, 물, 연기 등 반투명 재질 표현을 위한 하이브리드 접근법 |
| **워터타이트 메시** | 물리 기반 시뮬레이션을 위한 완전한 워터타이트(watertight) 및 매니폴드(manifold) 메시 보장 연구 |
| **신경 텍스처 통합** | 단순 정점 색상을 넘어 뷰 의존적 조명/텍스처(Neural Texture)를 메시에 연동 |
| **제너럴라이제이션** | 특정 씬 타입(실내/실외)에 관계없이 동작하는 범용 모델 설계 |
| **위상 변화 처리** | Restricted Delaunay Triangulation의 한계인 위상적(topological) 구조 변화 처리 능력 향상 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

Triangle Splatting은 볼류메트릭 및 암묵적 방법의 매력적인 대안으로, 더 빠른 렌더링 성능으로 높은 시각적 충실도를 달성한다. 이 결과들은 Triangle Splatting을 메시 인식 Neural Rendering을 향한 유망한 단계로 자리매김하게 하며, 수십 년의 GPU 가속 그래픽스와 현대 미분 가능 프레임워크를 통합한다.

| 방법 | 연도 | 표현 방식 | 게임 엔진 호환 | 연결 메시 | 성능 |
|------|:----:|:--------:|:------------:|:--------:|:----:|
| **NeRF** | 2020 | 암묵적(implicit) | ❌ | ❌ | PSNR 기준선 |
| **3D Gaussian Splatting** | 2023 | Gaussian | ❌ (변환 필요) | ❌ | 높음 |
| **2D Gaussian Splatting** | 2024 | 2D Gaussian | ❌ | ❌ | 표면 품질 개선 |
| **DMesh++** | 2024 | 미분가능 메시 | △ | ✅ | 단일 객체 한정 |
| **3D Convex Splatting** | 2025 | 3D 볼록 Primitive | △ | ❌ | 볼류메트릭 |
| **Triangle Splatting** | 2025 | 삼각형 수프 | △ | ❌ (수프) | 높음 |
| **MiLo** | 2025 | GS + 메시 혼합 | △ | ✅ | SOTA (이전) |
| **MeshSplatting** | 2025 | 연결된 삼각형 메시 | ✅ (네이티브) | ✅ | **현 SOTA** |

MiLo(Mesh-In-the-Loop Gaussian Splatting)는 Mip-NeRF360에서 메시 기반 Novel View Synthesis의 현재 SOTA 방법으로, Gaussian 스플랫 표현과 메시 표현을 공동으로 최적화한다. 그러나 MeshSplatting은 이를 넘어서는 성능을 달성했습니다.

Triangle Splatting은 MeshSplatting이 미분 가능 삼각형 파라미터화와 렌더링 기술을 계승한 가장 관련성 높은 연구이며, 3D Convex Splatting은 볼록 Primitive를 이용한 미분 가능 렌더링을 도입하여 삼각형을 활용한 접근법에 영감을 주었다. Convex Splatting은 표면 기반인 MeshSplatting의 볼류메트릭 대응 방법이다.

---

## 📚 참고 자료 (출처 목록)

1. **arXiv 논문 원문**: Held et al., "MeshSplatting: Differentiable Rendering with Opaque Meshes", arXiv:2512.06818, 2025. — https://arxiv.org/abs/2512.06818
2. **공식 프로젝트 페이지**: https://meshsplatting.github.io/
3. **공식 GitHub 코드**: https://github.com/meshsplatting/mesh-splatting
4. **Hugging Face 논문 페이지**: https://huggingface.co/papers/2512.06818
5. **ResearchGate 논문 PDF**: https://www.researchgate.net/publication/398475190_MeshSplatting_Differentiable_Rendering_with_Opaque_Meshes
6. **alphaXiv 논문 리뷰**: https://www.alphaxiv.org/overview/2512.06818v1
7. **ORBi (University of Liège)**: https://orbi.uliege.be/handle/2268/338445
8. **Liner Quick Review**: https://liner.com/review/meshsplatting-differentiable-rendering-with-opaque-meshes
9. **Emergent Mind**: https://www.emergentmind.com/papers/2512.06818
10. **관련 연구 - Triangle Splatting**: https://trianglesplatting.github.io/
11. **관련 연구 - Triangle Splatting+**: arXiv:2509.25122, https://arxiv.org/html/2509.25122v1
12. **관련 연구 - MeshSplats**: arXiv:2502.07754, https://arxiv.org/pdf/2502.07754

> ⚠️ **정확도 관련 주의사항**: 본 답변에서 수식(손실 함수 세부 항목, 렌더링 수식의 구체적 파라미터 표기)은 논문의 공개된 HTML/PDF 전문에서 직접 확인하기 어려운 부분에 대해 Triangle Splatting 계열 연구의 일반적 공식 형태를 기반으로 재구성한 내용이 포함되어 있습니다. 정확한 수식 표기는 arXiv 원문(https://arxiv.org/html/2512.06818v1) 또는 PDF를 직접 참고하시기 바랍니다.
