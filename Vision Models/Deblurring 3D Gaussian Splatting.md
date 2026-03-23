
# Deblurring 3D Gaussian Splatting

> **논문 정보**: Byeonghyeon Lee, Howoong Lee, Xiangyu Sun, Usman Ali, Eunbyung Park, *"Deblurring 3D Gaussian Splatting,"* **ECCV 2024**, Lecture Notes in Computer Science vol 15116, Springer.
> arXiv: 2401.00834

---

## 1. 핵심 주장 및 주요 기여 (요약)

본 논문은 소규모 Multi-Layer Perceptron(MLP)을 이용하여 각 3D Gaussian의 공분산(covariance)을 조작함으로써 장면의 블러를 모델링하는 실시간 디블러링 프레임워크, **Deblurring 3D Gaussian Splatting**을 제안합니다.

기존 3D Gaussian Splatting(3D-GS)은 학습 이미지가 흐릿할 경우 렌더링 품질이 심각하게 저하되는 문제가 있었습니다. 기존 디블러링 연구 대부분은 볼류메트릭 렌더링 기반 NeRF를 위해 설계되어 래스터화 기반 3D-GS에 직접 적용하기 어렵습니다.

**주요 기여 사항:**

1. 각 3D Gaussian의 공분산 행렬과 평균을 개별적으로 조작하여 공간적으로 변화하는 블러를 모델링하는 새로운 기술을 제안
2. SfM으로부터 얻은 희소 포인트 클라우드를 KNN 알고리즘을 이용하여 학습 중 밀도를 높이는 기법 도입
3. 200 fps 이상의 실시간 렌더링 속도(기존 NeRF 대비 200배)를 유지하면서 경쟁력 있는 이미지 품질 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

블러는 렌즈 디포커싱(defocusing), 객체 움직임(object motion), 카메라 흔들림(camera shake) 등으로 인해 발생하며, 깨끗한 이미지 획득을 불가피하게 방해합니다.

3D-GS는 학습 이미지가 흐릿하면 렌더링 품질이 심각하게 저하됩니다. 기존 연구들은 뉴럴 필드를 사용하여 흐릿한 입력에서 선명한 이미지를 렌더링하려 했으나, 대부분 볼류메트릭 렌더링 기반 NeRF용으로 설계되어 래스터화 기반 3D-GS에 바로 적용할 수 없습니다.

추가적으로, 3D-GS의 재구성 품질은 SfM에서 얻은 초기 포인트 클라우드에 크게 의존하는데, 입력 이미지가 흐릿하면 SfM이 희소한 포인트 클라우드만 생성하고, 특히 디포커스 블러 장면에서는 먼 곳에 위치한 포인트를 거의 추출하지 못합니다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) 3D Gaussian Splatting 기본 렌더링

3D-GS에서의 색상 렌더링은 알파 블렌딩으로 수행됩니다:

$$C = \sum_{i \in N} T_i \, c_i \, \alpha_i, \quad \text{where} \quad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$$

3D-GS는 다수의 색상이 부여된 3D Gaussian을 결합하여 3D 장면을 표현하며, 미분 가능한 splatting 기반 래스터화를 통해 렌더링합니다. 렌더링 과정에서 3D Gaussian 포인트를 2D 이미지 평면에 투영하고, 위치·크기·회전·색상·불투명도를 경사 기반 최적화로 조정합니다.

#### (B) Covariance 조작을 통한 디블러링

MLP는 3D Gaussian의 위치(position), 회전(rotation), 스케일(scale), 시선 방향(viewing direction)을 입력으로 받아 회전과 스케일에 대한 오프셋(offset)을 출력합니다. 이 오프셋은 원래의 회전 및 스케일에 요소별(element-wise)로 곱해져 변환된 3D Gaussian 기하 구조를 얻습니다. 각 Gaussian 별로 오프셋을 예측하므로 학습 이미지에서 흐릿한 부분의 Gaussian 공분산을 선택적으로 확대할 수 있습니다.

수식으로 표현하면, 논문에서 제시된 변환된 회전 $\hat{r}_j$와 스케일 $\hat{s}_j$는 다음과 같습니다:

$$\hat{r}_j = r_j \cdot \min\!\big(1.0,\; \lambda_s \, \delta r_j + (1 - \lambda_s)\big)$$

$$\hat{s}_j = s_j \cdot \min\!\big(1.0,\; \lambda_s \, \delta s_j + (1 - \lambda_s)\big)$$

여기서:
- $r_j$, $s_j$: $j$번째 3D Gaussian의 원래 회전, 스케일
- $\delta r_j$, $\delta s_j$: MLP가 출력한 오프셋
- $\lambda_s$: 학습 스케줄링 파라미터 (학습 진행에 따라 0→1로 증가)

이 방법론은 $\delta r_j$, $\delta s_j$를 통해 근접 영역(A)의 세밀한 블러링, 원거리 영역(B)의 블러링, 그리고 전체적인 블러/선명도 조절(C, D)을 유연하게 수행할 수 있습니다.

#### (C) 블러 시뮬레이션의 물리적 근거

이미지의 픽셀 블러는 디포커싱과 카메라 움직임에 의해 발생하며, 이 현상은 일반적으로 컨볼루션 연산으로 모델링됩니다. 카메라로 촬영된 이미지는 실제 이미지와 PSF(Point Spread Function)의 컨볼루션 결과입니다.

큰 분산을 가진 Gaussian은 더 넓은 이미지 영역에 영향을 미쳐 이웃 픽셀 간의 간섭을 표현할 수 있으며, 반면 작은 3D Gaussian은 장면의 세밀한 디테일을 모델링합니다.

#### (D) 카메라 모션 블러 처리

카메라 모션 블러의 경우, 카메라 노출 시간 동안의 카메라 움직임을 암시적으로 모델링합니다. 구체적으로, 기존 3D Gaussian 세트의 위치를 이동시켜 움직임의 이산적 순간을 표현하는 다중 보조 3D Gaussian 세트를 생성하여 카메라 모션 블러를 시뮬레이션합니다.

#### (E) 추론 시 구조

추론 시에는 MLP의 오프셋 없이 원래 3D-GS 구성 요소만으로 장면을 렌더링합니다. 이를 통해 각 픽셀이 인접 픽셀의 간섭에서 자유로워져 선명한 이미지를 생성합니다. MLP가 추론 시 비활성화되므로 3D-GS와 동일한 실시간 렌더링 속도를 유지합니다.

### 2.3 모델 구조

| 구성 요소 | 세부 사항 |
|---|---|
| **백본** | 3D Gaussian Splatting (래스터화 기반) |
| **MLP 구조** | 4층 MLP: 처음 3개 층은 공유, 이후 3개의 단일 층 헤드(δr, δs 등)로 분기 |
| **입력** | 각 Gaussian의 position, rotation, scale, viewing direction |
| **출력** | rotation offset ($\delta r_j$), scale offset ($\delta s_j$) |
| **포인트 클라우드 보강** | KNN 알고리즘으로 기존 포인트 주변에 추가 포인트 생성 + 상대적 깊이 기반 3D Gaussian 가지치기(pruning) |
| **학습 하드웨어** | NVIDIA RTX 4090 GPU |
| **학습 반복** | 카메라 모션 디블러링 시 M=5, 총 학습 반복 20,000회 |

**전체 워크플로우:**

점선 화살표와 대시 화살표는 각각 학습 시 카메라 모션 블러 모델링과 디포커스 블러 모델링 파이프라인을 나타내며, 실선 화살표는 추론 시 선명한 이미지 렌더링 과정을 보여줍니다.

### 2.4 성능 향상

실제 디포커스 블러 데이터셋에서 PSNR은 SOTA 모델과 동등하고, SSIM에서는 SOTA 성능을 달성합니다. 실제 카메라 모션 블러 데이터셋에서는 모든 메트릭(PSNR, SSIM, FPS)에서 SOTA 성능을 달성합니다.

렌더링 속도 면에서 기존 디블러링 NeRF(~1 FPS) 대비 800 FPS 이상의 렌더링 속도를 달성하면서도 경쟁력 있는 이미지 품질을 유지합니다.

| 메트릭 | Deblurring 3D-GS | 기존 Deblurring NeRF |
|---|---|---|
| **FPS** | > 800 FPS | ~1 FPS |
| **학습 시간** | ~30분 수준 | > 10시간 |
| **PSNR** | SOTA 수준 또는 동등 | 비교 가능 |
| **SSIM** | SOTA (defocus + motion) | 비교 가능 |

### 2.5 한계점

1. **단일 블러 유형 제한**: Lee et al.은 블러를 Gaussian 포인트의 공분산에 귀속시키고 MLP 기반으로 회전 및 스케일 오프셋을 예측하는 접근법을 제안했으나, 이 방법은 디포커스 블러 처리에는 효과적이지만 모션 관련 블러는 충분히 다루지 못한다는 지적이 있습니다. (논문 업데이트 v2에서 카메라 모션 블러 처리 추가)

2. **Per-scene 최적화 패러다임**: 대부분의 디블러링 3DGS 방법(본 논문 포함)은 per-scene 최적화 방식으로 작동하며, 이 설계는 보지 못한 장면에 대한 일반화를 제한하고 새로운 입력 시퀀스에 대해 재최적화가 필요합니다.

3. **SfM 의존성**: 초기 포인트 클라우드의 품질에 크게 의존하며, 심하게 흐릿한 이미지에서는 COLMAP이 실패할 수 있음

4. **심한 블러 한계**: RGB 단일 모달리티 디블러링 3DGS는 경미한 블러 또는 단순한 카메라 모션 시나리오에서만 효과적이며, 입력 이미지가 심하게 흐릿한 경우 3D 객체를 재구성하지 못하고 특정 각도에서만 비교적 선명한 2D 렌더링을 생성하는 데 그칩니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재의 일반화 한계

Deblurring 3D-GS는 **per-scene 최적화** 방식을 채택합니다. 즉, 각 장면마다 별도의 학습이 필요하며 새로운 장면에 대한 즉각적인 일반화(generalization)가 불가능합니다.

- **MLP의 역할**: MLP가 각 Gaussian의 오프셋을 학습하지만, 이는 특정 장면에 맞게 최적화됨
- **블러 유형 의존성**: 디포커스 블러와 카메라 모션 블러에 대해 별도의 파이프라인 설계가 필요
- **데이터셋 한정**: Deblur-NeRF 벤치마크(합성 + 실제 이미지)에서 평가되었으나, 다양한 실세계 시나리오로의 확장성은 검증 부족

### 3.2 일반화 향상을 위한 방향

피드포워드 3DGS 모델의 성장 추세를 고려할 때, 디블러링 모듈을 일반화 가능한 3DGS 방법에 통합하여 일반화 가능한 Gaussian Splatting의 입력 열화 문제를 해결하는 것이 유망한 미래 연구 방향입니다.

구체적 전략:
1. **Feed-forward 구조 통합**: Per-scene 최적화 대신, 사전 학습된 피드포워드 모델에 디블러링 모듈을 통합
2. **멀티모달 데이터 활용**: 이벤트 카메라 데이터와 3DGS를 통합하여 흐릿한 입력에서 더 robust한 3D Gaussian 표현 학습
3. **Diffusion Prior 활용**: DiET-GS처럼 확산 모델(diffusion prior)과 이벤트 스트림을 활용하여 블러-프리 정보를 두 단계 학습 전략으로 효과적 활용
4. **Dual-blur 통합 처리**: 기존 방법들은 단일 블러 시나리오에 특화되어 이중 블러가 존재하는 실세계 사례로 일반화하기 어렵습니다. 모션 블러와 디포커스 블러를 동시에 처리하는 통합 프레임워크 필요

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

1. **실시간 디블러링 렌더링의 가능성 입증**: NeRF 기반 디블러링의 느린 렌더링 속도 문제를 3D-GS 프레임워크로 전환하여 해결할 수 있음을 최초로 보여줌
2. **3D 표현 공간에서의 블러 모델링 패러다임**: 2D 이미지 공간이 아닌 3D Gaussian의 기하학적 속성 변환을 통한 블러 모델링 접근법은 후속 연구의 기반이 됨
3. **경량 보조 네트워크 설계**: 소규모 MLP로 큰 성능 향상을 달성하여, 효율적인 디블러링 모듈 설계의 방향을 제시

### 4.2 향후 연구 시 고려할 점

| 연구 방향 | 고려 사항 |
|---|---|
| **일반화** | Per-scene → generalizable 3DGS로 확장; feed-forward 방식 탐색 |
| **심한 블러** | 이벤트 카메라 등 보조 모달리티 활용 필요 |
| **동적 장면** | 기존 접근법은 정적 장면 재구성에 초점을 맞추고 동적 객체의 전용 모션 모델링이 부족합니다. |
| **Sparse View** | SOTA 3D 디블러링 방법들은 물리적 카메라 모션 모델링에 뛰어나지만 근본적으로 조밀한 다중 뷰 기하학적 제약에 의존하므로, 희소 뷰에서는 이 제약이 사라져 모션 추정이 ill-posed 문제가 됩니다. |
| **카메라 포즈 정확도** | 기존 3DGS 기반 디블러링 방법들의 성능은 카메라 포즈 정확도에 대한 극도의 의존성 등 내재적 한계로 제한됩니다. |
| **통합 블러 처리** | 모션 블러 + 디포커스 블러를 단일 프레임워크에서 처리하는 연구 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 (연도) | 접근 방식 | 블러 유형 | 렌더링 속도 | 핵심 특징 |
|---|---|---|---|---|
| **Deblur-NeRF** (CVPR 2022) | 분석-합성 접근법으로 흐릿한 입력에서 선명한 NeRF 복구를 최초로 시도 | Defocus + Motion | ~1 FPS | 최초의 NeRF 디블러링, 블러 커널 추정 |
| **DP-NeRF** (CVPR 2023) | DP-NeRF는 3D 카메라 모션을 통한 rigid 블러 커널 모델링 | Motion | ~1 FPS | 물리적 장면 사전지식 활용 |
| **BAD-NeRF** (2023) | SE(3) 공간에서 카메라 궤적 보간 | Motion | ~1 FPS | 노출 기간 동안 시작/끝 카메라 포즈를 예측하고 SE(3) 공간에서 선형 보간하여 가상 카메라 포즈 생성 |
| **Deblurring 3D-GS** (ECCV 2024) | MLP기반 공분산 조작 | Defocus + Motion | > 800 FPS | 본 논문: 실시간 렌더링 + 선택적 Gaussian 블러 조정 |
| **BAD-Gaussians** (ECCV 2024) | 3D Gaussian 타원체를 사용해 모션 블러의 물리적 이미징 프로세스를 명시적으로 모델링 | Motion | > 200 FPS | 200 FPS 이상, 학습 약 30분 (기존 NeRF 방법 10시간 이상) |
| **Deblur-GS** (I3D 2024) | 카메라 궤적 추정과 시간 샘플링의 공동 최적화로 모션 블러 모델링 | Motion | 실시간 | 카메라 궤적-시간 공동 최적화 |
| **DeblurGS** (2024) | Bézier 곡선으로 카메라 모션을 모델링하고 서브프레임 정렬 도입 | Motion | 실시간 | 복잡한 카메라 궤적 처리 |
| **EaDeblur-GS** (2024) | 이벤트 카메라 데이터를 통합하여 3DGS의 모션 블러 강건성을 높이며, Adaptive Deviation Estimator(ADE) 네트워크와 두 가지 새로운 손실 함수를 활용 | Motion | 실시간 | 이벤트 카메라 + 3DGS 통합 |
| **BSGS** (ACM MM 2025) | 2단계 프레임워크: (1) 카메라 포즈 개선으로 모션 왜곡 감소, (2) 글로벌 강체 변환으로 추가 보정 | Motion | 실시간 | 서브프레임 그래디언트 집계 전략 |
| **DiET-GS** (CVPR 2025) | 확산 사전지식과 이벤트 스트림을 활용한 모션 디블러링 3DGS, 이벤트 이중 적분(event double integral)으로 정확한 색상과 세밀한 디테일 달성 | Motion | 실시간 | Diffusion + Event stream 결합 |
| **MoBGS** (2025) | 흐릿한 단안 비디오에서 선명하고 고품질의 새로운 시공간 뷰를 end-to-end로 재구성하는 모션 디블러링 3DGS 프레임워크 | Motion (동적) | 실시간 | 최신 SOTA 동적 디블러링 NVS 방법 대비 모든 메트릭에서 대폭 개선, 4,800배 빠른 렌더링 |
| **CoherentGS** (2025) | 희소하고 흐릿한 이미지로부터 고품질 3D 재구성을 위한 이중 사전지식(dual-prior) 전략: 특화 디블러링 네트워크 + 확산 모델의 기하학적 사전지식 | Motion (Sparse) | — | 가장 도전적인 3-뷰 시나리오에서 GenFusion 대비 4.28 dB PSNR 향상 |
| **Unified 3DGS** (2025) | 모션 블러와 디포커스 블러를 동시 처리하는 통합 3DGS 프레임워크 | Motion + Defocus | — | PSNR 0.28 dB, SSIM 2.46%, LPIPS 39.88% 개선 |

### 연구 발전 흐름 요약

```
NeRF 기반 디블러링 (2022-2023)
  └─ Deblur-NeRF → DP-NeRF → BAD-NeRF → ExBluRF
        │
        ▼
3DGS 기반 디블러링 (2024~) ← Deblurring 3D-GS (본 논문) 이 핵심 전환점
  ├─ 공분산 조작: Deblurring 3D-GS
  ├─ 물리적 모션 모델링: BAD-Gaussians, Deblur-GS, DeblurGS
  ├─ 멀티모달: EaDeblur-GS, DiET-GS (이벤트 카메라)
  ├─ 동적 장면: MoBGS
  ├─ 희소 뷰: CoherentGS (diffusion prior 활용)
  └─ 통합 처리: Unified 3DGS (dual-blur)
```

---

## 참고자료 출처

1. **[논문 원문]** Lee, B., Lee, H., Sun, X., Ali, U., Park, E., "Deblurring 3D Gaussian Splatting," ECCV 2024, LNCS vol 15116, arXiv:2401.00834 — https://arxiv.org/abs/2401.00834
2. **[프로젝트 페이지]** https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/
3. **[GitHub]** https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting
4. **[SpringerLink]** https://link.springer.com/chapter/10.1007/978-3-031-73636-0_8
5. **[ECCV 2024 PDF]** https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07539.pdf
6. **[BAD-Gaussians]** Zhao et al., "BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting," ECCV 2024 — https://arxiv.org/abs/2403.11831
7. **[Deblur-GS]** Chen & Liu, "Deblur-GS: 3D Gaussian Splatting from Camera Motion Blurred Images," I3D 2024 — https://dl.acm.org/doi/abs/10.1145/3651301
8. **[EaDeblur-GS]** "EaDeblur-GS: Event assisted 3D Deblur Reconstruction with Gaussian Splatting," 2024 — https://arxiv.org/abs/2407.13520
9. **[MoBGS]** Bui et al., "MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video," 2025 — https://arxiv.org/abs/2504.15122
10. **[BSGS]** Zhao et al., "BSGS: Bi-stage 3D Gaussian Splatting for Camera Motion Deblurring," ACM MM 2025 — https://arxiv.org/abs/2510.12493
11. **[CoherentGS]** "Breaking the Vicious Cycle: Coherent 3D Gaussian Splatting from Sparse and Motion-Blurred Views," 2025 — https://arxiv.org/abs/2512.10369
12. **[DiET-GS]** Lee & Lee, "DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting," CVPR 2025 — https://diet-gs.github.io/
13. **[Unified 3DGS]** "Unified 3D Gaussian splatting for motion and defocus blur reconstruction," ScienceDirect 2025 — https://www.sciencedirect.com/science/article/pii/S2468502X25000531
14. **[HuggingFace Paper Page]** https://huggingface.co/papers/2401.00834
15. **[DAGS]** "Deblur-aware Gaussian splatting simultaneous localization and mapping," Knowledge-Based Systems 2025 — https://www.sciencedirect.com/science/article/abs/pii/S0950705125004137
