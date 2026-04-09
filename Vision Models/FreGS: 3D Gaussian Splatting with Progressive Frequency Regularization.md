
# FreGS: 3D Gaussian Splatting with Progressive Frequency Regularization 

> **논문 정보**
> - **저자:** Jiahui Zhang, Fangneng Zhan, Muyu Xu, Shijian Lu, Eric Xing
> - **발표:** CVPR 2024 (IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21424–21433)
> - **arXiv:** [2403.06908](https://arxiv.org/abs/2403.06908)
> - **IEEE Xplore:** [10.1109/CVPR52733.2024.02024](https://ieeexplore.ieee.org/document/10655777/)
> - **프로젝트 페이지:** [rogeraigc.github.io/FreGS-Page](https://rogeraigc.github.io/FreGS-Page/)

---

## 1. 핵심 주장 및 주요 기여 요약

3D Gaussian Splatting은 실시간 Novel View Synthesis(NVS)에서 매우 인상적인 성능을 달성했지만, Gaussian Densification 과정에서 **과재구성(over-reconstruction)** 문제가 자주 발생한다. 이는 분산이 높은 이미지 영역이 소수의 큰 Gaussian으로만 덮여 렌더링된 이미지에 흐림(blur)과 아티팩트를 유발하는 현상이다.

이 논문의 핵심 기여는 세 가지로 요약된다:


1. **최초의 주파수 공간 기반 접근**: over-reconstruction 문제를 주파수 공간의 정규화로 해결하는 최초의 시도이다.
2. **Frequency Annealing 기법**: 저주파에서 고주파로 점진적으로 정규화를 수행하여 충실한 coarse-to-fine Gaussian Densification을 달성한다.
3. **다중 벤치마크에서 우수한 성능**: 여러 벤치마크에서 3D-GS를 일관되게 능가하는 NVS 성능을 보인다.


---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능

### 2-1. 해결하고자 하는 문제: Over-Reconstruction

3D-GS는 NeRF의 부피 렌더링 대신 효율적인 Splatting을 통해 3D Gaussian을 2D 평면에 직접 투영하여 실시간 렌더링을 보장하지만, Gaussian Densification 과정에서 over-reconstruction이 빈번히 발생한다. 고분산 이미지 영역이 소수의 큰 Gaussian으로만 표현되고, 이는 렌더링된 2D 이미지에서 블러와 아티팩트로 나타난다. 이 over-reconstruction은 렌더링 이미지와 ground truth의 **주파수 스펙트럼 불일치**로도 명확하게 관찰된다.

이러한 관찰을 기반으로, over-reconstruction이 주파수 스펙트럼의 불일치로 명확히 나타남을 확인하고 Fourier 공간에서 주파수 신호를 정규화함으로써 over-reconstruction을 해결하는 혁신적인 3D Gaussian Splatting 기법, FreGS를 설계하였다.

---

### 2-2. 3D Gaussian Splatting 기반 표현 (배경)

3D-GS에서 각 3D Gaussian은 다음과 같이 정의된다:

$$
G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})}
$$

여기서 $\boldsymbol{\mu}$는 중심(위치), $\boldsymbol{\Sigma}$는 공분산 행렬이다. 공분산 행렬은 회전 행렬 $R$과 스케일 행렬 $S$로 분해된다:

$$
\boldsymbol{\Sigma} = RSS^\top R^\top
$$

3D Gaussian은 Structure-from-Motion(SfM)으로 초기화된다. Splatting 이후 2D Gaussian을 얻고, 표준 $\alpha$-블렌딩을 통해 렌더링한다.

렌더링 픽셀 컬러는 다음과 같이 계산된다:

$$
\hat{I}(\mathbf{p}) = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j < i}(1-\alpha_j)
$$

---

### 2-3. 제안 방법: Progressive Frequency Regularization

#### (1) 주파수 정규화 (Frequency Regularization)

주파수 스펙트럼 $\hat{F}$와 $F$는 각각 렌더링 이미지 $\hat{I}$와 ground truth $I$에 Fourier 변환을 적용하여 생성된다. 주파수 정규화는 Fourier 공간에서 진폭(amplitude) $|F(u,v)|$와 위상(phase) $\angle F(u,v)$의 차이를 정규화함으로써 달성된다.

2D Discrete Fourier Transform(DFT)은 다음과 같이 정의된다:

$$
F(u,v) = \sum_{x=0}^{H-1}\sum_{y=0}^{W-1} I(x,y) \cdot e^{-j2\pi\left(\frac{ux}{H}+\frac{vy}{W}\right)}
$$

렌더링 이미지 $\hat{I} \in \mathbb{R}^{H \times W \times C}$와 ground truth 이미지 $I \in \mathbb{R}^{H \times W \times C}$에 2D DFT를 적용하여 각각의 주파수 표현 $\hat{F}$와 $F$를 얻는다.

주파수 정규화 손실은 **진폭 손실**과 **위상 손실**로 구성된다:

$$
\mathcal{L}_{\text{amp}} = \|H \odot |\hat{F}(u,v)| - H \odot |F(u,v)|\|_1
$$

$$
\mathcal{L}_{\text{phase}} = \|H \odot \angle\hat{F}(u,v) - H \odot \angle F(u,v)\|_1
$$

$$
\mathcal{L}_f = \mathcal{L}_{\text{amp}} + \mathcal{L}_{\text{phase}}
$$

여기서 $H$는 주파수 필터 마스크(저역 통과 필터 $H_l$ 또는 동적 고역 통과 필터 $H_h$)이다.

이 프레임워크는 렌더링 이미지와 ground truth의 스펙트럼 표현 사이의 진폭 및 위상 성분 불일치를 모두 최소화하며, 이 이중 초점 정규화는 장면의 기하학적 충실도와 텍스처 뉘앙스를 모두 잘 포착하도록 보장한다.

#### (2) Frequency Annealing (점진적 주파수 어닐링)

FreGS는 저주파에서 고주파 신호로 정규화를 점진적으로 어닐링하는 coarse-to-fine Gaussian Densification을 수행한다. 이는 저주파와 고주파 신호가 각각 큰 스케일(전역 패턴과 구조 — 모델링이 쉬운)과 작은 스케일(로컬 디테일 — 모델링이 어려운) 특징을 인코딩하는 합리적 근거에 기반한다.

새로운 Frequency Annealing 기법은 점진적 주파수 정규화를 달성하도록 설계된다. 저역 통과 필터 $H_l$와 동적 고역 통과 필터 $H_h$를 사용하여 저주파에서 고주파 성분을 점진적으로 활용하여 coarse-to-fine Gaussian Densification을 수행한다.

동적 고역 통과 필터의 컷오프 주파수는 학습 과정에서 점진적으로 확장된다:

$$
r_t = r_{\min} + \frac{t}{T}(r_{\max} - r_{\min})
$$

여기서 $t$는 현재 학습 step, $T$는 총 step, $r_{\min}$과 $r_{\max}$는 최소/최대 컷오프 반경이다.

#### (3) 최종 손실 함수

점진적 주파수 정규화 손실 $\mathcal{L}_f$는 3D-GS의 표준 손실 항인 L1 픽셀 손실 및 D-SSIM 항과 결합된다. D-SSIM은 렌더링 이미지와 ground truth 이미지 사이의 구조적 유사성을 향상시키는 데 도움을 준다.

$$
\mathcal{L}_{\text{total}} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}} + \mu \mathcal{L}_f
$$

점진적 주파수 정규화는 $\hat{I}$와 $I$ 사이의 픽셀 단위 손실을 보완한다.

---

### 2-4. 모델 구조 개요

FreGS의 개요는 다음과 같다. 3D Gaussian은 Structure-from-Motion으로 초기화된다. 3D Gaussian을 Splatting한 후 2D Gaussian을 얻고 표준 $\alpha$-블렌딩으로 렌더링한다. 주파수 정규화는 Fourier 공간에서 진폭 $|F(u,v)|$와 위상 $\angle F(u,v)$의 불일치를 정규화함으로써 달성된다.

전체 파이프라인을 도식화하면:

```
SfM Point Cloud
      ↓
3D Gaussian 초기화 (μ, Σ, c, α)
      ↓
Splatting (3D→2D 투영)
      ↓
α-blending 렌더링 → Î (렌더링 이미지)
      ↓
[픽셀 공간] L1 + D-SSIM 손실
      ↓
[주파수 공간] DFT(Î) → F̂
              DFT(I)  → F
              Amplitude/Phase 불일치 최소화
              (Low-Pass Hl + Dynamic High-Pass Hh via Frequency Annealing)
      ↓
Gaussian Densification 가이드 (Coarse-to-Fine)
```

---

### 2-5. 성능 향상

FreGS는 Mip-NeRF360, Tanks & Temples, Deep Blending을 포함한 여러 실세계 데이터셋에서 일관되게 우수한 정량적 성능을 달성하며, 3D-GS, Mip-NeRF360, INGP-Base, INGP-Big, Plenoxels 등의 최신 방법을 능가한다. 예를 들어 Mip-NeRF360 데이터셋에서 FreGS는 PSNR 27.85, SSIM 0.826, LPIPS 0.209를 달성하며, 이는 3D-GS의 PSNR 27.21, SSIM 0.815, LPIPS 0.214보다 향상된 수치이다.

FreGS는 3D-GS와 유사한 수의 Gaussian으로도 일관되게 우월한 성능을 보이며, 이는 계산 오버헤드나 메모리 사용량을 크게 증가시키지 않고 렌더링 품질을 향상시키는 실용적 발전임을 시사한다.

| 방법 | Mip-NeRF360 PSNR | SSIM | LPIPS |
|------|-----------------|------|-------|
| 3D-GS | 27.21 | 0.815 | 0.214 |
| **FreGS** | **27.85** | **0.826** | **0.209** |

---

### 2-6. 한계점

점진적 주파수 정규화 방식은 세심하게 설계된 손실 함수에 의존하므로, 실제 적용 시 튜닝이 어려울 수 있다. 더 자동화되거나 적응형 접근 방식을 탐색하는 것이 흥미로울 것이다.

FreGS는 LF와 HF 성분을 분리하는 완전 감독된 Fourier 공간 손실을 사용하며, HF 항은 초기 LF 중심 단계 이후에만 도입된다. 그러나 저변동 LF 구조 학습이 상대적으로 단순하기 때문에, 이 감독이 HF 디테일에 편향되어 HF 과적합 약점으로 되돌아갈 수 있다.

또한 FreGS가 동적 또는 비정적 장면을 어떻게 처리하는지, 그리고 시간적 일관성을 지원하기 위한 잠재적 수정이 있는지에 대한 질문이 남아 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재 FreGS의 일반화 기여

FreGS는 주파수 불일치를 점진적으로 최소화함으로써 Gaussian Densification 과정을 효과적으로 안내한다. 특히 over-reconstructed 영역에서 렌더링 이미지의 주파수 내용이 ground truth와 일치하지 않는 영역에 더 많은 Gaussian의 생성을 촉진한다.

FreGS는 'Garden', 'Room', 'Train', 'Truck', 'Drjohnson'과 같은 다양한 장면에서 다른 최신 방법들에 비해 아티팩트가 현저히 적고 더 세밀한 디테일의 렌더링 이미지를 생성하며, 이는 블러를 완화하고 복잡한 장면 구조를 보존하는 더 정제된 Gaussian Densification 과정에서 비롯된다.

Mip-NeRF360, Tanks-and-Temples, Deep Blending 등 다양하게 채택된 여러 벤치마크에서의 실험을 통해 FreGS가 일관되게 최신 기술을 능가함을 보인다.

### 3-2. 희소 뷰(Sparse-view) 환경에서의 일반화 문제

희소 뷰 3DGS는 희소한 훈련 뷰의 크게 변동하는 고주파(HF) 디테일에 과적합되는 경향이 있어 고품질 새 뷰 재구성에 상당한 어려움을 보인다. 주파수 정규화는 유망한 접근 방식이 될 수 있지만, Fourier 변환에 대한 일반적인 의존성은 어려운 매개변수 튜닝과 해로운 HF 학습에 대한 편향을 야기한다.

FreGS는 Fourier 도메인에서 보조적 HF 감독을 도입하여 밀집 뷰(dense-view) 3DGS의 과평활화 아티팩트를 완화하지만, 희소 뷰에서의 효과는 여전히 불명확하다.

### 3-3. 일반화 향상을 위한 후속 연구: DWTGS

DWTGS는 추가적인 공간 감독을 제공하는 웨이블릿 공간 손실을 활용하여 주파수 정규화를 재고하는 프레임워크를 제안한다. 구체적으로 여러 DWT 레벨에서 저주파(LF) LL 서브밴드만 감독하고, HF HH 서브밴드에 자기지도 방식으로 희소성을 적용한다. 이 LF 중심 전략은 일반화를 향상시키고 HF 환각을 줄여 Fourier 기반 대응 방법을 일관되게 능가한다.

FreGS는 점진적 주파수 정규화를 사용하여 Densification 중 over-reconstruction 문제를 해결하지만, 이러한 접근 방식은 3D 표현 자체의 명시적 주파수 분해가 부족하여 LOD 렌더링, 효율적 스트리밍, 다양한 주파수 성분 제어 같은 작업을 수행하기 어렵게 한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. NeRF 계열 (배경 기술)

3D 컴퓨터 비전과 Novel View Synthesis 분야는 Neural Radiance Fields(NeRF)와 그 파생 기법의 발전으로 크게 진보했다. 특히 3D Gaussian Splatting은 훈련 효율성과 렌더링 품질의 균형을 맞추면서 실시간 NVS를 가능하게 하는 유망한 대안으로 부상했다.

| 연구 | 연도 | 특징 | FreGS와의 관계 |
|------|------|------|---------------|
| NeRF (Mildenhall et al.) | 2020 | 암시적 신경 표현, 볼륨 렌더링 | FreGS의 배경 기술 |
| InstantNGP | 2022 | 다중 해상도 해시 테이블로 빠른 학습 | 3D-GS의 선행 연구 |
| 3D-GS (Kerbl et al.) | 2023 | 이방성 3D Gaussian + 실시간 Splatting | FreGS가 개선 대상으로 삼는 기반 모델 |
| **FreGS** | **2024** | **Fourier 주파수 정규화** | **본 논문** |
| AbsGS | 2024 | 절대값 기반 기울기로 세밀한 Densification | 유사 문제 접근 |
| BAGS | 2024 | 공간 블러 커널 학습 | HF 복원 관점 유사 |
| Frequency-Aware GSD | 2025 | 라플라시안 밴드 기반 주파수 분해 | FreGS를 발전시킨 후속 연구 |
| DWTGS | 2025 | 웨이블릿 공간 손실로 일반화 개선 | FreGS 한계 보완 |

### 4-2. Frequency-Aware Gaussian Splatting Decomposition (2025)

이 연구는 LOD(Level of Detail)를 단순한 계층적 희소화가 아닌 표현 자체의 **주파수 분할**로 재정의한다. 주요 기여는 이미지 공간 라플라시안 밴드에 의해 구동되는 3D-GS의 주파수 인식 분해, 서명된 잔차 색상 및 밴드 분리 정규화를 포함한 점진적 훈련 전략, 그리고 LOD/주파수 인식 방법 중 SOTA 품질을 보이는 실험이다.

이 방법은 입력 이미지의 주파수 내용을 기반으로 Gaussian 그룹을 명시적으로 정의하고, 모든 주파수 그룹을 순차적으로 고정하는 방식 대신 점진적으로 공동 학습하며, 그룹 간의 명확한 스펙트럼 경계를 강제하는 전용 주파수 도메인 정규화를 사용한다.

### 4-3. DWTGS (2025)

PGDGS는 FreGS를 희소 뷰 설정에 적용했지만, 여전히 Fourier 공간에서 작동하여 HF 학습에 편향되어 과적합되기 쉬운 손실 가중치의 광범위한 튜닝이 필요하다. 반면 DWTGS는 희소 뷰 3DGS의 HF 과적합을 효과적으로 완화하는 LF 중심의 웨이블릿 공간 손실을 활용한다.

---

## 5. 연구에 미치는 영향과 향후 고려사항

### 5-1. 앞으로의 연구에 미치는 영향

FreGS는 over-reconstruction 문제를 주파수 공간의 정규화로 해결하는 혁신적인 3D Gaussian Splatting 프레임워크이며, 이는 **3D Gaussian Splatting의 over-reconstruction 문제를 스펙트럼 관점에서 다루는 최초의 시도**이다.

FreGS가 도입한 스펙트럼 정규화 원리는 3D 메쉬 처리나 포인트 클라우드 노이즈 제거와 같은 다른 도메인으로 확장될 수 있는 잠재력을 갖는다.

FreGS 모델은 신경 렌더링 분야의 3D 뷰 합성에서 중요한 진전을 나타낸다. Gaussian Splatting과 점진적 주파수 정규화를 결합함으로써 새로운 시점에서 고품질·고해상도 3D 장면 이미지를 생성할 수 있다. 다양한 벤치마크에서의 강력한 성능은 가상·증강 현실, 자율 주행, 디지털 콘텐츠 창작 등 광범위한 응용 가능성을 시사한다.

### 5-2. 향후 연구 시 고려할 점

1. **손실 함수 자동화/적응화**
점진적 주파수 정규화 방식은 세심하게 설계된 손실 함수에 의존하므로, 실제 적용 시 튜닝이 어려울 수 있다. 더 자동화되거나 적응적인 접근 방식 탐색이 필요하다.

2. **희소 뷰 환경 대응**
DWTGS의 LF 중심 전략이 일반화를 향상시키고 HF 환각을 줄인다는 점에서, 희소 뷰 환경에서 FreGS를 안전하게 적용하려면 Fourier 공간의 HF 편향 문제를 더 신중하게 다루어야 한다.

3. **실시간·인터랙티브 렌더링 지원**
현재 FreGS의 구현은 오프라인 렌더링에 초점을 맞추고 있어, 많은 실세계 응용 프로그램에서 요구되는 실시간 또는 인터랙티브 성능을 지원하기 위한 최적화가 필요하다.

4. **동적 장면 확장**
동적 또는 비정적 장면 처리와 시간적 일관성 지원을 위한 잠재적 수정이 중요한 미래 연구 과제이다.

5. **주파수 분해의 3D 표현 통합**
3D-GS는 주파수 개념이 없고 수백만 개의 Gaussian이 비구조적 풀을 이루고 있으며, FreGS는 점진적 주파수 정규화를 사용하지만 3D 표현 자체의 명시적 주파수 분해가 부족하다. 따라서 LOD 렌더링과 효율적 스트리밍 같은 응용을 위해 명시적 주파수 분해를 3D 표현 수준에서 통합하는 방향이 유망하다.

6. **Total Variation 정규화와의 상호보완성**
FreGS는 주파수 정규화 외에도 Gaussian 위치와 특징의 공간적 일관성을 촉진하기 위한 Total Variation(TV) 정규화를 사용한다. 이 두 정규화의 최적 균형과 시너지를 탐구하는 연구가 의미 있을 것이다.

---

## 📚 참고 자료 (출처)

1. **arXiv 원문**: Zhang, J., Zhan, F., Xu, M., Lu, S., & Xing, E. (2024). *FreGS: 3D Gaussian Splatting with Progressive Frequency Regularization.* arXiv:2403.06908. https://arxiv.org/abs/2403.06908
2. **CVPR 2024 Open Access**: Zhang et al., *FreGS: 3D Gaussian Splatting with Progressive Frequency Regularization*, CVPR 2024, pp. 21424–21433. https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_FreGS_...
3. **IEEE Xplore**: https://ieeexplore.ieee.org/document/10655777/
4. **프로젝트 페이지**: https://rogeraigc.github.io/FreGS-Page/
5. **ArXiv HTML 상세 버전**: https://arxiv.org/html/2403.06908v2
6. **EmergentMind 리뷰**: https://www.emergentmind.com/papers/2403.06908
7. **Moonlight 리뷰**: https://www.themoonlight.io/en/review/fregs-3d-gaussian-splatting-with-progressive-frequency-regularization
8. **Liner 리뷰**: https://liner.com/review/fregs-3d-gaussian-splatting-with-progressive-frequency-regularization
9. **AI Models FYI**: https://www.aimodels.fyi/papers/arxiv/fregs-3d-gaussian-splatting-progressive-frequency-regularization
10. **ADS Abstract**: https://ui.adsabs.harvard.edu/abs/2024arXiv240306908Z/abstract
11. **후속 연구 — Frequency-Aware GSD (2025)**: https://arxiv.org/html/2503.21226
12. **후속 연구 — DWTGS (2025)**: https://arxiv.org/html/2507.15690
13. **후속 연구 — ARGS (2025)**: https://arxiv.org/html/2508.21344
14. **ResearchGate**: https://www.researchgate.net/publication/384168373_FreGS_...
