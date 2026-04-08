
# RAIN-GS: Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting

> **논문 정보**
> - **제목:** Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting
> - **저자:** Jaewoo Jung, Jisang Han, Honggyu An, Jiwon Kang, Seonghoon Park, Seungryong Kim (KAIST CV Lab)
> - **arXiv ID:** 2403.09413 (v1: 2024.03.14 / v2: 2024.05.28)
> - **GitHub:** https://github.com/cvlab-kaist/RAIN-GS

---

## 1. 핵심 주장 및 주요 기여 (요약)

3D Gaussian Splatting (3DGS)은 실시간 Novel View Synthesis와 3D 재건에 뛰어난 성능을 보이지만, SfM(Structure-from-Motion)으로부터 도출되는 정확한 초기화에 크게 의존한다. 초기 포인트 클라우드의 품질이 저하되거나 무작위로 초기화된 포인트 클라우드를 사용하면 3DGS의 성능이 크게 하락한다.

논문의 핵심 발견은 **"제한된 가우시안 이동성(limited Gaussian transportability)"** 문제이다. 가우시안들은 이미지 측광 손실(photometric loss)만을 기반으로 최적화되므로, 최적화 과정이 현재 위치에서의 재건을 개선하도록 투영된 가우시안의 파라미터를 과적합시키는 경향이 있으며, 더 최적인 위치로 재배치되지 않는다. 이로 인해 노이즈가 있거나 무작위 초기화로 시작할 때 충분히 재건되지 않은 영역이 생성된다.

이를 해결하기 위해 논문은 **RAIN-GS** (Relaxing Accurate INitialization Constraint for 3D Gaussian Splatting)라는 새로운 최적화 전략을 제안한다. 이 전략은 세 가지 핵심 요소로 구성된다: 1) 큰 분산을 가진 희소 가우시안으로 시작하는 새로운 초기화 방법(SLV), 2) 렌더링 과정에서 활용되는 점진적 가우시안 저역통과 필터링, 3) 적응적 밀도 제어에서 활용되는 새로운 적응적 경계 확장 분할(ABE-Split) 알고리즘.

최적화 후, 무작위 포인트 클라우드에 RAIN-GS를 적용하면 SfM으로 학습된 3DGS와 동등하거나 더 나은 결과를 달성한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

이 논문은 3DGS 최적화 체계의 한계를 조사하여, 노이즈가 있거나 무작위 포인트 클라우드로 초기화할 때 왜 성능이 크게 저하되는지 밝힌다. 분석을 통해 3DGS 최적화의 핵심 한계인 **제한된 가우시안 이동성**을 식별한다.

무작위로 초기화된 포인트 클라우드로 학습할 때 3DGS는 일반적으로 PSNR에서 4~5 dB의 큰 성능 하락을 경험한다.

SfM 초기화는 씬의 대략적인 근사치를 제공함으로써 이후 세밀한 조정의 견고한 기반을 제공하는 반면, 무작위 초기화는 이러한 필수적인 저주파 정보를 포착하지 못한다. 주파수 영역 분석 및 1D 회귀 과제를 통해, 최적화 과정을 효과적으로 유도하기 위해 초기에 대략적인 분포를 학습하는 것의 중요성을 강조한다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) Sparse-Large-Variance (SLV) 초기화

기존 3DGS는 SfM에서 얻은 포인트 클라우드를 그대로 가우시안의 초기 위치로 사용한다. RAIN-GS는 **희소하고 큰 분산**을 가진 가우시안으로 시작한다.

각 3D 가우시안은 다음과 같이 정의된다:

$$G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

여기서 $\boldsymbol{\mu}$는 가우시안의 평균(위치), $\boldsymbol{\Sigma}$는 공분산 행렬이다. SLV 초기화에서는 $\boldsymbol{\Sigma}$의 초기 스케일을 의도적으로 크게 설정하여 가우시안이 더 넓은 영역을 커버하도록 한다.

3DGS에서 공분산 행렬은 스케일링 행렬 $S$와 회전 행렬 $R$로 분해된다:

$$\boldsymbol{\Sigma} = R S S^\top R^\top$$

SLV에서는 초기 스케일 $s$ 값을 크게 설정한다. 이를 통해 가우시안이 더 넓은 영역을 커버하며 전역적인 저주파 성분부터 학습하게 된다.

#### (B) 점진적 가우시안 저역통과 필터링 (Progressive Gaussian Low-Pass Filtering)

이 필터의 효과를 보여주는 시각화에서, 저역통과 필터와 스플랫된 2D 가우시안의 컨볼루션은 가우시안이 스플랫되는 영역을 확장시켜, 가우시안이 단순한 스플래팅보다 더 넓은 영역에 영향을 미치게 한다.

렌더링 과정에서의 컨볼루션은 다음과 같이 표현된다:

$$G'_i = G_i * \mathcal{F}_{\sigma_s}$$

여기서 $\mathcal{F}_{\sigma_s}$는 표준편차 $\sigma_s$를 가진 가우시안 저역통과 필터이다. 2D 투영 후의 유효 공분산은:

$$\boldsymbol{\Sigma}'_{\text{eff}} = \boldsymbol{\Sigma}'_i + \sigma_s^2 \mathbf{I}$$

훈련이 진행됨에 따라 $\sigma_s$를 점진적으로 감소시켜 점차 고주파 세부 사항을 학습하는 **Coarse-to-Fine** 방식을 구현한다:

$$\sigma_s(t) = \sigma_{\max} \cdot \left(1 - \frac{t}{T}\right)$$

여기서 $t$는 현재 학습 반복 횟수, $T$는 총 반복 횟수이다.

#### (C) Adaptive Bound-Expanding Split (ABE-Split)

업데이트 핵심은 가우시안들이 시점으로부터 더 멀리 있는 씬을 모델링할 수 있도록 원래 3DGS의 분할 알고리즘을 수정하는 것이며, 이 새로운 분할 알고리즘을 ABE-Split 알고리즘이라 명명한다.

기존 3DGS의 분할에서는 부모 가우시안의 위치 근처에서 자식 가우시안을 샘플링한다:

$$\boldsymbol{\mu}_{\text{child}} = \boldsymbol{\mu}_{\text{parent}} + \mathcal{N}(0, \boldsymbol{\Sigma}_{\text{parent}})$$

ABE-Split에서는 분할 시 자식 가우시안의 이동 범위를 확장한다:

$$\boldsymbol{\mu}_{\text{child}} = \boldsymbol{\mu}_{\text{parent}} + \alpha \cdot \mathcal{N}(0, \boldsymbol{\Sigma}_{\text{parent}})$$

여기서 $\alpha > 1$은 경계 확장 계수(bound-expanding factor)로, 가우시안이 초기화된 위치에서 더 멀리 이동할 수 있게 한다.

#### (D) 전체 최적화 손실 함수

기반이 되는 3DGS의 손실 함수는 다음과 같다:

$$\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$$

여기서 $\mathcal{L}\_1$은 픽셀별 L1 손실, $\mathcal{L}_{\text{D-SSIM}}$은 D-SSIM 손실, $\lambda = 0.2$이다. RAIN-GS는 이 손실 위에 SLV 초기화, 점진적 필터링, ABE-Split을 적용한다.

### 2.3 모델 구조

RAIN-GS의 전략은 가우시안들이 저주파 성분의 학습을 우선시하고, 초기화된 위치에서 더 멀리 이동할 수 있도록 하는 세 가지 핵심 요소로 구성된다: 1) SLV 초기화, 2) 점진적 가우시안 저역통과 필터링, 3) ABE-Split 알고리즘.

RAIN-GS는 3D Gaussian Splatting의 공식 구현 위에 구현된다.

모델 구조를 시각화하면 다음과 같다:

```
[Random/Sparse Point Cloud]
         ↓
[SLV 초기화: 희소 + 큰 분산]
         ↓
[훈련 루프]
  ├── [Progressive Low-Pass Filter: σ_s(t) 점진 감소]
  ├── [렌더링 (타일 기반 래스터라이제이션)]
  ├── [손실 계산: L1 + D-SSIM]
  └── [Adaptive Density Control + ABE-Split]
         ↓
[최종 3D 가우시안 표현]
```

### 2.4 성능 향상 및 한계

**성능 향상:**

여러 데이터셋에 대한 정량적·정성적 비교를 통해 전략의 효능을 입증하며, 무작위 포인트 클라우드로 학습된 RAIN-GS는 정확한 SfM 포인트 클라우드로 학습된 3DGS와 동등하거나 더 나은 성능을 달성한다.

세 가지 핵심 요소(SLV 초기화, 점진적 가우시안 저역통과 필터링, ABE-Split)를 통해 SfM 초기화된 3DGS와 동등하거나 더 나은 성능을 보이며, RAIN-GS는 초기 포인트 클라우드가 희소하기만 해도 적용 가능하므로 SfM/노이즈 포인트 클라우드에도 추가적으로 적용할 수 있다.

또한 논문은 희소 뷰 설정에서 3DGS를 학습시키는 RAIN-GS의 잠재적 확장을 탐색하며, 이러한 시나리오에서 SfM의 한계를 보완하는 능력을 보여준다. 이론적으로, 이 연구는 3D 가우시안 모델의 수렴에서 초기화의 핵심 역할과 강건한 최적화를 위한 저주파 성분 학습 우선화의 중요성을 밝힌다.

**한계:**

논문은 완화된 초기화가 3D 재건의 장기적 안정성이나 강건성에 미치는 영향을 탐구하지 않는다. 이 접근 방식에서 발생할 수 있는 트레이드오프와 잠재적 문제를 완전히 이해하기 위한 추가 연구가 필요할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

RAIN-GS는 SfM에서 얻은 정확한 포인트 클라우드에 대한 엄격한 의존성을 효과적으로 제거하여, 정확한 포인트 클라우드 획득이 어려운 시나리오에서 3DGS의 새로운 가능성을 열어준다.

실용적으로, 정확하게 초기화된 포인트 클라우드의 엄격한 요구 사항을 완화함으로써 RAIN-GS는 그러한 초기화를 획득하는 것이 어렵거나 불가능한 시나리오로 3DGS의 적용 가능성을 넓힌다.

구체적인 일반화 가능성 영역은 다음과 같다:

1. **동적 장면(Dynamic Scenes)으로 확장:**
   RAIN-GS는 무작위 초기화된 포인트 클라우드로 학습하고 SfM 포인트 클라우드로 학습된 3DGS와 동등한 성능을 달성하는 전략을 제안한다. 이 아이디어를 동적 장면, 특히 단안 동적 장면으로 확장하는 연구도 활발히 진행되고 있다.

2. **의료 영상 및 특수 도메인:**
   수술 장면의 실시간 3D 재건에서도 3DGS는 초기화를 위해 SfM으로 생성된 정확한 포즈와 포인트 클라우드에 의존하지만, SfM이 최소한의 텍스처와 측광 불일치로 인해 수술 장면에서 정확한 카메라 포즈와 기하를 복원하는 데 실패한다.

3. **희소 뷰(Sparse-View) 설정:**
   저자들은 RAIN-GS가 Novel View Synthesis, 의미론적 가우시안 스플래팅, 구조 인식 3D 가우시안 스플래팅, 고품질 동적 3D 재건 등 여러 응용에서의 성능 향상을 가져온다는 것을 보여준다.

4. **SfM-Free 파이프라인과의 결합:**
   가우시안 스플래팅을 위한 다양한 초기화 전략을 조사하고 NeRF 모델로부터의 구조 증류를 통해 SfM 데이터 의존성을 우회하는 방법을 연구한 결과, 무작위 초기화가 신중하게 설계된다면 훨씬 더 나은 성능을 보일 수 있으며, SfM 초기화와 동등하거나 때로는 더 나은 결과를 달성할 수 있다는 것을 보인다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 연도 | 핵심 접근법 | SfM 의존성 |
|------|------|------------|------------|
| NeRF (Mildenhall et al.) | 2020 | 암시적 신경 표현, 볼류메트릭 렌더링 | 필요 |
| 3DGS (Kerbl et al.) | 2023 | 명시적 3D 가우시안, 실시간 래스터화 | 강하게 필요 |
| **RAIN-GS (Jung et al.)** | **2024** | **SLV+저역통과 필터+ABE-Split** | **불필요** |
| COLMAP-Free 3DGS (Fu et al.) | 2024 | 포즈 없이 3DGS 학습 | 불필요 |
| InstantSplat (Fan et al.) | 2024 | 기하 기반 모델로 SfM 대체 | 불필요 |
| Does GS Need SfM? (2024) | 2024 | NeRF 구조 증류로 SfM 대체 | 선택적 |

InstantSplat은 희소 시점 이미지에서 밀집된 3D 모델을 빠르게 재건하는 방법으로, 전통적인 SfM 포즈와 3D-GS의 복잡한 적응적 밀도 제어 전략에 의존하는 대신 기하학적 기반 모델(MASt3R)을 활용하여 카메라 포즈와 3D 씬을 공동 최적화한다.

COLMAP-Free 3DGS(CF-3DGS)는 알려진 카메라 파라미터 없이 Novel View Synthesis를 위한 방법을 제안하며, 포즈 추정에서 더 높은 강건성과 Novel View Synthesis에서 더 나은 품질을 달성한다.

**RAIN-GS의 차별점:**
- 기존 SfM-Free 연구들이 주로 **외부 모델(NeRF, 기하 기반 모델)**로 SfM을 대체하는 반면, RAIN-GS는 **최적화 전략 자체**를 개선하여 랜덤 초기화에서도 수렴 가능하도록 한다.
- 추가적인 외부 모델 없이도 SfM 초기화와 동등한 성능을 달성한다.
- 기존 3DGS 코드베이스에 간단한 수정만으로 적용 가능하다.

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 앞으로의 연구에 미치는 영향

SLV 초기화, 점진적 가우시안 저역통과 필터 제어, ABE-Split 알고리즘의 결합으로 3D 가우시안이 저주파 성분 학습을 우선시하고 초기화된 위치에서 더 멀리 이동하도록 성공적으로 유도하며, SfM의 정확한 포인트 클라우드에 대한 엄격한 의존을 효과적으로 제거하여 3DGS의 새로운 가능성을 열어준다.

**1) 실용적 적용 범위 확대:**
- SfM이 실패하는 환경(수중, 의료 영상, 반복 텍스처, 야간 등)에서의 3DGS 적용 가능성이 열린다.
- 실시간 or 스트리밍 3D 재건 시스템에서 전처리 비용 절감에 기여한다.

**2) 동적 장면 재건으로의 확장:**
모노큘러 동적 장면 재건을 위한 새로운 최적화 전략이 정확한 초기화의 요구 사항을 완화하고 무작위 포인트 클라우드를 초기화로 지원하며, 이는 특히 복잡한 동적 장면에서 정확한 초기화를 얻기 어려운 경우 의존성을 우회할 수 있게 한다.

**3) 주파수 도메인 분석의 중요성 재조명:**
- 저주파 성분 우선 학습의 중요성은 NeRF에서의 위치 인코딩 연구, BARF 등과 연결되어 3D 표현 학습의 일반적인 원리로 발전될 수 있다.

### 5.2 앞으로 연구 시 고려할 점

**1) 계산 비용 트레이드오프:**
RAIN-GS는 추가적인 필터링 및 분할 연산으로 인해 기존 3DGS보다 약간 더 많은 계산 비용이 발생할 수 있다. 향후 연구에서는 계산 효율성을 유지하면서 성능을 개선하는 방향을 고려해야 한다.

**2) 하이퍼파라미터 민감도:**
점진적 필터 스케줄( $\sigma_s(t)$ )과 ABE-Split의 경계 확장 계수( $\alpha$ )에 대한 민감도 분석이 더 필요하다. 다양한 씬 복잡도에 따른 자동 하이퍼파라미터 조정 방법 연구가 필요하다.

**3) 장기 안정성 문제:**
논문은 완화된 초기화가 3D 재건의 장기적 안정성이나 강건성에 미치는 영향을 탐구하지 않는다. 이 접근 방식에서 발생할 수 있는 트레이드오프와 잠재적 문제를 완전히 이해하기 위한 추가 연구가 필요하다.

**4) 카메라 포즈 추정과의 통합:**
현재 RAIN-GS는 정확한 카메라 포즈를 가정한다. COLMAP-Free 방법들과의 통합으로 완전히 SfM-Free한 파이프라인 구성이 가능하다.

**5) 대규모 씬에서의 검증:**
SfM 기술이 수렴하기 어려운 씬들이 존재하며, 이런 상황에서는 초기화를 위한 정확한 포인트 클라우드를 얻을 수 없어 가우시안 스플래팅을 사용한 재건에 어려움이 생긴다. 대규모 야외 환경이나 텍스처가 없는 씬에서의 검증이 더 필요하다.

**6) 다른 3DGS 변형과의 통합:**
RAIN-GS를 semantic-aware 가우시안 스플래팅, structure-aware 3D 가우시안 스플래팅, 고품질 동적 3D 재건에 적용하는 방향이 이미 제시되어 있으며, 2D-GS, Scaffold-GS, Mip-Splatting 등 최신 변형 모델에도 적용 가능성이 높다.

---

## 📚 참고 자료 (출처)

1. **arXiv 원문**: Jung, J. et al. "Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting." *arXiv:2403.09413*, 2024. https://arxiv.org/abs/2403.09413
2. **arXiv HTML (v2)**: https://arxiv.org/html/2403.09413v2
3. **GitHub (공식 코드)**: https://github.com/cvlab-kaist/RAIN-GS
4. **OpenReview**: https://openreview.net/forum?id=R9lgWYE508
5. **EmergentMind 분석**: https://www.emergentmind.com/papers/2403.09413
6. **AI Models FYI**: https://www.aimodels.fyi/papers/arxiv/relaxing-accurate-initialization-constraint-3d-gaussian-splatting
7. **ADS Abstract**: https://ui.adsabs.harvard.edu/abs/2024arXiv240309413J/abstract
8. **MDPI (모노큘러 동적 확장 연구)**: "Relaxing Accurate Initialization for Monocular Dynamic Scene Reconstruction with Gaussian Splatting." *Applied Sciences*, 2026. https://www.mdpi.com/2076-3417/16/3/1321
9. **HuggingFace (관련 논문 "Does GS need SfM?")**: https://huggingface.co/papers/2404.12547
10. **COLMAP-Free 3DGS**: Fu, Y. et al. "COLMAP-Free 3D Gaussian Splatting." *CVPR 2024*.
11. **InstantSplat**: Fan, Z. et al. "InstantSplat: Sparse-view Gaussian Splatting in Seconds." *arXiv:2403.20309*, 2024. https://instantsplat.github.io/
12. **AttentionGS (후속 연구)**: "AttentionGS: Towards Initialization-Free 3D Gaussian Splatting via Structural Attention." *arXiv:2506.23611*, 2025.
13. **3D Gaussian Splatting 원논문**: Kerbl, B. et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM TOG 42(4)*, 2023.
14. **저자 홈페이지**: https://hg010303.github.io/ (Honggyu An)
