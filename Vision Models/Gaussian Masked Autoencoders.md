
# Gaussian Masked Autoencoders

> **논문 정보**
> - **제목:** Gaussian Masked Autoencoders
> - **저자:** Jathushan Rajasegaran et al.
> - **arXiv:** [2501.03229](https://arxiv.org/abs/2501.03229) (2025년 1월 6일)
> - **프로젝트 페이지:** https://brjathu.github.io/gmae/
> - **OpenReview:** https://openreview.net/forum?id=BoRmf8wDZ7 (ICLR 2025 제출)

---

## 1. 🔑 핵심 주장과 주요 기여 요약

이 논문은 Masked Autoencoders(MAE)와 Gaussian Splatting을 결합하는 방법을 탐구한다. 기존 MAE와 같은 재구성 기반 자기지도 학습(Self-Supervised Learning) 프레임워크는 우수한 시맨틱 추상화 능력을 갖추고 있으나, 명시적 공간 인식(spatial awareness)을 위한 훈련은 되어 있지 않다. 이에 GMAE는 시맨틱 추상화와 공간 이해를 **동시에** 학습하는 것을 목표로 한다.

**주요 기여:**

| 기여 | 설명 |
|------|------|
| **새로운 중간 표현** | MAE의 픽셀 공간 복원에 3D Gaussian 기반 중간 표현 도입 |
| **Zero-shot 공간 이해** | 추가 학습 없이 figure-ground 분리, edge detection 등 수행 |
| **선구적 연구** | Gaussian primitives를 단일 장면 재구성이 아닌 시각 표현 학습에 최초 적용 |
| **효율성** | 기존 MAE 대비 연산 오버헤드 단 1.5% 추가 |

저자들의 지식 범위 내에서, 최적화 기반 단일 장면 재구성을 넘어 이미지 표현 학습 프레임워크에 Gaussian primitives를 적용한 최초의 연구이다.

---

## 2. 🔍 해결하고자 하는 문제

### 2-1. 기존 MAE의 한계
주류 자기지도 학습 프레임워크인 MAE는 저수준 픽셀(low-level pixel) 수준에서 동작하는 반면, 이미지 합성 커뮤니티는 더 나은 생성적 시각 데이터 모델링을 위해 잠재적(latent) 중간 수준 표현으로 발전해왔다. GMAE는 이 두 접근법의 장점을 모두 취하고자 한다.

이 논문은 자기지도 학습을 통해 objectness, grouping, semantic structure와 같은 고수준 시맨틱 추상화를 2.1D layering과 함께 학습하는 것을 제안한다. 핵심 아이디어는 픽셀 기반 MAE에서 desirable한 중간 표현이 학습될 수 있도록 메커니즘을 설계하는 것이며, 3D Gaussians이 시맨틱 및 공간 이해로 이어질 수 있는 중간 이미지 표현의 훌륭한 후보임을 제시한다.

### 2-2. 공간적 표현의 필요성
Wang and Adelson의 연구에서 영감을 받아, 2.1D 레이어로 구성된 공간 인식 표현의 가장 단순한 형태가 서로 움직이는 물체를 표현하기에 충분함을 제시한다. 정적 이미지에서도 레이어드 표현(layered representation)은 세계의 구조에 대해 더 많은 것을 학습할 수 있게 한다. 따라서 단일 오브젝트 수준의 추상화를 갖는 레이어드 이미지 표현 학습이 목표로 설정된다.

---

## 3. 🏗️ 모델 구조

GMAE의 구조는 다음과 같다: **ViT 인코더**는 마스킹된 입력 이미지 패치를 처리하여 잠재 임베딩을 생성한다. **ViT 디코더**는 query 토큰을 기반으로 색상(color), 불투명도(opacity), 중심(center), 스케일(scale), 방향(orientation)을 포함하는 명시적 Gaussian 파라미터를 예측한다. 이 Gaussians들은 differentiable volume splatting을 통해 렌더링되어 원본 이미지를 재구성한다. 모델은 전체적으로 자기지도 학습 방식으로 end-to-end 학습된다.

```
입력 이미지
    ↓ (패치화 및 랜덤 마스킹)
[ViT 인코더]
    ↓ (잠재 임베딩)
[ViT 디코더 (경량)]
    ↓ (3D Gaussian 파라미터 예측)
[Differentiable Splatting Renderer]
    ↓
재구성 이미지 (픽셀 공간)
```

GMAE는 MAE 프레임워크를 기반으로 하되 3D Gaussian primitives를 이용한 중간 수준 표현으로 이를 강화한다. 아키텍처는 Vision Transformer(ViT) 인코더, 경량 ViT 디코더, 그리고 differentiable 렌더링 모듈로 구성된다.

---

## 4. 📐 제안하는 방법 (수식 포함)

### 4-1. 3D Gaussian 표현

각 3D Gaussian은 아래와 같이 정의된다:

$$\mathcal{G} = \{(\mu_i, \Sigma_i, c_i, \alpha_i)\}_{i=1}^{N}$$

여기서:
- $\mu_i \in \mathbb{R}^3$: Gaussian의 3D 중심 위치
- $\Sigma_i \in \mathbb{R}^{3\times3}$: 공분산 행렬 (스케일 + 방향으로 분해)
- $c_i \in \mathbb{R}^3$: RGB 색상
- $\alpha_i \in [0,1]$: 불투명도(opacity)

공분산 행렬은 스케일 행렬 $S$와 회전 행렬 $R$로 다음과 같이 분해된다:

$$\Sigma_i = R_i S_i S_i^\top R_i^\top$$

### 4-2. Gaussian Splatting (2D 투영)

3D Gaussian을 2D 이미지 평면에 투영하면 2D Gaussian이 된다:

$$\Sigma^{2D}_i = J W \Sigma_i W^\top J^\top$$

여기서 $W$는 world-to-camera 변환 행렬, $J$는 투영 야코비안이다.

### 4-3. 이미지 렌더링 (Splatting)

각 픽셀 $\mathbf{p}$에서의 색상은 depth 순으로 정렬된 Gaussian의 alpha-compositing으로 결정된다:

$$\hat{I}(\mathbf{p}) = \sum_{i \in \mathcal{N}} c_i \cdot \alpha_i \cdot \prod_{j < i}(1 - \alpha_j) \cdot \exp\left(-\frac{1}{2}(\mathbf{p} - \mu_i^{2D})^\top (\Sigma_i^{2D})^{-1} (\mathbf{p} - \mu_i^{2D})\right)$$

### 4-4. 학습 목적 함수 (MSE Loss)

MAE에서 직접 패치 픽셀을 예측하는 것과 달리, GMAE의 ViT 기반 디코더는 명시적 3D Gaussians(색상, 3D 중심 위치, 스케일, 방향)을 예측한다. 이 Gaussians들은 splatting differentiable renderer를 통해 이미지로 렌더링되고, 전체 모델은 픽셀 공간에서 MSE 손실(Mean Squared Error loss)을 사용해 학습된다.

$$\mathcal{L}_{recon} = \frac{1}{|\mathcal{M}|} \sum_{p \in \mathcal{M}} \left\| I(p) - \hat{I}(p) \right\|^2$$

여기서 $\mathcal{M}$은 마스킹된 패치 영역, $I(p)$는 원본 픽셀 값, $\hat{I}(p)$는 Gaussian splatting으로 렌더링된 픽셀 값이다.

모델은 재구성된 이미지의 픽셀 값과 마스킹된 위치에서의 원본 이미지 픽셀을 비교하는 MSE 손실로 학습된다. 이는 모델로 하여금 픽셀 데이터에 기반한 고충실도 재구성을 생성하는 동시에 Gaussian primitives가 표현하는 공간 구조를 학습하도록 장려한다.

---

## 5. ⚡ 성능 향상

### 5-1. 핵심 성능 지표

GMAE의 접근법은 표준 MAE 훈련 대비 무시할 수 있는 오버헤드만 추가한다 — splatting 추가로 연산 시간이 1.5%만 증가한다. 표현 학습 성능을 저해하지 않으면서 GMAE는 zero-shot 능력에서 상당한 성과를 거둔다.

GMAE는 분류(classification), 감지(detection), 분할(segmentation)과 같은 지도 학습 기반 표현 학습 태스크에서 높은 성능을 유지하며, 더 중요하게는 zero-shot 능력을 가능하게 한다. GMAE는 Gaussians을 픽셀 공간으로 렌더링하는 방식으로 직접적 지도 없이 픽셀 기반 이미지 재구성 손실을 사용하여 학습된 3D Gaussians의 중간 표현을 도입한다.

### 5-2. Gaussian 표현의 비균일성(Non-uniformity) 효과

이 재구성 손실을 통해 Gaussian 집합은 공간과 스케일 상에서 비균일하게 분포하는 법을 학습하며, 입력 이미지의 정보 밀도와 고주파 세부 사항을 동적으로 따른다. 깊이(depth) 자유도를 가짐으로써 모델은 오브젝트와 장면의 레이어링을 학습하며, 이는 어떠한 추가 학습 없이도 figure-ground separation, layering, edge detection을 가능케 한다.

### 5-3. Gaussian 수(N)의 영향

Gaussians의 수가 표현 품질에 미치는 영향을 조사한 결과, 더 많은 Gaussians을 추가할수록 성능이 향상되며 약 256개에서 평탄화(plateau)됨이 밝혀졌다. Gaussians의 스케일 변화도 성능에 영향을 주며, 다양한 스케일링 인자가 이미지의 고주파 세부 사항을 모델링하는 Gaussians의 능력에 어떤 영향을 미치는지에 대한 통찰이 제공된다.

---

## 6. 🌐 모델의 일반화 성능 향상 가능성 (중점)

### 6-1. Zero-shot 일반화 능력

GMAE는 3D Gaussian 표현을 활용하여 zero-shot 학습 능력을 가능하게 하며, 레이블된 데이터셋에 대한 광범위한 학습 없이도 효과적인 figure-ground 분할과 edge detection을 수행한다.

### 6-2. 비균일 표현의 일반화 기여

비균일 표현은 이미지의 정보 밀도와 상관관계가 있는 표현 밀도의 공간적 분포를 초래한다는 점에서 일반화에 기여한다.

정사각형 픽셀 패치처럼 기하학적으로 균일한 표현과는 달리, Gaussian의 크기, 위치, 이미지 상의 정보 분포가 동적으로 학습된다. 또한, Gaussian 기반 표현은 픽셀 공간으로 다시 매핑하는 splatting 이미지 렌더링 덕분에 end-to-end 학습에 적합하다. 따라서 MAE와 같은 자기지도 프레임워크 내에서 이러한 중간 수준 표현을 함께 학습할 수 있다.

### 6-3. 2.1D 레이어드 표현과 일반화

시각적 추론에 필요한 공간 이해의 종류에 대해, Wang and Adelson으로부터 영감을 받아 2.1D 레이어로 구성된 공간 인식 표현의 가장 단순한 형태가 서로 움직이는 물체를 표현하기에 충분함을 보인다. 정적 이미지에서도 레이어드 표현은 세계 구조에 대해 더 많은 것을 학습할 수 있게 한다.

### 6-4. Video-GMAE로의 확장과 일반화
마스킹된 시공간 오토인코딩 목적 함수를 differentiable 3D Gaussian splatting 렌더러 내에 패킹함으로써, Video-GMAE는 시간 구조화된 비디오 표현을 학습하기 위한 강력한 자기지도 패러다임을 확립한다. 이 표현들은 강력한 zero-shot 및 fine-tuned 추적 벤치마크를 제공할 뿐만 아니라, 동영상을 진화하는 3D 환경의 2D 투영으로 보는 물리적 직관과 정렬된 동적 장면 구조를 명시적으로 인코딩한다.

### 6-5. 의료 이미징으로의 일반화 (MedGMAE)

MedGMAE는 이러한 한계를 해결하기 위해 밀집 복셀 강도를 재구성하는 대신 희소한 3D Gaussian 표현을 학습하는 핵심 통찰에 기반한 3D 의료 이미지 사전학습을 위한 자기지도 프레임워크로 도입되었다.

MedGMAE 사전 학습은 3D Gaussian Splatting 재구성을 활용하여 CT 볼륨의 희소성(해부학적 장기가 공간의 11.8%만 차지)을 이용해 복셀 기반 MIM 방법 대비 99.25%의 파라미터 감소와 우수한 일관성을 달성한다.

---

## 7. ⚠️ 한계점

Video-GMAE와 같은 확장 모델에서도 나타나는 한계로서, 정적 카메라 가정(static camera assumption)에 대한 모델의 의존성과 Gaussians 수의 제약이 매우 동적인 환경이나 세밀한 디테일 설정에서의 적용 가능성을 제한할 수 있다.

추가적으로 확인된 한계:

- **Gaussian 수의 한계:** Gaussians의 수가 성능에 미치는 영향에서, 약 256개에서 성능이 평탄화됨이 밝혀졌으며, 이는 매우 복잡한 이미지에서 표현력의 상한이 존재할 수 있음을 시사한다.
- **2D 단일 이미지 한계:** 3D Gaussians를 사용하지만 단일 2D 이미지로부터 학습하므로, 진정한 3D 기하 이해에는 제약이 있다.
- **렌더링 해석 의존성:** Splatting 기반 렌더링은 단일 뷰포인트 가정에 기반하므로, 멀티뷰(multi-view) 환경으로의 직접 적용이 제한될 수 있다.

---

## 8. 📊 2020년 이후 관련 최신 연구 비교 분석

| 방법론 | 연도 | 핵심 특징 | GMAE와 차이점 |
|--------|------|-----------|--------------|
| **MAE** (He et al.) | 2022 | 랜덤 패치 마스킹 → 픽셀 재구성, ViT 기반 | 중간 표현 없음, 공간 이해 부족 |
| **BEiT** (Bao et al.) | 2022 | 이산 시각 토큰을 예측 목표로 사용 | 외부 dVAE 필요, 픽셀 직접 재구성 없음 |
| **SimMIM** (Xie et al.) | 2022 | 픽셀 직접 재구성, 간소화된 프레임워크 | 경량 디코더 없이 full ViT 사용 |
| **VideoMAE** (Tong et al.) | 2022 | 비디오로 MAE 확장, 시간적 마스킹 | 공간 이해나 Gaussian 표현 없음 |
| **GMAE** (본 논문) | 2025 | 3D Gaussian 중간 표현 + Splatting | Zero-shot 공간 이해, 비균일 표현 학습 |
| **Video-GMAE** (Baranwal et al.) | 2025 | GMAE를 비디오로 확장, 시공간 Gaussian | 동적 3D 구조 학습, 추적 태스크 강점 |
| **MedGMAE** | 2025 | 의료 3D 볼륨에 Gaussian MAE 적용 | CT 희소성 활용, 99% 파라미터 절감 |

기존 MAE는 컴퓨터 비전을 위한 확장 가능한 자기지도 학습기(scalable self-supervised learner)임을 보였으며, 방법은 단순하다: 입력 이미지의 랜덤 패치를 마스킹하고 누락된 픽셀을 재구성한다.

BERT 설계에서 영감을 받아, Masked Image Modeling(MIM)이 마스킹된 이미지를 재구성하는 방식으로 학습하기 위해 제안되었다. BEiT는 사전 학습된 tokenizer로 생성된 시각적 토큰을 예측하는 선구적 연구이다. SimMIM은 픽셀 RGB 값을 재구성 목표로 직접 사용하여 프레임워크를 단순화한다. MAE는 더 나은 학습 효율성을 위해 비대칭 인코더-디코더 구조를 제안한다.

---

## 9. 🔭 미래 연구에 미치는 영향과 고려할 점

### 9-1. 앞으로의 연구에 미치는 영향

최적화 기반 단일 장면 재구성을 넘어 이미지 표현 학습 프레임워크에 Gaussian primitives를 최초 적용한 이 연구는, GMAE가 이 방향의 추가 연구에 영감을 주고 고충실도 시각 데이터 모델링을 위한 차세대 기술 개발에 기여할 것으로 기대된다.

**구체적 영향 영역:**

1. **다운스트림 태스크 일반화:** GMAE는 ImageNet과 같은 벤치마크 데이터셋에서 모델 성능을 향상시킬 뿐만 아니라 실시간 객체 감지 및 분할을 포함한 컴퓨터 비전의 고급 응용 프로그램을 촉진한다.

2. **의료·과학 이미징 확장:** MedGMAE 사전 학습은 3D Gaussian Splatting 재구성을 활용하여 CT 볼륨의 희소성을 이용해 복셀 기반 MIM 방법 대비 우수한 일관성을 달성하며, 다운스트림 분할, 등록, 분류 태스크를 위한 강력한 인코더 표현을 학습할 수 있다.

3. **비디오 이해로의 확장:** Video-GMAE는 비디오 프레임 시퀀스를 시간적으로 진화하는 3D Gaussian primitives 집합으로 인코딩하는 자기지도 표현 학습 프레임워크이다. 아키텍처는 시공간 마스킹 오토인코더 백본과 differentiable Gaussian splatting 볼륨 렌더러를 기반으로 구축된다. 마스킹된 비디오를 이동하는 Gaussians을 통해 재구성하는 사전 학습 태스크를 구조화함으로써 모델은 기저 동적 3D 구조를 학습하는 귀납적 편향을 강제한다.

### 9-2. 앞으로 연구 시 고려할 점

**① Gaussian 수 및 최적화 전략**

Gaussians의 수가 약 256개에서 평탄화됨이 확인된 만큼, 태스크별로 Gaussian 수를 적응적으로 조정하거나, 더 복잡한 장면에서 표현력을 확장하는 연구가 필요하다.

**② 다중 카메라·멀티뷰 확장**

모델의 정적 카메라 가정(static camera assumption)에 대한 의존성과 Gaussians 수의 제약이 매우 동적인 환경이나 세밀한 디테일 설정에서의 적용 가능성을 제한하므로, 멀티뷰 학습 환경이나 동적 카메라 설정으로의 확장이 중요한 연구 방향이다.

**③ 더 나은 손실 함수 탐색**

현재 픽셀 수준 MSE 손실 외에, 지각적 손실(perceptual loss), SSIM, 또는 adversarial 손실을 결합하여 재구성 품질과 표현 학습 간의 균형을 더 잘 맞추는 연구가 필요하다.

**④ 대규모 데이터 및 모델 스케일링**

end-to-end 학습 가능한 3D Gaussians이 비균일 특성 덕분에 중간 수준 이미지 표현의 우수한 후보임을 제시하며, 대규모 이미지 컬렉션으로 MAE를 학습시켜 마스킹된 입력으로부터 전체 이미지를 재구성하는 방향에서, 더 대규모 데이터셋(예: JFT, LAION)과 더 큰 모델 스케일(ViT-L, ViT-H)에서의 검증이 중요하다.

**⑤ 다운스트림 태스크별 Gaussian 제약 연구**

분류·검출·분할 등 태스크별로 Gaussian의 의미론적 해석 가능성(interpretability)을 높이는 제약 조건(regularization)이나 손실 설계가 일반화 성능 향상에 기여할 수 있다.

**⑥ 비교 연구 강화**

주류 자기지도 학습 프레임워크들이 저수준 픽셀에서 동작하는 반면, 이미지 합성 커뮤니티는 더 나은 생성적 시각 데이터 모델링을 위한 잠재적 중간 수준 표현으로 발전해 왔는데, GMAE는 이 두 세계의 장점을 모두 취하고자 한다. 이에 따라 DALL-E, Stable Diffusion 등 생성 모델과의 결합 가능성과 비교 연구도 중요한 방향이다.

---

## 📚 참고 자료 (출처)

1. **Jathushan Rajasegaran et al.**, "Gaussian Masked Autoencoders", arXiv:2501.03229, January 2025.
   - arXiv: https://arxiv.org/abs/2501.03229
   - HTML 전문: https://arxiv.org/html/2501.03229v1
   - 프로젝트 페이지: https://brjathu.github.io/gmae/

2. **OpenReview (ICLR 2025 제출)**: https://openreview.net/forum?id=BoRmf8wDZ7

3. **Kaiming He et al.**, "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022. arXiv:2111.06377

4. **Themoonlight.io**, Literature Review – Gaussian Masked Autoencoders: https://www.themoonlight.io/en/review/gaussian-masked-autoencoders

5. **ResearchGate** – Gaussian Masked Autoencoders 다이어그램: https://www.researchgate.net/figure/...

6. **DEV Community** – "Unlocking Visual Intelligence: The Power of Gaussian Masked Autoencoders": https://dev.to/gilles_hamelink_ea9ff7d93/...

7. **Video-GMAE (Baranwal et al., Dec 2025)**, Emergent Mind: https://www.emergentmind.com/topics/video-gaussian-masked-autoencoders-video-gmae

8. **MedGMAE**, OpenReview: https://openreview.net/pdf/c8a8ac67bb2b3ddd8bcb4bc1e35d3a649f009034.pdf

9. **Survey on Masked Autoencoder for Visual Self-supervised Learning**, IJCAI 2023: https://www.ijcai.org/proceedings/2023/0762.pdf

10. **Masked Image Modeling: A Survey**, International Journal of Computer Vision (Springer, 2025): https://link.springer.com/article/10.1007/s11263-025-02524-1

11. **GitHub (비공식 구현)**: https://github.com/PatrickHua/GMAE
