
# Light-A-Video: Training-free Video Relighting via Progressive Light Fusion

> **논문 정보**
> - **저자**: Yujie Zhou\*, Jiazi Bu\*, Pengyang Ling\*, Pan Zhang†, Tong Wu, Qidong Huang, Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Anyi Rao, Jiaqi Wang, Li Niu†
> - **게재**: ICCV 2025 (Accepted)
> - **arXiv**: [arXiv:2502.08590](https://arxiv.org/abs/2502.08590) (2025.02.12 초고, 2025.03.12 v2)
> - **GitHub**: [bcmi/Light-A-Video](https://github.com/bcmi/Light-A-Video)
> - **Project Page**: https://bujiazi.github.io/light-a-video.github.io/

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

본 논문은 **Light-A-Video**라는 학습(training) 없이 시간적으로 부드럽고 일관된 비디오 리라이팅(relighting)을 달성하는 방법을 제안한다. 이미지 리라이팅 모델로부터 착안하여, 조명 일관성을 강화하기 위한 두 가지 핵심 기법을 도입한다.

최근 이미지 리라이팅 모델의 급격한 발전에도 불구하고, 비디오 리라이팅은 과도한 학습 비용과 다양하고 고품질의 비디오 리라이팅 데이터셋 부족으로 인해 여전히 뒤처져 있다. 이미지 리라이팅 모델을 단순히 프레임 단위로 적용하면, 조명 소스의 불일치와 리라이팅된 외형(appearance) 불일치 문제가 발생해 영상에서 깜빡임(flickering)이 나타난다.

### 1.2 주요 기여 (Contributions)

Light-A-Video는 학습 없는 접근법으로서 특정 비디오 확산 모델에 국한되지 않아 AnimateDiff, CogVideoX 등 UNet 기반 및 DiT 기반을 포함한 다양한 인기 백본과 높은 호환성을 가진다. 주요 기여는 다음과 같다:
- **이미지 리라이팅 모델의 능력을 비디오 도메인으로 일반화**하는 새로운 학습 없는 비디오 리라이팅 프레임워크 제안
- **Consistent Light Attention(CLA) 모듈**과 **Progressive Light Fusion(PLF) 전략**이라는 두 핵심 설계 도입

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

최근 대규모 데이터셋과 사전 학습된 확산 모델에 의해 이미지 리라이팅은 일관된 조명을 적용하는 것이 가능해졌다. 그러나 비디오 리라이팅은 과도한 학습 비용과 다양하고 고품질의 비디오 리라이팅 데이터셋의 부족으로 인해 여전히 뒤처져 있다. 이미지 리라이팅 모델을 프레임별로 단순히 적용하면 조명 소스의 불일치와 리라이팅된 외형의 불일치, 그리고 비디오 내 깜빡임(flickering)이 발생한다.

요약하면, 기존 접근법의 두 가지 핵심 문제는 다음과 같다:

| 문제 | 설명 |
|------|------|
| **조명 소스 불일치** | 프레임마다 배경 조명 방향·강도가 달라지는 현상 |
| **외형(appearance) 불일치** | 리라이팅 후 객체 외형이 프레임마다 다르게 보이는 깜빡임 현상 |

---

### 2.2 제안 방법 (수식 포함)

#### 전체 파이프라인

Light-A-Video는 이미지 리라이팅 모델(예: IC-Light)과 비디오 확산 모델(예: CogVideoX, AnimateDiff)을 결합하여, 입력 비디오 또는 전경(foreground) 시퀀스에 대한 학습 없는 비디오 리라이팅을 가능하게 한다.

---

#### ① Consistent Light Attention (CLA)

조명 소스 생성을 안정화하고 일관된 결과를 보장하기 위해, 이미지 리라이팅 모델의 self-attention 레이어 내에 **CLA(Consistent Light Attention) 모듈**을 설계한다. CLA는 **시간적으로 평균화된(temporally averaged) 특징을 어텐션 연산에 추가로 통합**하여 프레임 간 상호작용을 촉진하고 구조적으로 안정된 조명 소스를 생성한다.

각 프레임을 독립적으로 처리하는 대신, CLA는 이웃 프레임들의 정보를 결합하여 비디오 전반에 걸쳐 일관된 조명 조건을 유지한다. **이중 스트림(dual-stream) 어텐션 융합 전략**을 활용하여, 원본 특징 맵과 평균화된 버전 모두에 대해 어텐션을 계산함으로써 고주파 노이즈를 필터링하고 시간에 따른 조명 전환을 부드럽게 한다.

CLA의 핵심 수식은 다음과 같이 표현할 수 있다.

프레임 $i$의 특징 $f_i$와 전체 프레임에 대한 시간적 평균 특징 $\bar{f}$를 정의할 때:

$$
\bar{f} = \frac{1}{N} \sum_{i=1}^{N} f_i
$$

CLA 모듈에서 수정된 Query($Q$), Key($K$), Value($V$)는:

$$
Q_i^{CLA} = W_Q \cdot f_i, \quad K_i^{CLA} = W_K \cdot \bar{f}, \quad V_i^{CLA} = W_V \cdot \bar{f}
$$

어텐션 출력:

$$
\text{CLA}(f_i) = \text{Softmax}\!\left(\frac{Q_i^{CLA} (K_i^{CLA})^\top}{\sqrt{d}}\right) V_i^{CLA}
$$

여기서 $d$는 어텐션 차원, $N$은 총 프레임 수이다.

> ⚠️ 위 수식은 논문의 공개된 설명("temporally averaged features into the attention computation")을 기반으로 재구성한 수식이며, 논문 원문의 기호와 세부 표기가 다를 수 있습니다. 정확한 수식은 원문 PDF를 참고하시기 바랍니다.

---

#### ② Progressive Light Fusion (PLF)

물리적 빛 전달 독립성(light transport independence) 원리를 활용하여, 소스 비디오의 외형(appearance)과 리라이팅된 외형 사이에 **선형 블렌딩(linear blending)**을 적용하는 **PLF(Progressive Light Fusion) 전략**을 사용하여 시간적으로 부드러운 조명 전환을 보장한다.

외형의 프레임 간 안정성을 더욱 향상시키기 위해, 비디오 확산 모델의 **모션 프라이어(motion prior)**를 활용한다. PLF는 빛 전달의 물리적 원리를 준수하여, CLA로부터 얻은 리라이팅된 외형을 원본 디노이징 타겟에 점진적으로 선형 블렌딩 방식으로 통합하며, 이를 통해 비디오 디노이징 과정이 원하는 리라이팅 방향으로 점진적으로 안내된다.

PLF의 핵심 블렌딩 수식은 다음과 같이 표현된다:

디노이징 단계 $t$에서의 혼합 비율 $\alpha_t$를 활용한 선형 블렌딩:

$$
\hat{x}_t^{fused} = \alpha_t \cdot \hat{x}_t^{relight} + (1 - \alpha_t) \cdot \hat{x}_t^{orig}
$$

여기서:
- $\hat{x}_t^{fused}$: 단계 $t$에서의 최종 융합된 디노이징 타겟
- $\hat{x}_t^{relight}$: CLA에서 생성된 리라이팅 타겟
- $\hat{x}_t^{orig}$: 원본 비디오의 디노이징 타겟
- $\alpha_t$: 노이즈 제거 진행도에 따라 점진적으로 변화하는 블렌딩 가중치

$\alpha_t$는 디노이징 스텝이 진행될수록 (즉, $t$가 감소할수록) 점점 증가하여, 초기에는 원본 외형을 유지하고 후기에는 리라이팅 방향으로 점진적으로 이동한다.

> ⚠️ 위 수식 역시 논문의 공개된 서술("progressive linear blending" 및 "gradually guides the video denoising process")을 기반으로 재구성한 것이며, 원문의 정확한 기호 및 표기를 직접 확인하시기 바랍니다.

---

### 2.3 모델 구조

Light-A-Video는 학습 없는(training-free) 환경에서 작동하며, 사전 학습된 이미지 리라이팅 모델과 비디오 확산 모델(VDM)을 결합하여 추가적인 학습이나 데이터셋 없이 비디오 프레임 전반에 걸쳐 조명 조건을 변경한다.

전체 파이프라인 구조를 도식화하면 다음과 같다:

```
입력 비디오
     │
     ▼
[이미지 리라이팅 모델 (예: IC-Light)]
  ─ 프레임별 리라이팅 수행
  ─ CLA 모듈 삽입 (self-attention 내 temporally averaged feature 통합)
     │  → 조명 소스 안정화
     │
     ▼
[CLA 출력: 구조적으로 안정된 리라이팅 타겟]
     │
     ▼
[Progressive Light Fusion (PLF)]
  ─ 원본 VDM 디노이징 타겟과 선형 블렌딩
  ─ 블렌딩 비율 α_t 점진적 증가
     │
     ▼
[Video Diffusion Model (예: CogVideoX, AnimateDiff)]
  ─ VDM 모션 프라이어로 시간적 일관성 확보
     │
     ▼
출력: 시간적으로 일관된 리라이팅 비디오
```

학습 없는 접근법으로서 Light-A-Video는 특정 비디오 확산 모델에 국한되지 않으며, UNet 기반 및 DiT 기반 모델(AnimateDiff, CogVideoX 등)을 포함한 다양한 인기 비디오 생성 백본들과 높은 호환성을 가진다.

---

### 2.4 성능 향상

실험 결과, Light-A-Video는 리라이팅된 이미지 품질을 유지하면서 리라이팅된 비디오의 시간적 일관성을 향상시키며, 프레임 간 일관된 조명 전환을 보장하는 것으로 나타났다.

VDM 모션 프라이어를 도입하고 PLF의 리라이팅 타겟을 원본 디노이징 타겟에 점진적으로 융합하는 전략을 통해, Light-A-Video는 시간적으로 부드러운 리라이팅을 보장한다. VDM 프라이어의 도움으로 전체적인 비디오 품질 또한 크게 향상되었다.

비교 실험에서 Light-A-Video는 CogVideoX, AnimateDiff, Wan 등 세 가지 다른 VDM 백본을 활용한 오픈소스 SOTA 모델로 평가되었으며, TC-Light와 비교 분석이 수행되었다.

---

### 2.5 한계

학습 없는 방법으로 인상적인 결과를 달성했음에도 불구하고, 성능은 본질적으로 기반이 되는 이미지 리라이팅 모델과 VDM의 역량에 의해 제약된다.

Light-A-Video는 안정적인 조명 및 시간적 일관성을 보장하는 데 있어 뛰어난 역량을 보이지만, 배경 조명을 안정화하기 위해 설계된 CLA 모듈은 **동적인 조명 변화(dynamic lighting changes)를 모델링하는 데 한계**가 있다. 이 한계를 해결하기 위해 미래 연구는 동적 조명 조건을 더욱 효과적으로 처리할 수 있는 새로운 방법 개발에 집중할 것이다.

---

## 3. 모델 일반화 성능 향상 가능성

Light-A-Video는 학습 없는 접근법으로서 특정 비디오 확산 모델에 국한되지 않아 UNet 기반 및 DiT 기반 모델을 포함한 다양한 인기 비디오 생성 백본과 높은 호환성을 가진다. 이는 일반화 성능 측면에서 다음과 같은 의미를 가진다:

| 일반화 측면 | 설명 |
|---|---|
| **모델 백본 독립성** | AnimateDiff(UNet), CogVideoX(DiT) 등 다양한 VDM 백본에 플러그인 방식으로 적용 가능 |
| **이미지 리라이팅 모델 독립성** | IC-Light 외에 다른 이미지 리라이팅 모델로 교체 가능 |
| **zero-shot 능력** | 학습 없이 임의의 비디오 시퀀스 또는 전경 시퀀스에 대해 zero-shot 조명 제어를 가능하게 하는 프레임워크이다. |
| **데이터 의존성 없음** | 고품질 비디오 리라이팅 데이터셋 없이도 동작 가능 |

비교 실험에서 100개 비디오 클립(70개 인물 초상화, 30개 비인물 환경)을 사용하였으며, 영상 내용은 실내외 환경, 비교적 정적인 장면과 역동적인 장면 등 다양한 시나리오를 포괄한다.

이를 통해 Light-A-Video의 일반화 가능성은 다음과 같이 정리할 수 있다:

1. **새로운 이미지 리라이팅 모델과의 결합**: 더 강력한 이미지 리라이팅 모델(예: IC-Light 후속 모델)이 등장하면, 별도 학습 없이 즉시 성능 향상이 가능하다.
2. **더 강력한 VDM 백본 활용**: Wan, HunyuanVideo 등 최신 비디오 생성 모델을 백본으로 채택하면 일반화 성능이 자동으로 향상된다.
3. **다양한 도메인 적용 가능성**: 사람, 동물, 실내외 환경 등 다양한 영상 콘텐츠에 적용 가능하다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 연도 | 방법 | 특징 | 한계 |
|------|------|------|------|------|
| **IC-Light** (Zhang et al.) | 2024 | 확산 모델 파인튜닝 기반 이미지 리라이팅 | 다양한 조명 조건 제어 가능 | 이미지 단위, 비디오 일관성 없음 |
| **RelightVid** (Fang et al.) | 2025 | 다양한 조명 조건 비디오 데이터셋 구축 후 VDM 학습 | 고품질 비디오 리라이팅 | 다양한 조명 조건의 고품질 비디오 데이터셋을 구축하여 VDM을 학습시키지만, 데이터 제작 비용이 매우 높다. |
| **Light-A-Video** (Zhou et al.) | 2025 | CLA + PLF, Training-free | 학습 불필요, 다양한 VDM 백본 지원 | 동적 조명 변화 모델링 한계 |
| **RelightMaster** (2025) | 2025 | 다중 평면 광 이미지(Multi-plane Light Images) 기반 | 정밀한 조명 제어 | 환경 맵에 의존하는 조명 제어 방식으로, 전경 객체를 수작업으로 추출해야 하며 환경 맵 기반의 정적 배경 교체를 강제하는 문제가 있다. |
| **LightCtrl** (2026) | 2026 | Training-free 제어 가능 비디오 리라이팅 | Light-A-Video의 진보로서, 사전 학습된 이미지 리라이팅 모델로 생성된 개별 리라이팅 프레임을 점진적으로 통합하는 방식. | — |
| **UniLumos** (2025) | 2025 | 물리 기반 피드백 통합 이미지/비디오 통합 리라이팅 | RGB 공간의 깊이·법선(normal) 감독을 통해 조명을 장면 기하와 정렬하며, 6차원 조명 어노테이션 프로토콜로 세밀한 조건 부여 가능. | — |
| **Learning Physics-Guided Face Relighting** (Nestmeyer et al.) | 2020 | 물리 기반 얼굴 리라이팅 학습 | 물리적 정확도 높음 | 얼굴에 특화, 비디오 미지원 |

---

## 5. 앞으로의 연구에 미치는 영향과 고려 사항

### 5.1 연구에 미치는 영향

1. **Training-free 패러다임의 확산**: Light-A-Video는 비디오 리라이팅에서 학습 없는 접근법의 실현 가능성을 입증함으로써, 향후 비디오 편집 분야에서 **훈련 없는 플러그인 방식**의 연구를 촉진할 것이다.

2. **이미지→비디오 도메인 전이의 새로운 방법론**: 이미지 리라이팅과 비디오 확산 모델의 발전을 활용하는 집중적인 접근 방식을 제시한다. 이는 다른 이미지 편집 기능(스타일 전환, 색상 편집 등)을 비디오 도메인으로 확장하는 데도 영향을 미칠 것이다.

3. **비디오 생성 모델 생태계와의 통합**: 특정 비디오 확산 모델에 국한되지 않아 다양한 인기 비디오 생성 백본과 높은 호환성을 가지므로, 향후 새로운 VDM 백본이 등장할 때마다 자동으로 활용 가능한 유연한 아키텍처를 제시한다.

4. **데이터셋 구축 비용 절감**: 비디오 리라이팅에서 과도한 학습 비용과 데이터셋 부족 문제를 우회하는 방법론을 제시함으로써, 이 분야의 진입 장벽을 낮춘다.

### 5.2 향후 연구 시 고려 사항

1. **동적 조명 처리**: CLA 모듈은 배경 조명을 안정화하는 데 설계되어 동적 조명 변화를 모델링하는 데 한계가 있다. 시간에 따라 변화하는 조명 조건(일출/일몰, 네온사인 깜빡임 등)을 처리하기 위한 **동적 조명 인식 모듈** 개발이 필요하다.

2. **기반 모델 성능 의존성 극복**: 성능이 기반 이미지 리라이팅 모델과 VDM의 역량에 의해 제약된다는 점을 감안하여, 경량 파인튜닝(LoRA 등)과의 결합을 통해 기반 모델 의존성을 줄이는 연구가 필요하다.

3. **물리적 정확성 향상**: RelightMaster, UniLumos 등 후속 연구들이 방향으로 제시하듯, 단순한 선형 블렌딩을 넘어 렌더링 방정식 기반의 물리적으로 정확한 조명 모델과 결합하는 연구가 의미 있을 것이다.

4. **고해상도·장시간 비디오 처리**: 비교 실험에서 1080p~2160p의 다양한 해상도와 정적·동적 장면을 포괄하는 100개 클립을 사용하였지만, 더 긴 영상과 더 높은 해상도에서의 효율성 및 일관성 유지가 과제로 남는다.

5. **다중 광원(multi-light) 및 시변 광원(time-varying light) 지원**: 시변 조명(조명 소스가 좌상단에서 우하단으로 이동하거나 색상이 붉은색에서 녹색으로 전환되는 경우)과 다중 조명 시나리오를 지원하는 방향으로의 확장이 요구된다.

6. **평가 지표(metric) 표준화**: 비디오 리라이팅 분야는 아직 표준화된 평가 지표가 부족하므로, 시간적 일관성, 조명 정확도, 지각적 품질을 종합적으로 평가하는 새로운 벤치마크 구축이 필요하다.

---

## 📚 참고 자료 (출처)

1. **arXiv 원문**: Zhou, Y. et al. "Light-A-Video: Training-free Video Relighting via Progressive Light Fusion." arXiv:2502.08590, 2025. — https://arxiv.org/abs/2502.08590
2. **ICCV 2025 공식 논문 PDF**: https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Light-A-Video_Training-free_Video_Relighting_via_Progressive_Light_Fusion_ICCV_2025_paper.pdf
3. **GitHub 공식 구현**: https://github.com/bcmi/Light-A-Video
4. **HuggingFace Paper Page**: https://huggingface.co/papers/2502.08590
5. **Moonlight Literature Review**: https://www.themoonlight.io/en/review/light-a-video-training-free-video-relighting-via-progressive-light-fusion
6. **LightCtrl** (비교 논문): arXiv:2603.27083, 2026. — https://arxiv.org/pdf/2603.27083
7. **UniLumos** (비교 논문): arXiv:2511.01678, 2025. — https://arxiv.org/html/2511.01678v1
8. **Hi-Light** (비교 논문): arXiv:2601.23167, 2026. — https://arxiv.org/html/2601.23167
9. **RelightMaster** (비교 논문): arXiv:2511.06271, 2025. — https://arxiv.org/html/2511.06271v1
10. **ICCV 2025 Poster Page**: https://iccv.thecvf.com/virtual/2025/poster/1746

> ⚠️ **주의사항**: CLA 및 PLF의 세부 수식은 논문의 공개된 서술을 기반으로 재구성된 것입니다. 논문 원문 PDF에서 정확한 수식을 직접 확인하시기를 강력히 권장합니다.
