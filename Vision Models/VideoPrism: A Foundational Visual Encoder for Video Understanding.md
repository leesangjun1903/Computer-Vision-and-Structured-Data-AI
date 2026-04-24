# VideoPrism: A Foundational Visual Encoder for Video Understanding

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

VideoPrism은 **단일 동결(frozen) 모델**로 다양한 비디오 이해 태스크를 처리할 수 있는 범용 비디오 인코더입니다. 기존 Video Foundation Models(ViFMs)가 외관(appearance) 중심 태스크와 모션(motion) 중심 태스크 사이에서 균형을 잡지 못하는 한계를 극복하고자 합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| 대규모 이종 데이터셋 구축 | 36M 고품질 video-caption 쌍 + 582M 노이즈 포함 video-text 클립 |
| 2단계 사전학습 전략 | 대조 학습 → 마스크 비디오 모델링 순차 진행 |
| Global-Local Distillation | 의미론적 임베딩의 전역/지역 증류 |
| Token Shuffling | 디코더의 단축 학습(shortcut) 방지 |
| 포괄적 평가 | 33개 벤치마크 중 **31개에서 SOTA** 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: 외관-모션 불균형**
기존 ViFMs는 외관 중심 태스크(예: Kinetics-400)에서는 강하지만, 모션 중심 태스크(예: Something-Something v2)에서 약한 경향이 있습니다. 비디오 캡션은 주로 외관 정보를 포함하며(Wang et al., 2023f), 모션 정보를 충분히 학습하기 어렵습니다.

**문제 2: 데이터 품질 이질성**
대부분의 대규모 비디오 데이터는 ASR 자막, 메타데이터 등 노이즈가 많은 텍스트를 포함합니다. 고품질 캡션 데이터는 이미지 데이터 대비 수십 배 부족합니다.

**문제 3: Masked Video Modeling의 의미론적 부재**
원시 픽셀 신호는 언어 토큰과 달리 내재적 의미론(semantics)이 없어, 마스크 자동인코딩만으로는 풍부한 의미 표현 학습이 어렵습니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Stage 1: Video-Text Contrastive Training

비디오-텍스트 쌍에 대해 대칭적 대조 손실(symmetric cross-entropy loss)을 최소화합니다:

$$\mathcal{L}_{\text{contrastive}} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j)/\tau)} + \log \frac{\exp(\text{sim}(t_i, v_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(t_j, v_i)/\tau)} \right]$$

여기서:
- $v_i$: $i$번째 비디오의 임베딩 (Multi-head Attention Pooler를 통해 집계)
- $t_i$: $i$번째 텍스트의 임베딩
- $\tau$: 학습 가능한 온도 파라미터
- $N$: 미니배치 크기
- $\text{sim}(\cdot, \cdot)$: 코사인 유사도

#### Stage 2: Masked Video Modeling with Global-Local Distillation

**(1) Token-wise Distillation Loss**

마스크된 비디오의 가시 토큰( $\mathbf{x}\_{\text{vis}}$ )으로부터 전체 토큰( $\mathbf{z}_{\text{all}}$ )을 예측합니다. 코사인 거리(cosine distance)를 최소화:

$$\mathcal{L}_{\text{token}} = \frac{1}{N_{\text{mask}}} \sum_{i \in \mathcal{M}} \left(1 - \frac{\hat{\mathbf{z}}_i \cdot \mathbf{z}_i^{\text{teacher}}}{\|\hat{\mathbf{z}}_i\| \cdot \|\mathbf{z}_i^{\text{teacher}}\|}\right)$$

여기서:
- $\mathcal{M}$: 마스킹된 토큰의 인덱스 집합
- $\hat{\mathbf{z}}_i$: 2단계 디코더의 예측 임베딩
- $\mathbf{z}_i^{\text{teacher}}$: 1단계 모델(teacher)의 토큰 임베딩

**(2) Global Distillation Loss**

가시 토큰만으로 1단계 teacher의 전역 임베딩을 예측:

$$\mathcal{L}_{\text{global}} = 1 - \frac{\hat{\mathbf{g}} \cdot \mathbf{g}^{\text{teacher}}}{\|\hat{\mathbf{g}}\| \cdot \|\mathbf{g}^{\text{teacher}}\|}$$

여기서:
- $\hat{\mathbf{g}}$: 2단계 모델이 가시 토큰으로부터 생성한 전역 임베딩
- $\mathbf{g}^{\text{teacher}}$: 1단계 teacher가 **전체(비마스크)** 비디오로부터 생성한 전역 임베딩

**(3) 최종 2단계 학습 손실**

$$\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{token}} + \lambda \cdot \mathcal{L}_{\text{global}}$$

논문에 따르면 $\lambda = 1$ (동일 가중치 적용)

**(4) Token Shuffling**

```
# 의사 코드 (논문 Algorithm 1 기반)
z = mask_embedding.expand(batch, n-m, dim)  # 마스크 토큰
out_emb = concat([token_emb, z], axis=1)     # [b, n, dim]
out_emb = shuffle(out_emb, axis=1)           # 랜덤 셔플
out_emb = out_emb + positional_embedding     # 위치 임베딩 추가 후 디코더 입력
```

셔플 후 위치 임베딩을 추가함으로써, 디코더가 가시 토큰의 위치 정보를 직접 이용하는 단축 학습을 방지합니다.

---

### 2.3 모델 구조

```
입력 비디오 (T × H × W × 3)
        ↓ Patchify (18×18 패치)
  [T × (H/18) × (W/18)] 토큰 시퀀스
        ↓ Drop/Mask (비율 ρ)
  ┌─────────────────────┐
  │   Spatial Encoder   │  ← ViT-giant (40 layers, MSA 6144)
  │  (같은 시간 인덱스)  │
  └─────────────────────┘
        ↓ Transpose
  ┌─────────────────────┐
  │  Temporal Encoder   │  ← 4 layers
  │  (다른 시간 인덱스)  │
  └─────────────────────┘
        ↓
  공간-시간 토큰 출력: [T × (H/18) × (W/18) × D]
```

| 구성요소 | VideoPrism-g | VideoPrism-B |
|---------|--------------|--------------|
| 기반 아키텍처 | ViT-giant (ViViT 기반) | ViT-Base |
| 파라미터 수 | ~1B (공간 인코더) | 소규모 |
| Spatial Encoder | 40 layers, MSA 6144 | Base 설정 |
| Temporal Encoder | 4 layers | 4 layers |
| 공간 해상도 | 288×288, 패치 18×18 | 동일 |
| 입력 프레임 수 | 사전학습: 8, 평가: 16 | 동일 |

**핵심 설계 결정**: ViViT의 global average pooling 제거 → 시공간 토큰 시퀀스 유지 → 세밀한 특징(localization 등)이 필요한 다운스트림 태스크 지원

---

### 2.4 성능 향상

#### VideoGLUE 벤치마크 (동결 백본 기준)

| 태스크 | 최고 베이스라인 | VideoPrism-B | 향상폭 |
|--------|---------------|--------------|--------|
| K400 (VC) | UMT-B: 77.1 | **84.2** | +7.1 |
| SSv2 (VC) | InternVideo-B: 58.2 | **63.6** | +5.4 |
| Diving48 (VC) | InternVideo-L: 69.6 | **67.4** | (g: 71.3, +1.7) |
| Charades (VC) | VATT-B: 33.3 | **40.4** | +7.1 |
| AVA (STAL) | UMT-B: 24.4 | **30.6** | +6.2 |
| AVA-K (STAL) | VATT-B: 22.2 | **31.8** | +9.6 |

#### Zero-shot 비디오-텍스트 검색 (R@1)

| 데이터셋 | 방향 | 최고 베이스라인 | VideoPrism-g | 향상폭 |
|---------|------|---------------|--------------|--------|
| MSRVTT | T→V | VideoCoCa-g: 43.9 | **52.7** | +8.8 |
| ActivityNet | T→V | UMT-L: 42.8 | **52.7** | +9.9 |
| VATEX | T→V | VideoCoCa-g: 53.2 | **62.5** | +9.3 |

#### VideoGLUE Score (VGS) — 4가지 적응 방법 종합

$$\text{VGS}_{\text{VideoPrism-B}} = 51.25 \quad \text{vs} \quad \text{VGS}_{\text{UMT-B}} = 45.3 \quad (+13.6\%)$$

---

### 2.5 한계

1. **노이즈 데이터 의존성**: 사전학습 데이터의 대부분이 노이즈 텍스트(ASR, 메타데이터 등)로, 잠재적으로 편향되거나 불완전할 수 있습니다.

2. **단기 비디오 클립 한정**: 현재 16프레임 입력에 최적화되어 있어 **장기 비디오 이해**에 제약이 있습니다.

3. **동결 백본의 한계**: 일부 태스크에서는 end-to-end 파인튜닝이 더 유리하나, 비디오 모델의 높은 계산 비용으로 인해 실용적으로 제한적입니다.

4. **데이터 공개 한계**: 핵심 데이터셋(Anonymous-Corpus #1~#3)이 비공개로, 재현성에 제약이 있습니다.

5. **멀티모달 편향**: 텍스트 캡션이 주로 외관 정보를 포착하는 특성상, 순수 모션 이해에는 2단계 학습이 필수적이지만 여전히 일부 모션 벤치마크(VATEX 캡션 등)에서 개선 여지가 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화의 핵심 메커니즘

VideoPrism의 일반화 성능은 다음 세 가지 축에서 이해될 수 있습니다:

#### (A) 데이터 다양성 기반 일반화

$$\mathcal{D}_{\text{pretrain}} = \mathcal{D}_{\text{HQ}}^{36M} \cup \mathcal{D}_{\text{noisy}}^{582M}$$

- **도메인 다양성**: 웹 비디오, 강의 비디오, 의미있는 인간 행동 등 다양한 소스
- **Caption 품질 다양성**: 수동 레이블 ~ ASR ~ VLM 생성 등 이질적 품질 혼합
- **평가 데이터 분리**: 33개 평가 벤치마크 훈련셋을 사전학습에서 완전 제외 + 비디오 중복 제거 → 데이터 누출 방지

#### (B) 이중 감독 신호 (Dual Supervisory Signals)

```
외관 정보 학습 ← [1단계: 텍스트-비디오 대조 학습]
                       ↓ Global Distillation
모션 정보 학습 ← [2단계: 마스크 비디오 모델링]
                       ↓ Token-wise Distillation
```

이 이중 구조는 한쪽으로 치우치지 않는 표현 학습을 가능하게 합니다:
- 외관 중심 태스크: K400에서 87.2% (기존 SOTA 대비 +4.4%)
- 모션 중심 태스크: SSv2에서 68.5% (기존 SOTA 대비 +1.1%)

#### (C) 과학 분야로의 도메인 이전 (CV for Science)

논문에서 처음으로 ViFM을 과학 데이터셋에 평가한 결과:

| 데이터셋 | 도메인 | 도메인 전문 모델 | VideoPrism-B | 결과 |
|---------|--------|----------------|--------------|------|
| Fly vs. Fly | 곤충 행동학 | 88.6 | **89.1** | 초월 |
| CalMS21 | 신경과학(마우스) | 88.9 | **91.1** | 초월 |
| ChimpACT | 인지과학(침팬지) | 24.4 | **28.8** | 초월 |
| KABR | 생태학(케냐 동물) | 61.9 | 61.6 | 준하는 성능 |

이는 VideoPrism이 **한 번도 학습하지 않은 도메인**에서도 도메인 특화 모델과 동등하거나 우수한 성능을 발휘함을 의미합니다.

#### (D) 스케일링에 따른 일반화 향상

논문의 스케일링 실험(Figure 8)에서:

$$\text{SSv2 향상폭} \approx +8\% \quad \text{(2단계 모델이 1단계 모델 대비, 모든 모델 크기에서 일관됨)}$$

$$\text{AVA 향상폭} \approx +2.2\% \quad \text{(1단계 대비, 모델 크기 전반에 걸쳐 일관됨)}$$

고정된 1단계 모델(Base)에 더 큰 2단계 모델을 적용해도 합리적인 스케일링 성능을 보입니다.

#### (E) 적응 비용 대비 일반화 성능

동결 백본 설정에서의 우수한 성능은 VideoPrism의 표현이 광범위한 태스크에 범용적으로 유용함을 시사합니다:

$$\Delta\text{VGS} = 51.25 - 45.3 = 5.95 \quad \text{(2위 UMT 대비)}$$

낮은 적응 비용(frozen + MAP head) 설정에서 다른 모델들보다 더 큰 폭으로 개선됩니다(Table 17).

### 3.2 일반화의 잠재적 확장 방향

1. **AudioVisual 통합**: 현재는 시각 정보만 활용하나, 오디오 모달리티 추가 시 일반화 더욱 향상 가능
2. **장기 비디오로의 확장**: 현재 16프레임 제한 → 계층적 처리 구조 도입 시 장기 의존성 학습 가능
3. **더 많은 과학 도메인**: 의료 영상, 로봇 비전, 위성 영상 등으로의 전이 학습 가능성

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 관련 연구 타임라인

| 연도 | 모델/논문 | 핵심 방법 | VideoPrism 대비 |
|------|-----------|----------|----------------|
| 2020 | BERT (Devlin et al., 2019) | 마스크 언어 모델링 | 영감의 원천 |
| 2021 | CLIP (Radford et al., 2021) | 이미지-텍스트 대조 학습 | 이미지 중심, 모션 부족 |
| 2021 | VideoMAE (Tong et al., 2022) | 마스크 비디오 오토인코딩 | 의미론적 감독 부족 |
| 2021 | VATT (Akbari et al., 2021) | 멀티모달 자기지도 학습 | 단일 모달리티 포커스 미흡 |
| 2022 | InternVideo (Wang et al., 2022c) | VideoMAE + 비디오-언어 결합 | 두 모델이 사전학습 시 상호작용 없음 |
| 2022 | UMT (Li et al., 2023b) | Unmasked Teacher | 시맨틱은 CLIP에서만, 자체 학습 없음 |
| 2022 | VideoMAE v2 (Wang et al., 2023b) | 이중 마스킹 스케일링 | 비디오-텍스트 대조 학습 없음 |
| 2022 | VideoCoCa (Yan et al., 2022) | CoCa를 비디오로 확장 | 이미지 FMs 적응, 비디오 고유 학습 부족 |
| 2022 | EVA/EVA-02 (Fang et al., 2022; 2023) | CLIP으로 마스크 모델링 부트스트랩 | 이미지 중심 |
| 2023 | ImageBind (Girdhar et al., 2023) | 6가지 모달리티 바인딩 | 비디오 특화 표현 부족 |
| 2024 | **VideoPrism** | 이중 단계 + Global-Local 증류 + 토큰 셔플 | **범용성, 과학 도메인 최초 평가** |

### 4.2 핵심 방법론 비교

| 비교 축 | InternVideo | VideoMAE v2 | UMT | **VideoPrism** |
|---------|-------------|-------------|-----|----------------|
| 사전학습 데이터 | Kinetics 포함 | Kinetics 포함 | 25M | **618M+ (Kinetics 제외)** |
| 대조 학습 | ✓ (별도 모델) | ✗ | ✓ (CLIP 활용) | ✓ (통합 파이프라인) |
| 마스크 모델링 | ✓ (별도 모델) | ✓ | ✓ | ✓ (2단계 통합) |
| 의미론적 증류 | ✗ | ✗ | 간접적 | ✓ (Global+Local) |
| 토큰 셔플 | ✗ | ✗ | ✗ | ✓ |
| 과학 도메인 평가 | ✗ | ✗ | ✗ | ✓ (최초) |
| 동결 평가 중심 | ✗ | ✗ | 일부 | ✓ (핵심 설정) |

### 4.3 VideoPrism의 차별점 요약

**기존 접근법의 문제:**
- **InternVideo**: 두 개의 독립적 모델(VideoMAE + 비디오-언어 모델)을 사후에 결합 → 사전학습 시 상호작용 없음
- **VideoMAE 계열**: 풍부한 모션 학습 가능하나 의미론적 지식 부족
- **CLIP/VideoCoCa 계열**: 강력한 의미론 학습 가능하나 모션 이해 부족

**VideoPrism의 해결책:**
- 이중 단계 학습으로 두 세계의 장점을 **유기적으로** 통합
- Global Distillation으로 catastrophic forgetting 방지
- Token Shuffling으로 더 어려운 학습 목표 제시 → 모션 이해 향상

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

#### (A) 동결 인코더 패러다임의 확산

VideoPrism은 동결 백본 설정에서도 충분한 성능을 보임으로써:
- **VideoLLM(Video Large Language Model)** 개발의 표준 비디오 인코더로 활용될 가능성이 높습니다
- 계산 비용 절감: 비디오 인코딩 비용을 여러 다운스트림 태스크에 분산

#### (B) CV for Science 분야의 새로운 방향

논문이 제시한 과학 분야 ViFM 평가 프레임워크는 다음 분야에 영향을 미칩니다:
- 의료 영상 분석 (수술 동영상, 세포 분열 관찰 등)
- 기후 과학 (위성 비디오 분석)
- 로보틱스 (조작 동작 학습)

#### (C) 데이터 전략의 재정의

이질적 품질의 대규모 데이터를 효과적으로 활용하는 방법론(AGD, 2단계 학습)은 다른 멀티모달 학습 연구에도 일반적으로 적용 가능한 원칙을 제공합니다.

#### (D) 지식 증류 방법론의 고도화

Global-Local Distillation은 teacher-student 학습 패러다임의 새로운 변형으로, 이미지/3D 포인트클라우드 등 다른 도메인에도 적용 가능합니다.

### 5.2 앞으로 연구 시 고려할 점

#### (A) 장기 비디오 이해

**현재 한계**: 16프레임 입력으로 제한
**연구 방향**:
- 계층적 비디오 처리 (단기 클립 → 장기 구조 이해)
- 메모리 효율적 시간 어텐션 메커니즘
- 이벤트 기반 희소 샘플링 전략

$$\text{연구 제안: } \mathcal{L}_{\text{long-video}} = \mathcal{L}_{\text{local}}(\text{clip}) + \mathcal{L}_{\text{global}}(\text{video})$$

#### (B) 데이터 품질과 다양성의 균형

**고려사항**:
- CLIP 유사도 점수(grounding score)와 실제 다운스트림 성능 간의 관계 심층 분석
- 노이즈 텍스트에 포함된 사회적 편향 제거
- 지리적, 문화적 다양성 확보를 위한 데이터 큐레이션

#### (C) 효율적인 파인튜닝 방법

**현재 접근**: LoRA, Adapter, 완전 파인튜닝
**미래 연구 방향**:
- 비디오에 특화된 파라미터 효율적 적응(PEFT) 방법
- 적응 비용 대비 성능 트레이드오프 최적화

$$\text{효율성 목표: } \min_{\theta_{\text{adapter}}} \mathcal{L}_{\text{downstream}} \quad \text{s.t.} \quad |\theta_{\text{adapter}}| \ll |\theta_{\text{total}}|$$

#### (D) 멀티모달 확장 (오디오 통합)

현재 VideoPrism은 시각 정보만을 처리하나, 오디오 정보 통합 시:
- 소리-동작 연관성 학습 (예: 음악 연주 동작)
- 언어 장벽 없는 다국어 비디오 이해
- ImageBind(Girdhar et al., 2023)와 같은 멀티모달 결합 방식과의 비교 연구 필요

#### (E) 평가 프로토콜의 표준화

**문제**: 다양한 논문이 서로 다른 평가 설정 사용 → 공정한 비교 어려움
**제안**:
- VideoGLUE와 같은 표준 벤치마크 확장
- LLM-in-the-loop 평가 방법론의 표준화
- 계산 비용 대비 성능을 명시하는 표준 보고 체계

#### (F) 책임 있는 AI 개발

논문이 명시한 바와 같이:
- 알고리즘적 편향 모니터링 (특정 문화, 행동 유형에 대한 편향)
- 프라이버시 보호: 비식별화 처리된 데이터셋 구축 필요
- 오용 방지 메커니즘: 딥페이크, 감시 등에의 악용 방지

#### (G) 벤치마크 포화 문제

33개 벤치마크 중 31개에서 SOTA를 달성한 현시점에서:
- 기존 벤치마크의 포화(saturation) 문제 해결을 위한 **더 도전적인 벤치마크** 설계 필요
- 특히 구성적 추론(compositional reasoning), 반사실적 이해(counterfactual understanding) 등 고차원 능력 평가

---

## 참고 자료 (출처)

**주 논문:**
- Zhao, L., Gundavarapu, N. B., Yuan, L., Zhou, H., Yan, S., Sun, J. J., et al. (2024). **VideoPrism: A Foundational Visual Encoder for Video Understanding**. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR 235. arXiv:2402.13217v3.

**논문 내 인용 핵심 참고문헌:**
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*. [CLIP]
- Tong, Z., et al. (2022). VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training. *NeurIPS*. [VideoMAE]
- Wang, Y., et al. (2022c). InternVideo: General video foundation models via generative and discriminative learning. arXiv:2212.03191. [InternVideo]
- Li, K., et al. (2023b). Unmasked teacher: Towards training-efficient video foundation models. *ICCV*. [UMT]
- Wang, L., et al. (2023b). VideoMAE v2: Scaling video masked autoencoders with dual masking. *CVPR*. [VideoMAE v2]
- Yuan, L., et al. (2023). VideoGLUE: Video general understanding evaluation of foundation models. arXiv:2307.03166. [VideoGLUE]
- He, K., et al. (2022). Masked autoencoders are scalable vision learners. *CVPR*. [MAE]
- Yu, J., et al. (2022). CoCa: Contrastive captioners are image-text foundation models. *TMLR*. [CoCa]
- Arnab, A., et al. (2021). ViViT: A video vision transformer. *ICCV*. [ViViT]
- Fang, Y., et al. (2022; 2023). EVA / EVA-02. *CVPR*. [EVA]
- Girdhar, R., et al. (2023). ImageBind: One embedding space to bind them all. *CVPR*. [ImageBind]
- Alayrac, J.-B., et al. (2022). Flamingo: A visual language model for few-shot learning. *NeurIPS*. [Flamingo]
- Yan, S., et al. (2022). VideoCoCa: Video-text modeling with zero-shot transfer from contrastive captioners. arXiv:2212.04979. [VideoCoCa]
- Wang, R., et al. (2023c). Masked video distillation. *CVPR*. [MVD]
- Zhai, X., et al. (2022b). LiT: Zero-shot transfer with locked-image text tuning. *CVPR*. [LiT]
- Noroozi, M., & Favaro, P. (2016). Unsupervised learning of visual representations by solving Jigsaw puzzles. *ECCV*. [Jigsaw]
- Anil, R., et al. (2023). PaLM 2 technical report. arXiv:2305.10403. [PaLM-2]
- Zhu, B., et al. (2024). LanguageBind. *ICLR*. [LanguageBind]
