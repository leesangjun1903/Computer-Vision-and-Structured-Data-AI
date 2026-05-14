
# Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs

> **논문 정보**
> - **제목**: Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs
> - **저자**: Brandon Smart, Chuanxia Zheng, Iro Laina, Victor Adrian Prisacariu
> - **소속**: Active Vision Lab & Visual Geometry Group, University of Oxford
> - **arXiv**: [2408.13912](https://arxiv.org/abs/2408.13912) (2024년 8월)
> - **프로젝트 페이지**: https://splatt3r.active.vision/
> - **코드**: https://github.com/btsmart/splatt3r

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

Splatt3R는 포즈 정보 없이(pose-free) 스테레오 이미지 쌍으로부터 야생(in-the-wild) 3D 재구성 및 새로운 시점 합성(Novel View Synthesis)을 수행하는 피드-포워드(feed-forward) 방법으로, **캘리브레이션되지 않은 자연 이미지에서 카메라 파라미터나 깊이(depth) 정보 없이 3D Gaussian Splat을 예측**할 수 있습니다.

저자들의 주장에 따르면, Splatt3R는 **광역(wide), 비포즈(unposed) 스테레오 이미지 쌍으로부터 피드-포워드 방식으로 새로운 시점 합성을 위한 3D 재구성을 수행하는 최초의 모델**입니다.

### 1.2 주요 기여 (Key Contributions)

| 기여 | 내용 |
|------|------|
| **① Foundation 모델 활용** | MASt3R 위에 Gaussian 외양 예측을 추가 |
| **② 새로운 예측 헤드 도입** | 공분산, 구면 조화, 불투명도 예측 |
| **③ 2단계 학습 전략** | 기하 손실 → 시점 합성 손실 |
| **④ Loss Masking 전략** | 외삽 시점에서의 성능 향상 |
| **⑤ 실시간 렌더링** | 4 FPS, 512×512 해상도 달성 |

저자들은 MASt3R 아키텍처의 간단한 수정과 잘 선택된 학습 손실만으로도 강력한 새로운 시점 합성 결과를 달성하기에 충분함을 보여줍니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존의 새로운 시점 합성 방법들은 테스트 시 각 이미지에 대한 카메라 내부/외부 파라미터(intrinsics, extrinsics)가 필요하여 야생(in-the-wild) 사진 쌍에 대한 적용이 제한됩니다. 또한 알려지지 않은 카메라 포즈를 가진 장면 최적화 방법들은 대량의 이미지 컬렉션을 필요로 합니다.

구체적으로 두 가지 핵심 문제가 존재합니다:

1. **포즈 없는 스테레오 재구성**: 카메라 캘리브레이션 없이 두 장의 이미지만으로 3D 장면 복원
2. **로컬 미니마 문제**: 포즈 정보 없이는 3D Gaussian 중심 위치 결정이 매우 어려우며, 포즈 정보가 있어도 반복적 3D Gaussian Splatting 최적화는 로컬 미니마에 취약합니다. 저자들의 해결책은 각 학습 샘플에 대해 정답 3D 포인트 클라우드를 명시적으로 지도학습함으로써 카메라 포즈 부재 문제와 로컬 미니마 문제를 동시에 해결하는 것입니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 3D Gaussian Splatting 표현

각 장면은 3D Gaussian Primitive들의 집합으로 표현됩니다. 각 Gaussian은 다음 파라미터로 구성됩니다:

$$\mathcal{G} = \{\mu_i, \Sigma_i, \alpha_i, \mathbf{c}_i\}_{i=1}^N$$

- $\mu_i \in \mathbb{R}^3$: 3D 중심 위치 (mean)
- $\Sigma_i \in \mathbb{R}^{3\times3}$: 공분산 행렬 (covariance)
- $\alpha_i \in [0,1]$: 불투명도 (opacity)
- $\mathbf{c}_i$: 색상 (구면 조화 함수, SH로 표현)

공분산 행렬은 회전 쿼터니언 $\mathbf{q}_i$와 스케일 $\mathbf{s}_i$로 분해되어 파라미터화됩니다:

$$\Sigma_i = R(\mathbf{q}_i) \cdot \text{diag}(\mathbf{s}_i)^2 \cdot R(\mathbf{q}_i)^T$$

#### 2.2.2 렌더링 과정

픽셀 $\mathbf{p}$의 색상은 깊이 순서로 정렬된 Gaussian들의 알파 합성(alpha compositing)으로 결정됩니다:

$$\hat{C}(\mathbf{p}) = \sum_{i \in \mathcal{N}} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

여기서 각 Gaussian의 투영된 2D 불투명도는:

$$\alpha_i^{2D} = \alpha_i \cdot \exp\left(-\frac{1}{2}(\mathbf{p} - \mu_i^{2D})^T (\Sigma_i^{2D})^{-1} (\mathbf{p} - \mu_i^{2D})\right)$$

#### 2.2.3 포인트 맵 표현

모델은 두 이미지 $I^1, I^2 \in \mathbb{R}^{W \times H \times 3}$로부터 픽셀-정렬된 두 포인트 맵 $C^1, C^2 \in \mathbb{R}^{W \times H}$를 동시에 예측하며, **첫 번째 이미지의 좌표 프레임에서 두 포인트 맵을 모두 예측**함으로써 카메라 포즈를 이용한 포인트 클라우드 변환 필요성을 제거합니다.

#### 2.2.4 학습 손실 함수

**1단계: 기하 손실 (MASt3R의 Confidence-Weighted Regression Loss)**

$$\mathcal{L}_\text{geo} = \sum_{v \in \{1,2\}} \sum_{i} C_i^v \left\| \hat{P}_i^v - P_i^v \right\|^2 - \lambda \log C_i^v$$

- $\hat{P}_i^v$: 예측된 3D 포인트
- $P_i^v$: 정답(GT) 3D 포인트
- $C_i^v$: 신뢰도(confidence) 가중치

**2단계: 새로운 시점 합성 손실**

LPIPS 손실 항을 사용하면 재구성의 시각적 품질이 의미 있게 향상됩니다.

$$\mathcal{L}_\text{NVS} = \mathcal{L}_1(\hat{I}, I) + \lambda_\text{SSIM} \cdot \mathcal{L}_\text{SSIM}(\hat{I}, I) + \lambda_\text{LPIPS} \cdot \mathcal{L}_\text{LPIPS}(\hat{I}, I)$$

**전체 학습 목표:**

$$\mathcal{L}_\text{total} = \underbrace{\mathcal{L}_\text{geo}}_{\text{Phase 1}} + \underbrace{\mathcal{L}_\text{NVS} \cdot \mathbf{M}}_{\text{Phase 2 (masked)}}$$

여기서 $\mathbf{M}$은 Loss Masking 전략에 의한 마스크입니다.

#### 2.2.5 Loss Masking 전략

Gaussian 파라미터 예측을 최적화하기 위해, 학습 중 각 샘플은 장면 재구성에 사용되는 두 개의 '컨텍스트(context)' 이미지와 렌더링 손실 계산에 사용되는 여러 '타겟(target)' 이미지로 구성됩니다. 타겟 이미지들 중 일부는 가려지거나 시야 밖에 있어 두 컨텍스트 뷰에서 보이지 않는 장면 영역을 포함할 수 있습니다. 이를 해결하기 위해 컨텍스트 이미지의 픽셀과 직접적인 대응 관계를 가진 픽셀을 포함하는 렌더링 영역만 지도학습합니다. 이를 통해 두 입력 이미지 사이의 보간이 아닌 시점에서도 렌더링을 지도학습할 수 있습니다.

Loss Masking 전략을 생략하면 Gaussian의 크기가 무한정 커져 렌더링의 메모리 비용으로 인해 학습이 중단됩니다.

---

### 2.3 모델 구조

#### 2.3.1 전체 아키텍처

각 이미지를 Vision Transformer(ViT) 인코더로 동시에 인코딩한 후, 이미지 간 크로스-어텐션(cross-attention)을 수행하는 Transformer 디코더로 전달합니다.

```
입력: 두 장의 비캘리브레이션 이미지 (I¹, I²)
          │
    ┌─────▼─────┐
    │  ViT Encoder   │  ← MASt3R 사전학습 가중치 (Frozen)
    │  (공유 가중치)  │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ Cross-Attention  │  ← MASt3R Transformer Decoder (Frozen)
    │   Transformer    │
    └─────┬─────┘
          │
    ┌─────▼───────────────────┐
    │        예측 헤드들           │
    ├──────────────────────────┤
    │ Head 1: 3D 포인트 + 신뢰도  │  ← MASt3R 기존 헤드
    │ Head 2: 피처 매칭           │  ← MASt3R 기존 헤드
    │ Head 3: Gaussian 속성      │  ← Splatt3R 신규 헤드 ✨
    │  (회전 쿼터니언, 스케일,       │
    │   구면 조화, 불투명도,         │
    │   평균 오프셋)                │
    └──────────────┬───────────┘
                   │
    ┌──────────────▼───────────┐
    │   3D Gaussian Primitives  │
    │   → 미분 가능 렌더러        │
    │   → Novel View Synthesis  │
    └───────────────────────────┘
```

#### 2.3.2 새로운 예측 헤드

기존 MASt3R는 픽셀-정렬 3D 포인트와 신뢰도를 예측하는 헤드와 피처 매칭에 사용되는 헤드, 두 개의 예측 헤드를 가집니다. Splatt3R는 **세 번째 헤드를 추가**하여 각 포인트에 대한 공분산(회전 쿼터니언과 스케일로 파라미터화), 구면 조화, 불투명도, 평균 오프셋을 예측합니다.

학습은 MASt3R 저자들이 제공하는 사전학습된 `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` 체크포인트를 활용합니다.

#### 2.3.3 학습 데이터 및 해상도

ScanNet++ 데이터셋으로 학습하며, 512×512 해상도에서 초당 4 프레임으로 장면 재구성이 가능하고 결과 splat은 실시간으로 렌더링됩니다.

---

### 2.4 성능 향상

실험 결과에서 Splatt3R는 MASt3R와 피드-포워드 스플래팅의 현재 최첨단 방법 모두를 능가합니다.

도입된 평균 오프셋은 모든 메트릭에서 성능을 소폭 개선합니다.

**정량적 비교 (ScanNet++ 기준, NoPoSplat 논문에서 인용):**

ScanNet++ 기준, Splatt3R의 PSNR은 약 11.6~13.3 수준으로 보고됩니다.

| 방법 | 포즈 필요 | PSNR (ScanNet++) |
|------|-----------|-----------------|
| pixelSplat | ✅ (GT 포즈) | ~18.4 |
| MVSplat | ✅ (GT 포즈) | ~17.1 |
| **Splatt3R** | ❌ (포즈 불필요) | ~13.3 |
| NoPoSplat | ❌ (포즈 불필요) | ~22.1 |

---

### 2.5 한계점

Splatt3R의 성능은 MASt3R의 예측 오류에 의해 제약을 받습니다. 이는 동결된(frozen) 아키텍처와 Gaussian 헤드만 학습하는 방식으로 인해 Gaussian 위치가 고정되는 문제에서 비롯됩니다.

또한 Splatt3R는 **메트릭 포즈 정답 데이터를 가진 데이터셋에서만 학습이 가능**하여, 다른 데이터셋(예: RealEstate10K)으로 재학습 시 훈련 자체가 실패하는 한계가 있습니다.

추가적인 한계로는 ① 얇거나 투명한 물체의 경우 Gaussian splat 표현이 이를 잘 포착하지 못할 수 있으며, ② 3D 연산의 계산 복잡도로 인해 고해상도 입력으로의 확장이 어렵고, ③ 두 장 이상의 입력 이미지를 처리하는 방법으로 확장이 아직 필요하다는 점이 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 기존 일반화 전략

일반화를 위해, Splatt3R는 MASt3R라는 "파운데이션(foundation)" 3D 기하 재구성 방법 위에 구축되어 3D 구조와 외양 모두를 처리할 수 있도록 확장됩니다.

Splatt3R는 캘리브레이션 되지 않은 스테레오 이미지에서 카메라 내부/외부 파라미터나 깊이 정보 없이 3D Gaussian Splat을 생성하는 피드-포워드 일반화 모델이며, MASt3R 아키텍처를 Loss Masking 전략과 결합하는 것만으로 광역 베이스라인에서 3D 외양과 기하 모두를 정확하게 재구성할 수 있음을 발견했습니다.

### 3.2 한계 및 향후 일반화 개선 방향

#### 문제 1: 동결된 인코더로 인한 병목

Splatt3R의 성능은 동결된 아키텍처와 Gaussian 헤드만 학습하는 방식에 기인하여 Gaussian 위치가 고정되는 문제로 제약됩니다. 이에 반해 렌더링 지도학습을 통한 엔드-투-엔드(end-to-end) 학습 방식은 Gaussian 위치와 속성을 동시에 최적화합니다.

**개선 방향**: MASt3R 인코더를 미세조정(fine-tuning)하거나 LoRA와 같은 파라미터 효율적 학습 기법을 적용하면 도메인 적응력이 향상될 수 있습니다.

#### 문제 2: 메트릭 포즈 데이터 의존성

현재 Splatt3R는 메트릭 포즈 정답 데이터가 있는 데이터셋에서만 학습 가능한 제약이 있습니다.

**개선 방향**: RealEstate10K, ACID 등 다양한 데이터셋으로 학습이 가능하도록 상대 포즈 기반 손실 함수 설계가 필요합니다.

#### 문제 3: 두 장 이미지로의 제한

두 장 이상의 입력 이미지를 처리할 수 있도록 방법을 확장하면 견고성과 재구성 품질이 향상될 수 있습니다.

**개선 방향**: multi-view attention 메커니즘을 통해 임의 개수의 이미지를 처리할 수 있는 아키텍처로 확장이 가능합니다.

#### 문제 4: 후속 연구에서의 일반화 비교

NoPoSplat(2024)은 Splatt3R 대비 아웃-오브-디스트리뷰션(out-of-distribution) 데이터에서 우수한 성능을 보이는데, 이는 네트워크 구조에서 최소한의 기하적 사전(geometric prior)을 사용하여 다양한 유형의 장면에 효과적으로 적응하기 때문입니다.

---

## 4. 후속 연구에 미치는 영향 및 고려사항

### 4.1 후속 연구에 미치는 영향

#### ① 파운데이션 모델 기반 3D 재구성 패러다임 확립

Splatt3R는 MASt3R/DUSt3R와 같은 파운데이션 모델을 렌더링 가능한 3D 표현으로 확장하는 **새로운 패러다임**을 제시했습니다. 이미 이 방향으로 다수의 후속 연구가 등장하고 있습니다:

**GSemSplat (2024)**: Splatt3R 아키텍처를 기반으로, 씬별 최적화나 밀집 이미지 컬렉션, 캘리브레이션 없이 3D Gaussian에 연결된 오픈-어휘 시맨틱 표현을 학습하는 프레임워크를 도입했습니다.

**FreeSplatter (2024)**: 비캘리브레이션 스파스-뷰 이미지에서 고품질 3D Gaussian을 생성하면서 수 초 내에 카메라 파라미터를 추정하는 확장 가능한 피드-포워드 프레임워크를 제안했습니다.

#### ② 손실 마스킹 전략의 광범위한 채택

FreeSplatter를 비롯한 후속 연구들은 Splatt3R가 제안한 **타겟-뷰 마스킹 전략**을 채택하여, 입력 뷰에서 보이지 않는 영역이 학습에 부정적 영향을 미치지 않도록 합니다.

#### ③ 포즈-프리 3D-GS 연구의 활성화

Splatt3R의 제안 이후 NoPoSplat(2024)과 같이 비포즈 스파스 멀티-뷰 이미지로부터 3D Gaussian으로 파라미터화된 3D 장면을 재구성하는 피드-포워드 모델 연구가 활발히 전개되고 있습니다.

---

### 4.2 관련 최신 연구 비교 분석 (2020년 이후)

| 연구명 | 연도 | 포즈 필요 | 방법 | 특징 |
|--------|------|-----------|------|------|
| **NeRF** (Mildenhall et al.) | 2020 | ✅ | 암묵적 MLP | 고품질, 느린 렌더링 |
| **DUSt3R** (Wang et al.) | 2023 | ❌ | 포인트맵 회귀 | NVS 불가, 재구성만 |
| **3D-GS** (Kerbl et al.) | 2023 | ✅ | 명시적 Gaussian | 실시간 렌더링, SfM 필요 |
| **pixelSplat** (Charatan et al.) | 2024 | ✅ | Feed-forward GS | 에피폴라 기하 활용 |
| **MASt3R** (Leroy et al.) | 2024 | ❌ | ViT + Cross-attn | 포인트클라우드만 출력 |
| **Splatt3R** (Smart et al.) | 2024 | ❌ | MASt3R + GS Head | NVS 가능, 실시간 |
| **NoPoSplat** (Ye et al.) | 2024 | ❌ | Canonical Space | 더 높은 PSNR |
| **FreeSplatter** (Xu et al.) | 2024 | ❌ | E2E Transformer | 포즈 추정도 동시 수행 |
| **GSemSplat** | 2024 | ❌ | Splatt3R 확장 | 시맨틱 3D-GS |

FlowCam과 같은 유사 연구는 카메라 포즈 사전 계산 필요성을 제거하나 순차적 입력이 필요하고 제한적인 렌더링 성능을 보이는 반면, Splatt3R는 MASt3R와 3D Gaussian을 통합하여 카메라 사전 처리 없이 더 넓은 베이스라인을 효과적으로 처리합니다.

---

### 4.3 앞으로의 연구 시 고려사항

#### 🔬 기술적 고려사항

1. **엔드-투-엔드 학습**: Splatt3R의 성능은 동결된 아키텍처와 Gaussian 헤드만 학습하는 방식으로 인해 MASt3R 예측 오류에 제약됩니다. 반면, 렌더링 지도학습을 통한 엔드-투-엔드 학습은 Gaussian 위치와 속성을 동시에 최적화할 수 있습니다. 향후 연구에서는 엔드-투-엔드 학습의 불안정성을 극복하는 안정적 학습 전략이 중요합니다.

2. **다양한 데이터셋으로의 확장**: Splatt3R는 메트릭 포즈 정답이 있는 데이터셋에서만 학습 가능하다는 제약이 있으므로, 다양한 데이터 소스 활용을 위한 손실 함수 재설계가 필요합니다.

3. **멀티-뷰 확장**: 두 장 이상의 이미지 처리를 위한 확장 아키텍처 설계 (예: Splatt3R → N-view 처리)

4. **동적 장면 처리**: 현재는 정적 장면만 처리 가능하므로, 동적 객체를 포함한 장면으로의 확장이 필요합니다.

5. **불확실성 정량화**: 신뢰도 기반 Gaussian 예측에서 불확실성을 명시적으로 모델링하면 견고성이 향상됩니다.

#### 🎯 응용 연구 고려사항

6. **다운스트림 태스크로의 확장**: GSemSplat이 Splatt3R 아키텍처를 시맨틱 3D 이해로 확장한 것처럼, 깊이 추정, 세그멘테이션, 로보틱스 등 다양한 응용 연구와의 결합 가능성을 탐구해야 합니다.

7. **실용적 배포**: Splatt3R는 512×512 해상도에서 4 FPS로 장면을 재구성하고 결과 splat을 실시간으로 렌더링할 수 있으므로, 모바일/엣지 디바이스에서의 경량화 연구가 중요한 연구 방향입니다.

8. **더 큰 사전 학습 데이터**: ScanNet++만으로는 일반화 한계가 있으므로, 더 다양한 실내외 데이터로의 학습이 필요합니다.

---

## 참고 자료 (References)

1. **공식 프로젝트 페이지**: https://splatt3r.active.vision/
2. **arXiv 논문**: Smart, B., Zheng, C., Laina, I., & Prisacariu, V. A. (2024). *Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs*. arXiv:2408.13912. https://arxiv.org/abs/2408.13912
3. **공식 GitHub**: https://github.com/btsmart/splatt3r
4. **Oxford Active Vision Lab 페이지**: https://www.robots.ox.ac.uk/ActiveVision/Publications/smart_etal_arxiv2024/
5. **Semantic Scholar**: https://www.semanticscholar.org/paper/Splatt3R:-Zero-shot-Gaussian-Splatting-from-Image-Smart-Zheng/44d8c25fecb5628382854ecfe92e3b1015749e7c
6. **ResearchGate**: https://www.researchgate.net/publication/383428857
7. **HuggingFace Paper Page**: https://huggingface.co/papers/2408.13912
8. **관련 비교 논문 - NoPoSplat** (Ye et al., 2024): arXiv:2410.24207 https://arxiv.org/abs/2410.24207
9. **관련 비교 논문 - FreeSplatter** (Xu et al., 2024): arXiv:2412.09573 https://arxiv.org/abs/2412.09573
10. **후속 확장 논문 - GSemSplat** (Wang et al., 2024): arXiv:2412.16932 https://arxiv.org/abs/2412.16932
11. **3D Gaussian Splatting 원 논문**: Kerbl, B. et al. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. https://github.com/graphdeco-inria/gaussian-splatting

> ⚠️ **정확도 안내**: 본 분석은 공개된 arXiv 논문, 공식 프로젝트 페이지, GitHub 코드 및 이를 인용한 후속 논문들을 기반으로 작성되었습니다. 수식의 세부 계수(예: $\lambda$ 값)나 ablation 수치는 전문 접근이 제한된 관계로 논문 본문의 정확한 수치와 일부 차이가 있을 수 있으며, 정확한 수치는 원문 PDF를 직접 확인하시기 바랍니다.
