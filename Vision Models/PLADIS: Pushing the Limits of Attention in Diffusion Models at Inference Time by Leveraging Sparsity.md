
# PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity

> **논문 정보**
> - **저자**: Kwanyoung Kim, Byeongsu Sim
> - **arXiv**: [2503.07677](https://arxiv.org/abs/2503.07677) (2025년 3월)
> - **학회**: ICCV 2025 (accepted)
> - **공식 코드**: [GitHub - cubeyoung/PLADIS](https://github.com/cubeyoung/PLADIS)
> - **프로젝트 페이지**: [cubeyoung.github.io/pladis-proejct](https://cubeyoung.github.io/pladis-proejct/)

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 Kwanyoung Kim과 Byeongsu Sim이 저술하였으며, **Sparse Attention 메커니즘**을 활용하여 Diffusion Model의 성능을 향상시키는 새로운 기법을 제안합니다.

### 🔑 핵심 주장

Diffusion Model은 CFG(Classifier-Free Guidance)와 같은 Guidance 기법을 통해 고품질의 조건부 샘플을 생성하는 데 인상적인 성과를 보여왔습니다. 그러나 기존 방법들은 추가 학습이나 NFE(Neural Function Evaluations)를 요구하여 guidance-distilled 모델과 호환되지 않으며, 목표 레이어를 식별해야 하는 휴리스틱 접근 방식에 의존합니다. PLADIS는 추가 학습이나 NFE 없이 inference 시 cross-attention 레이어에서 softmax와 그 sparse 대응물을 사용하여 query-key 상관관계를 외삽(extrapolate)합니다.

### 🏆 주요 기여 (4가지)

논문의 핵심 기여는 다음과 같습니다:
1. **PLADIS 제안**: Diffusion Model의 cross-attention을 sparse와 dense attention 간 외삽으로 대체하는 방법 제안
2. **이론적 분석**: SHN(Sparse Hopfield Network) 관점에서의 오류 경계(error bound) 및 중간 희소성(intermediate sparsity)에서의 노이즈 견고성(noise robustness) 분석
3. **SHN 최초 적용**: SHN 관점에서 Diffusion Model을 개선한 최초의 논문
4. **범용성**: 추가 학습이나 NFE 없이 다른 guidance 방법 및 guidance-distilled 모델과 결합 가능


---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

기존 guidance 방법들(CFG, PAG, SEG)은 null condition, identity matrix로의 self-attention 교란, 흐릿한 attention 가중치 등의 불필요한 경로로 인해 추가적인 inference 단계가 필요합니다. 반면 PLADIS는 스케일링 팩터 $\lambda$를 사용하여 모든 cross-attention 모듈 내에서 sparse와 dense attention을 동시에 계산함으로써 추가 inference 경로를 피합니다. 또한 PLADIS는 cross-attention 모듈을 단순히 교체함으로써 기존 guidance 접근법과 쉽게 통합될 수 있습니다.

**기존 방법의 문제점 정리:**

| 방법 | 문제점 |
|------|--------|
| CFG | 추가 NFE(null-condition 필요), guidance-distilled 모델과 비호환 |
| PAG | Self-attention을 identity matrix로 교란 → 부작용 발생 |
| SEG | 흐릿한 attention weight 사용 → 추가 연산 필요 |
| 기타 attention 수정 방법 | target layer 탐색을 위한 hyperparameter 검색 필요 |

---

### 2-2. 제안하는 방법 및 수식

#### (1) Standard (Dense) Attention

기존 Transformer의 standard attention 메커니즘:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

#### (2) $\alpha$-Entmax: Sparse Attention의 기반

$\alpha$-Entmax는 다음과 같이 정의됩니다:

$$\alpha\text{-Entmax}(z) = \arg\max_{p \in \Delta^M}\left[\langle p, z \rangle - H_\alpha(p)\right]$$

여기서 $\alpha$는 희소성(sparsity)을 제어합니다. $\alpha=1$이면 dense probability mapping인 softmax와 동일하고, $\alpha$가 2에 가까워질수록 $\alpha$-Entmax의 출력은 점점 더 희소해집니다.

이 공식에서 $\tau/(\alpha-1)$보다 작은 항목은 0으로 매핑되어 희소성이 달성됩니다. $\alpha$-Entmax에서 비-제로 항목의 수를 $\kappa(z)$로 표기합니다. $\alpha=2$(sparsemax)의 경우 정렬 알고리즘으로 효율적으로 계산됩니다.

#### (3) Sparse Attention (Sparse Hopfield Network 기반)

SHN(Sparse Hopfield Network)으로부터 유도된 Sparse Attention:

$$\text{At}_\alpha(Q, K, V) = \alpha\text{-Entmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

특히 실용적인 케이스:
- $\alpha = 1$: Softmax (dense)
- $\alpha = 1.5$: 1.5-Entmax (정확한 해 존재, 효율적 계산 가능)
- $\alpha = 2$: Sparsemax (정렬 알고리즘으로 효율적 계산 가능)

#### (4) PLADIS의 핵심 수식: Sparse-Dense Extrapolation

CFG, PAG, SEG 같은 guidance 방법에서 영감을 받아, PLADIS는 dense와 sparse attention에서 query-key 상관관계를 외삽합니다. $\lambda$는 하이퍼파라미터로, sparse attention 효과가 강조되는 정도를 결정합니다.

PLADIS의 핵심 공식 (cross-attention에 적용):

$$\text{PLADIS-Attn}(Q, K, V) = \text{At}_\alpha(Q, K, V) + \lambda\!\left(\text{At}_\alpha(Q, K, V) - \text{Attn}(Q, K, V)\right)$$

이를 정리하면:

$$\text{PLADIS-Attn}(Q, K, V) = (1+\lambda)\,\text{At}_\alpha(Q, K, V) - \lambda\,\text{Attn}(Q, K, V)$$

$\lambda > 1$일 때 PLADIS가 적용됩니다. Sparsity 정도 $\alpha$ ($1 < \alpha \leq 2$)는 또 다른 하이퍼파라미터이며, 효율적인 알고리즘이 존재하는 $\alpha = 1.5$와 $\alpha = 2$ 두 가지 옵션만 사용합니다.

#### (5) Noise Robustness: 이론적 근거

Diffusion 과정에서 내재적인 가우시안 노이즈 손상으로 인해 query 행렬은 자연스럽게 교란됩니다. 이로 인해 Sparse Attention의 견고성(robustness)이 이점을 발휘하고, 이 장점은 텍스트 정렬 및 전반적인 샘플 품질 향상으로 이어집니다.

논문은 $1 < \alpha \leq 2$ 범위에서 검색 오류 경계(retrieval error bound)를 도입하여, Diffusion Model에서 sparse cross-attention의 노이즈 견고성 간의 연결을 확립하고, T2I 생성을 위한 cross-attention 모듈의 희소성에 대한 심층 분석을 제공합니다.

---

### 2-3. 모델 구조

PLADIS는 U-Net 및 Transformer 기반의 사전 학습된 모델 모두를 부스팅합니다.

PLADIS는 Diffusion Model의 cross-attention을 sparse와 dense cross-attention 간의 외삽으로 조정된 attention 메커니즘으로 대체하며, SHN에 대한 이해를 바탕으로 철저한 이론적 분석을 제공합니다.

**지원 백본 모델:**
PLADIS는 CFG, PAG, SEG 등의 guidance sampling 방법, DMD2, SDXL-Lightning, Hyper-SDXL 등의 guidance-distilled 모델, 그리고 Stable Diffusion 1.5, SANA, FLUX 등 다양한 백본 모델에서 검증되었습니다.

SDXL 및 FLUX와 같은 고급 모델 백본과 호환되며, ControlNet 등의 downstream task도 지원합니다.

---

### 2-4. 성능 향상

광범위한 실험을 통해 텍스트 정렬과 인간 선호도(human preference)에서 주목할 만한 개선을 보여주며, 고효율적이고 범용적으로 적용 가능한 솔루션을 제공합니다.

정성적 비교에서 CFG, PAG, SEG 등의 guidance sampling 방법과 DMD2, SDXL-Lightning, Hyper-SDXL 등의 guidance-distilled 모델, 그리고 SD 1.5, SANA와의 비교에서 PLADIS는 모든 guidance 기법과 호환되며 다양한 백본에서 guidance-distilled 모델도 지원합니다. 추가 학습이나 추가 inference 없이 그럴듯하고 향상된 텍스트 정렬을 제공합니다.

**기존 방법 대비 주요 이점:**

| 특성 | CFG | PAG | SEG | PLADIS |
|------|-----|-----|-----|--------|
| 추가 NFE 필요 | ✅ | ✅ | ✅ | ❌ |
| 추가 학습 필요 | 경우에 따라 | ❌ | ❌ | ❌ |
| Guidance-distilled 호환 | ❌ | ❌ | ❌ | ✅ |
| Target layer 탐색 필요 | ❌ | ✅ | ✅ | ❌ |
| 수식적 이론 근거 | 경험적 | 경험적 | 부분적 | ✅ (SHN 기반) |

---

### 2-5. 한계

논문에서 명시적으로 언급된 한계는 다음과 같습니다:

1. **하이퍼파라미터 민감성**: $\lambda$와 $\alpha$ 값의 조정이 필요하며, 모델 백본 및 사용 케이스에 따라 달라질 수 있습니다. pladis_scale은 PLADIS의 스케일 파라미터로, 일반적으로 1.5와 2.0으로 고정하며 모델 백본과 케이스에 맞게 조정해야 합니다.

2. **Cross-Attention에 한정**: PLADIS는 주로 cross-attention에 적용되며, self-attention에 적용 시 효과가 제한적입니다.

3. **추가 계산 비용**: PLADIS는 모든 cross-attention 모듈 내에서 sparse와 dense attention을 모두 계산하여 추가 inference 경로를 피하지만, 두 attention을 병렬로 계산해야 하므로 단순 dense attention 대비 소폭의 연산 비용이 추가됩니다.

4. **Non-commercial 제한**: FLUX.1-dev 모델의 경우 라이선스에 따라 엄격히 비상업적 연구 목적으로만 사용되었으며, 추론 중 성능을 평가하고 향상시키기 위해서만 사용되었습니다.

---

## 3. 일반화 성능 향상 가능성

논문에서는 방법의 일반화 가능성(generalizability)을 강조합니다. attention 모듈을 수정하는 다른 방법들은 target layer에 대한 하이퍼파라미터 검색이 필요합니다. 반면 PLADIS는 해당 수식을 **모든 cross-attention 레이어에 적용**하는 것으로 충분하며, 이는 다른 케이스로의 확장을 더 쉽게 만듭니다.

이를 검증하기 위해 다양한 target layer에 대한 ablation study를 수행하였으며, 모든 레이어에 적용하는 것이 최적의 선택임을 확인하였습니다.

### 일반화의 주요 근거

**① 다양한 백본 아키텍처 지원**
PLADIS는 U-Net과 Transformer 기반 사전 학습된 모델 모두를 부스팅하며, sparse attention을 활용합니다.

**② Guidance-distilled 모델과의 호환성**
Sparse Attention의 노이즈 견고성을 활용하여 PLADIS는 텍스트-이미지 Diffusion Model의 잠재력을 발휘시키며, guidance-distilled 모델을 포함한 guidance 기법과 원활하게 통합됩니다.

**③ Multimodal 도메인으로의 확장 가능성**
PLADIS는 multimodal 생성 및 정렬에 관한 미래 연구의 길을 열며, cross-attention을 통한 정밀한 multimodal 정렬이 필요한 도메인에서의 잠재적 응용 가능성이 있습니다.

**④ 이론적 보장을 통한 일반화**
논문은 SHN(Sparse Hopfield Network)에 대한 이해를 바탕으로 중간 희소성 케이스에 대한 오류 경계 및 노이즈 견고성을 제안하며, SHN 관점에서 Diffusion Model을 적용하고 개선한 최초의 논문입니다.

**⑤ Temperature 조정과의 시너지**
1.5-Entmax에서 온도를 낮추면 시각적 품질과 텍스트 정렬 측면에서 생성 품질이 일관되게 향상됩니다. 매우 낮은 온도의 Softmax는 0보다 큰 intensity를 가진 거의 동일한 sparse transformation을 만들며, 이는 온도를 낮추는 것이 $1 \leq \alpha \leq 2$에서의 모든 $\alpha$-Entmax 변환에 유익함을 시사합니다.

**⑥ Downstream Tasks**
SDXL 및 FLUX와 같은 고급 모델 백본과 호환되며, ControlNet과 같은 downstream task도 지원합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 연도 | 핵심 아이디어 | PLADIS와의 차이 |
|------|------|---------------|-----------------|
| **DDPM** (Ho et al.) | 2020 | Denoising 기반 생성 모델 | PLADIS의 기반 프레임워크 |
| **CFG** (Ho & Salimans) | 2022 | Null-condition을 이용한 guidance | 추가 NFE 필요, PLADIS가 이를 보완 |
| **LDM/Stable Diffusion** (Rombach et al.) | 2022 | Latent space에서의 diffusion | PLADIS가 적용 가능한 U-Net 백본 |
| **PAG** (Ahn et al.) | 2024 | Self-attention을 identity matrix로 교란 | PAG는 identity attention을 사용한 이미지 안내 방식이나, 이로 인한 부작용으로 이미지의 시각적 구조 및 색상 분포가 변경되는 문제가 있습니다. PLADIS는 cross-attention을 활용하여 이러한 부작용을 피함 |
| **SEG** | 2024 | Blurred attention 기반 guidance | 추가 inference 경로 필요, PLADIS는 단일 forward pass |
| **Sparse Guidance (SG)** (Krause et al.) | 2025 | Token-level sparsity 기반 guidance | SG는 conditional dropout 대신 token-level sparsity를 신호로 사용하여 확산 모델을 안내하며, 조건부 예측의 높은 분산을 더 잘 보존합니다. SG는 token-sparsity에 초점, PLADIS는 attention-weight sparsity에 초점 |
| **PLADIS** (Kim & Sim) | 2025 | Cross-attention의 sparse-dense 외삽 | 추가 NFE/학습 없이 모든 guidance 방법과 호환 |

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 📌 미래 연구에 미치는 영향

**① Inference-time Optimization의 새로운 패러다임**

Sparse Attention의 노이즈 견고성을 활용하여 PLADIS는 텍스트-이미지 Diffusion Model의 잠재력을 발휘시키며, 이전에 어려움을 겪었던 영역에서 새로운 효과성으로 탁월함을 발휘할 수 있게 합니다. 이는 **학습 없이 inference 시에만 모델을 개선**하는 새로운 연구 방향을 제시합니다.

**② Sparse Hopfield Network의 확장 응용**

SHN 관점에서 Diffusion Model을 개선한 최초의 논문으로서, 이후 SHN 이론을 Video generation, 3D generation, Audio synthesis 등 다른 생성 모델에 적용하는 연구의 토대가 될 것입니다.

**③ Multimodal 정렬 연구로의 파급 효과**

PLADIS는 multimodal 생성 및 정렬에 관한 미래 연구의 길을 열며, cross-attention을 통한 정밀한 multimodal 정렬이 필요한 도메인에서의 잠재적 응용 가능성이 있습니다. 이는 VQA, Image Captioning, Text-to-Video 등 다양한 multimodal task에 영향을 줄 것입니다.

**④ Guidance-distilled 모델의 성능 한계 극복**

PLADIS는 CFG, PAG, SEG 등 guidance 방법과 DMD2, SDXL-Lightning, HyperSDXL 등 guidance-distilled 모델과 원활하게 통합됩니다. PLADIS는 추가 학습이나 추가 inference 단계 없이 텍스트 정렬과 샘플 생성 품질을 크게 향상시킵니다.

---

### ⚠️ 향후 연구 시 고려할 점

1. **Sparsity 정도의 최적화**: $\alpha$와 $\lambda$ 값이 태스크, 모델 크기, 도메인에 따라 다를 수 있으므로, 자동 하이퍼파라미터 탐색 방법(예: AutoML, Bayesian Optimization)과의 결합 연구가 필요합니다.

2. **Video/3D Diffusion 모델로의 확장**: Temporal attention이 포함된 Video diffusion model에서 temporal cross-attention에 PLADIS를 적용할 때의 효과 및 이론적 분석 연구가 필요합니다.

3. **Self-attention과의 통합**: 그럼에도 불구하고 dense alignment에서 온도를 낮추는 것만으로는 불충분하며, PLADIS를 사용하든 사용하지 않든 sparse attention이 모두 필요합니다. 1.5-Entmax를 활용한 PLADIS는 다양한 $\tau$ 값에서 성능이 수렴하므로 최적 하이퍼파라미터를 찾는 시간이 필요 없습니다. 이는 Diffusion Model에서 sparse cross-attention의 노이즈 견고성이 생성 성능에 중요함을 보여줍니다.

4. **이론적 확장**: 현재 $1 < \alpha \leq 2$ 범위에서의 오류 경계 분석에 초점을 맞추고 있으며, 다른 $\alpha$ 범위 및 다른 sparse 연산자(top-k 등)에 대한 이론적 보장 연구가 필요합니다.

5. **Large-scale 모델에서의 효율성 분석**: FLUX, SD3 등 수십억 파라미터 규모의 모델에서 sparse-dense 이중 계산에 따른 메모리 및 연산 비용 최적화가 필요합니다.

6. **Downstream Task 일반화**: PLADIS는 multimodal 생성 및 정렬에서 cross-attention을 통한 정밀한 multimodal 정렬이 필요한 도메인에 대한 잠재적 응용이 있습니다. 의료 영상, 위성 이미지 등 특수 도메인에서의 검증이 추가로 필요합니다.

---

## 📚 참고 자료 및 출처

1. **arXiv 논문 원문**: Kwanyoung Kim, Byeongsu Sim. "PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity." arXiv:2503.07677 (2025). https://arxiv.org/abs/2503.07677

2. **ICCV 2025 논문 HTML**: https://arxiv.org/html/2503.07677v2

3. **공식 프로젝트 페이지**: https://cubeyoung.github.io/pladis-proejct/

4. **공식 GitHub 코드**: https://github.com/cubeyoung/PLADIS (ICCV'25)

5. **Hugging Face 논문 페이지**: https://huggingface.co/papers/2503.07677

6. **ResearchGate PDF**: https://www.researchgate.net/publication/389748471

7. **ICCV 2025 보충 자료**: https://openaccess.thecvf.com/content/ICCV2025/supplemental/Kim_PLADIS_Pushing_the_ICCV_2025_supplemental.pdf

8. **Moonlight Literature Review**: https://www.themoonlight.io/en/review/pladis-...

9. **SEG (Smoothed Energy Guidance)**: NeurIPS 2024 Proceedings. https://proceedings.neurips.cc/paper_files/paper/2024/file/7b3f7b6670fdab2933411b5b922cdcc3

10. **Sparse Guidance (SG)** (Krause et al., 2025): arXiv:2601.01608. https://arxiv.org/pdf/2601.01608
