# SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer

---

## 1. Executive Summary (10문장 이내)

고압축률의 효율적인 이미지 토큰화는 생성 모델 훈련에 있어 여전히 중요한 과제로 남아있습니다.  
SoftVQ-VAE는 소프트 카테고리 사후확률(soft categorical posterior)을 활용해 여러 코드워드를 하나의 잠재 토큰에 집계함으로써 잠재 공간의 표현 용량을 크게 늘리는 연속(continuous) 이미지 토크나이저입니다.  
Transformer 기반 아키텍처에 적용했을 때, 이 접근법은 256×256 및 512×512 이미지를 단 32개 혹은 64개의 1차원 토큰만으로 압축합니다.  
SoftVQ-VAE는 일관되고 고품질의 재구성을 보여줄 뿐만 아니라, 다양한 디노이징 기반 생성 모델에서 최신 수준(SOTA)이면서도 훨씬 빠른 이미지 생성 결과를 달성합니다.  
놀랍게도 SoftVQ-VAE는 256×256 이미지 생성 시 추론 처리량을 최대 18배, 512×512 이미지 생성 시 최대 55배 향상시키면서도 SiT-XL 기준 FID 1.78 및 2.21의 경쟁력 있는 점수를 달성합니다.  
또한 생성 모델의 훈련 반복 횟수를 2.3배 줄이면서도 비슷한 성능을 유지하여 훈련 효율성도 향상시킵니다.  
핵심 아이디어는 기존 VQ-VAE의 argmax 기반 hard quantization을 softmax 기반의 완전 미분 가능한(fully-differentiable) 연산으로 대체한 것입니다.  
이를 통해 SoftVQ-VAE는 완전히 미분 가능해져 인코더와 코드북을 재구성 손실(및 기타 손실)로부터 직접 최적화할 수 있으며, 잠재 공간에 다양한 형태의 정규화를 더 쉽게 적용하여 품질을 크게 향상시킵니다.  
완전 미분 가능한 설계와 의미론적으로 풍부한 잠재 공간을 바탕으로, 실험은 SoftVQ-VAE가 생성 품질을 저해하지 않으면서 효율적인 토큰화를 달성함을 보여주며, 이는 더 효율적인 생성 모델로 가는 길을 열어줍니다.  
저자들은 CMU, AMD, William & Mary, MBZUAI 소속이며 CVPR 2025에 게재되었습니다.

### 1-1. 연구의 목적과 필요성

- **목적**: 이미지 생성 모델(Diffusion, Flow-matching, Autoregressive 등)의 백본으로 사용되는 토크나이저가 압축률과 재구성/생성 품질 사이에서 트레이드오프를 겪는 문제를 해결하고자 함. 특히 1차원(1D) 토큰 시퀀스로 이미지를 극도로 압축하면서도 생성 품질을 유지·향상시키는 것이 목표.
- **필요성**: 기존 VQ-VAE 계열은 codebook에서 가장 가까운 하나의 codeword만 선택하는 hard nearest-neighbor 방식(argmin)을 사용하기 때문에 (1) 미분 불가능하여 straight-through estimator 등 편법적 그래디언트 근사가 필요하고, (2) codebook collapse(일부 코드워드만 사용됨) 문제가 발생하며, (3) 토큰 수를 줄이면 정보 손실이 급격히 커지는 한계가 있음. 효율적인 고압축 이미지 토큰화는 생성 모델 훈련의 핵심 난제로 지적됨.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거(논문 인용) | 위치 |
|---|---|---|---|
| 1 | Soft categorical posterior로 다중 코드워드 집계가 가능해져 표현 용량 증가 | "instead of the conventional one-to-one mapping between tokens and codewords in VQ-VAE, our approach enables the adaptive aggregation of multiple codewords into each latent token, substantially increasing the representation capacity of the latent space." | Sec.3, Fig.2 |
| 2 | 32~64개의 1D 토큰만으로 고압축 달성 | "We apply SoftVQ-VAE to the Transformer auto-encoder architecture...and we succeed in using much fewer 1-dimensional latent tokens (i.e., 32 and 64) for both the reconstruction and subsequent generation tasks." | Sec.3, Table 1 |
| 3 | 동일 파라미터 대비 TiTok보다 재구성·생성 모두 우월 | "SoftVQ-S significantly outperforms both TiTok variants at 64 tokens, achieving better reconstruction quality with an rFID of 1.03 compared to 1.25, and better generation performance with a gFID of 11.24 and IS of 89.4" | Table, Sec.4.2 |
| 4 | 토큰 수를 줄여도 성능 저하가 완만함(Less-lossy) | "SoftVQ-S exhibits remarkably consistent performance across different compression ratios. When reducing tokens from 256 to 32, SoftVQ-S maintains a minimal degradation in rFID from 0.80 to 1.24, significantly outperforming both VQ-S, from 1.45 to 10.97, and AE-S, from 1.15 to 2.01." | Table 2 |
| 5 | KL 토크나이저 대비 32배 적은 토큰으로도 비슷하거나 더 나은 생성 성능 | "with only 32 tokens, the generation performance of DiT-XL and SiT-XL trained on all variants of the proposed SoftVQ significantly outperforms DiT-XL/2 and SiT-XL/2 with KL tokenizers of 1024 tokens...DiT-XL with SoftVQ-L of 32 tokens presents a 0.62 improvement on FID" | Table 7, Sec.4.3 |
| 6 | 재구성 품질(rFID) 향상이 반드시 생성 품질(gFID) 향상으로 이어지지 않음 | "We reveal that a better rFID does not necessarily translate to a better gFID." | Table 4, Sec.4.4 |
| 7 | Representation alignment(REPA 방식) 적용 시 잠재공간의 의미 구조 향상 | "Thanks to the fully-differentiable property of SoftVQ-VAE, we can now directly impose regularization on the latent space. Inspired by the recent work of REPA, we propose aligning the representations of latent codes with pre-trained vision encoders" | Sec.3.3, Fig.2 우하단 |
| 8 | DINOv2 초기화+정렬 조합이 최적 | "The encoder initialization and alignment with DINOv2-B achieve superior performance with rFID 0.88 and IS 103.4, compared to using either component alone." | Table 4 |
| 9 | GMMVQ(가우시안 혼합 변형)은 SoftVQ와 유사한 성능 | "While providing the additional benefit of interpretable Gaussian components in the latent space, GMMVQ achieves reconstruction quality and downstream generation performance similar to SoftVQ in our experiments, and thus we mainly adopt the simpler SoftVQ throughout this paper." | Appendix B.2 |

---

## 2-1. 상세 설명 (문제·방법·수식·구조·성능·한계)

### (1) 해결하고자 하는 문제
기존 이미지 토크나이저(VQ-VAE, KL-VAE 등)는 다음과 같은 문제가 있음:
- **압축률과 품질의 트레이드오프**: 토큰 수를 줄이면(높은 압축) 재구성/생성 품질이 급격히 저하됨.
- **VQ-VAE의 non-differentiability**: hard argmin 연산으로 인해 straight-through gradient 근사와 codebook/commitment loss 등 다수의 하이퍼파라미터 튜닝이 필요.
- **Codebook collapse**: 일부 코드워드만 반복 사용되어 표현력 저하.
- **2D 토큰화의 비효율성**: 기존 방식은 이미지 패치 수에 비례하는 토큰 그리드를 사용해, 근본적으로 압축률에 한계가 있음.

### (2) 제안하는 방법 (수식 포함)

**① Soft Quantization (posterior)**

기존 VQ-VAE의 hard assignment:

```math
z_q = C_{k^*}, \quad k^* = \arg\min_k \|\hat{z} - C_k\|^2
```

SoftVQ-VAE는 이를 soft categorical posterior로 대체:
$$q_\phi(z|x) = \mathrm{Softmax}\left(-\frac{\|\hat{z} - C\|^2}{\tau}\right)$$

- $\hat{z}$: 인코더의 출력(연속 잠재 벡터)
- $C = \{C_1, ..., C_K\}$: 학습 가능한 코드북(K개의 코드워드)
- $\tau$: softmax의 온도(temperature) 파라미터 — 값이 작을수록 hard quantization에 가까워지고, 클수록 여러 코드워드가 고르게 섞임
- $q_\phi(z|x)$: 각 코드워드에 대한 소프트 할당 확률 (사후확률)

이후 최종 양자화된 잠재 토큰 $z_q$는 이 확률로 가중합된 코드워드들의 조합:
$$z_q = \sum_{k=1}^{K} q_\phi(z_k|x)\, C_k$$

이 연산은 argmax가 아닌 미분 가능한 가중합이므로 **end-to-end backpropagation**이 가능함.

**② KL Divergence 정규화**

SoftVQ에서는 VQ와 마찬가지로 사전분포(prior)가 K개의 학습 가능한 코드워드에 대한 균등분포라고 가정하고, 사후확률을 소프트맥스 확률로 둠:

$$\mathcal{L}_{kl}(q_\phi(\mathbf{z}) \| p(\mathbf{z})) = \int q_\phi(\mathbf{z})(\log q_\phi(\mathbf{z}) - \log p(\mathbf{z}))\, d\mathbf{z} = H(q_\phi(\mathbf{z})) - H(q_\phi(\mathbf{z}), p(\mathbf{z}))$$

- $H(q_\phi(\mathbf{z}))$: 사후확률의 엔트로피(코드워드 사용의 다양성을 유도)
- $H(q_\phi(\mathbf{z}), p(\mathbf{z}))$: 균등 사전분포 $p(\mathbf{z}) \sim \mathcal{U}(0,K)$ 와의 교차 엔트로피
- 이 항은 codebook의 균등한 활용(collapse 방지)을 유도하는 정규화 역할을 함

**③ Representation Alignment Loss**

완전 미분 가능하다는 특성 덕분에 잠재 공간에 직접 정규화를 가할 수 있으며, REPA에서 영감을 받아 사전학습된 비전 인코더(DINOv2 등)와 잠재 코드의 표현을 정렬하는 방법을 제안함.

일반적 형태:

$$\mathcal{L}_{\text{align}} = -\,\text{sim}\big(z, f_{\text{DINOv2}}(x)\big)$$

- $z$: SoftVQ 인코더의 잠재 토큰
- $f_{\text{DINOv2}}(x)$: 고정된(frozen) DINOv2 등 사전학습 모델의 특징
- $\text{sim}(\cdot,\cdot)$: 코사인 유사도 등 정렬 지표

**④ 전체 학습 목적함수(개념적)**

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{kl}\mathcal{L}_{kl} + \lambda_{\text{align}}\mathcal{L}_{\text{align}} + \mathcal{L}_{\text{perceptual/adv}}$$

(각 손실 항의 가중치 $\lambda$는 하이퍼파라미터)

### (3) 모델 구조

Transformer 인코더-디코더 아키텍처로, 이미지 토큰, 임의 길이의 잠재 토큰, 마스크 토큰으로 구성되며, 완전 미분 가능한 SoftVQ와 잠재 공간 표현 정렬을 포함함. 이미지 특징을 잠재로 직접 사용하는 대신, 별도의 학습 가능한 1D 토큰 집합을 초기화하고 이 토큰들을 재구성 및 이후 생성에 사용하며, self-attention 메커니즘을 통해 학습 가능한 토큰들이 서로 다른 이미지 토큰을 적응적으로 집계하여 SoftVQ의 잠재 토큰을 얻음. (Fig.2 참조)

주요 구성 요소:
1. **Patchify + ViT 인코더**: 이미지를 패치로 분할 후 Transformer 인코더 통과
2. **학습 가능한 1D 잠재 토큰(latent query tokens)**: TiTok과 유사하게 별도의 쿼리 토큰이 attention을 통해 이미지 정보를 압축
3. **SoftVQ 양자화 모듈**: 위 수식대로 소프트 할당
4. **Transformer 디코더**: 마스크 토큰과 결합해 이미지 재구성
5. (옵션) DINOv2 등으로 인코더 초기화 + 정렬 손실

### (4) 성능 향상

- 256×256 이미지 생성 시 추론 처리량 최대 18배, 512×512 이미지 생성 시 최대 55배 향상, SiT-XL 기준 FID 1.78/2.21 달성 (Table 1, Fig.1)
- 훈련 반복 횟수를 2.3배 줄이면서도 비슷한 성능 유지 (Sec.4)
- SiT-L이 SoftVQ 32토큰으로 학습했을 때 1024토큰의 KL 토크나이저 대비 gFID 5.9 개선 (Table 3/7)
- 64토큰의 (모델)이 CFG 없이 REPA를 사용한 SiT-XL/2보다 우수한 성능을 보임 (Table 7)

### (5) 한계
- 잠재 공간의 차원을 늘리면 재구성 성능은 크게 개선되지만, 학습 난이도 등의 이유로 생성 성능은 오히려 저하됨 (Sec.4.4) — 재구성과 생성 사이의 근본적 트레이드오프가 완전히 해소된 것은 아님.
- 더 나은 rFID가 반드시 더 나은 gFID로 이어지지 않음 — 평가지표 간 불일치는 여전히 존재.
- 후속 연구(MacTok, arXiv 2603.29634)에서는 "It outperforms SoftVQ-VAE by 0.69 gFID using 64 tokens"라고 보고하여, 이후 더 우수한 방법들이 빠르게 등장하고 있음을 시사함.

---

## 3. 페이지/Figure/Table 표기 (요약)

| 내용 | 위치 |
|---|---|
| 모델 구조 개요 | Fig. 2 (Sec. 3) |
| ImageNet 256×256/512×512 생성 결과 요약 | Fig. 1 |
| Linear probing 정확도 (토크나이저 vs SiT 중간층) | Fig. 3 (Sec. 4.4, Appendix C.3) |
| UMAP 잠재공간 시각화 | Fig. 4 |
| 코드북 압축률별 rFID/gFID 비교(Less-lossy) | Table 2 |
| Alignment 초기화·타깃 모델 비교 | Table 4 |
| TiTok과의 상세 비교 (파라미터, rFID, gFID, IS, LP) | Table 10 (Appendix D.3) |
| 시스템 레벨 비교 (AR·Diffusion 모델 전체) | Table 7 |
| 재구성 시각화 | Fig. 7, Fig. 8 (Appendix D.4) |
| GMMVQ 관련 posterior 유도 | Appendix B.1, B.2 |

---

## 4. 저자 보고 결과 vs. 나의 해석 분리

| 구분 | 내용 |
|---|---|
| **저자 직접 보고 (원문 인용)** | "SoftVQ-VAE improves inference throughput by up to 18x for generating 256×256 images and 55x for 512×512 images while achieving competitive FID scores of 1.78 and 2.21 for SiT-XL." / "SoftVQ-S significantly outperforms both TiTok variants at 64 tokens, achieving better reconstruction quality with an rFID of 1.03 compared to 1.25" |
| **나의 해석** | 이러한 처리량 향상은 대부분 토큰 수 감소(256→32/64)에 따른 self-attention 연산량의 제곱 감소(O(N²)) 효과에서 기인한 것으로 보이며, SoftVQ 자체의 알고리즘적 혁신(soft quantization)보다는 "1D 토큰화 + 극단적 압축"이라는 아키텍처 설계가 속도 향상의 주요 요인일 가능성이 큽니다. FID 개선은 soft quantization의 미분가능성 덕분에 REPA류의 정렬 손실을 자유롭게 적용할 수 있었던 것이 핵심 기여로 판단됩니다. |
| **저자 보고** | "increasing the dimension of the latent space can result in a significant improvement in reconstruction, it leads to deterioration in generation performance" |
| **나의 해석** | 이는 diffusion/flow 기반 생성 모델이 고차원의 비정형(unstructured) 연속 잠재 공간에서 노이즈로부터 데이터로의 매핑을 학습하기 어렵다는, 기존 diffusion 모델 문헌에서 지적된 "curse of dimensionality" 현상과 일치하며, SoftVQ가 근본적으로 이 문제를 해결한 것이 아니라 우회(low-dim 1D 토큰)한 것으로 해석하는 것이 타당합니다. |

---

## 5. 통계적으로 취약한 부분 / 비교 불가능한 수치

1. **모델 크기 불일치 비교**: "With a much smaller model size, i.e., 46M vs 390M, SoftVQ-S significantly outperforms both TiTok variants" — 파라미터 수가 8배 이상 차이 나는 모델을 직접 비교하는 것은 공정한 ablation이라기보다 "효율성 우위"를 강조하기 위한 비교로, 순수 알고리즘 성능 비교로 해석하기엔 통계적으로 취약함.
2. **훈련 스텝 수 불일치**: "Our DiT/SiT results are reported with a total training of 3M iterations (compared to 4M of SiT/XL-2 + REPA and 7M for SiT/XL2 and DiT-XL/2)" — 서로 다른 총 학습 반복 수에서 얻은 FID 수치를 나란히 놓고 비교하는 것은 최종 수렴 성능이 아니라 "특정 시점"의 스냅샷 비교이므로 주의가 필요함.
3. **단일 시드/단일 실행 결과**: FID, IS 등은 일반적으로 확률적 샘플링에 의존하는 지표인데, 논문에서 반복 실행에 따른 표준편차나 신뢰구간이 보고되지 않음(검색된 자료 범위 내에서는 확인되지 않음). 특히 소수 토큰(32개) 조건에서의 rFID 차이(예: 0.61~0.89 수준)는 오차범위 내 차이일 가능성을 배제할 수 없음.
4. **GMMVQ와 SoftVQ의 "유사 성능" 주장**: "GMMVQ achieves reconstruction quality and downstream generation performance similar to SoftVQ in our experiments"라는 서술은 정량적 임계값(예: 몇 % 이내 차이인지) 없이 정성적으로 서술되어 있어 검증 가능성이 낮음.
5. **CFG(classifier-free guidance) 유무에 따른 결과 혼재**: 여러 표에서 CFG 유무 조건이 혼재되어 보고되는데, 서로 다른 CFG 설정(w/ CFG vs w/o CFG)에서 나온 gFID를 교차 비교하면 왜곡된 결론에 이를 수 있음.

---

## 6. 문서가 답하지 않는 질문

- SoftVQ-VAE가 ImageNet 외의 도메인(예: 자연 텍스처가 아닌 문서, 위성 이미지, 의료 영상 등)에서도 잘 작동하는지에 대한 실험적 증거는 확인되지 않음.
- 코드북 크기 $K$의 상한/하한이 어디까지 안전한지에 대한 이론적 분석은 제공되지 않음(경험적 ablation만 존재).
- Softmax 온도 $\tau$의 스케줄링(학습 중 점진적 감소 등)이 최종 성능에 미치는 영향에 대한 상세 분석 부족.
- 텍스트-이미지(conditional) 생성이나 편집(editing) 태스크로의 확장 가능성에 대한 직접적 실험 결과는 검색 자료 범위 내에서 확인되지 않음.
- 실제 서비스 환경에서의 end-to-end 지연시간(latency), 메모리 사용량 등 실무적 배포 지표에 대한 구체적 보고는 제한적임.
- 적대적 강건성(adversarial robustness)이나 분포 외(OOD) 이미지에 대한 재구성 실패 사례 분석은 다뤄지지 않음.

---

## 7. 가장 중요한 5개 Figure 해석

1. **Fig. 1 (ImageNet 256×256/512×512 생성 결과 요약)**: "ImageNet-1K 256×256 and 512×512 generation results of generative models trained on SoftVQ-VAE with 32 and 64 tokens"를 보여주며, 논문의 핵심 성과(초고압축 토큰으로도 고품질 생성)를 한눈에 요약하는 대표 이미지. 독자에게 "이렇게 적은 토큰으로도 이 정도 품질이 가능하다"는 임팩트를 전달하는 역할.

2. **Fig. 2 (모델 아키텍처 개요)**: Transformer 인코더-디코더 구조, 완전 미분 가능한 SoftVQ 양자화, 잠재 공간 표현 정렬(latent space representation alignment)을 시각적으로 보여줌. 이 그림은 논문의 세 가지 핵심 기여(1D 토큰화, soft quantization, alignment)를 하나의 다이어그램으로 통합해 설명하는 구조도.

3. **Fig. 3 (Linear probing 정확도)**: "Linear probing accuracy of ImageNet-1K val. set on (a) latent tokens of tokenizer and (b) intermediate features (layer 20) of SiT"를 보여주며, 단순 픽셀 재구성 품질(rFID)만으로는 드러나지 않는 잠재공간의 "의미론적 풍부함(semantic richness)"을 정량화. 이는 토크나이저의 표현이 다운스트림 분류 작업에도 유용한 특징을 담고 있음을 뒷받침하는 중요한 증거.

4. **Fig. 4 (UMAP 잠재공간 시각화)**: 인코더 출력과 디코더 입력 간의 변동이 최소화된, 구조적이고 판별적인 잠재 표현을 유지함을 보여주며, (d)와 (c)를 비교했을 때 더 큰 인코더가 더 판별적인 잠재 공간을 학습함을 관찰함. 이는 모델 스케일링이 잠재 표현의 질에 직접적으로 기여함을 시각적으로 증명.

5. **Table 2/관련 그래프 (압축률별 rFID 추이, "Less-lossy" 특성)**: "When reducing tokens from 256 to 32, SoftVQ-S maintains a minimal degradation in rFID from 0.80 to 1.24, significantly outperforming both VQ-S, from 1.45 to 10.97, and AE-S, from 1.15 to 2.01." 이 결과(표 형태이나 논문에서 그래프로도 제시됨)는 SoftVQ의 핵심 차별점인 "압축률에 대한 강건성"을 정량적으로 입증하는 가장 설득력 있는 근거.

---

## 8. 결론: 시사점, 후속 연구, 일반화 가능성, 최신 연구 비교

### 저자들이 제시한 시사점
완전 미분 가능한 설계와 의미론적으로 풍부한 잠재 공간을 통해, 생성 품질을 저해하지 않으면서 효율적인 토큰화가 가능함을 보여주었으며, 이는 더 효율적인 생성 모델로 가는 길을 열어준다는 것이 저자들의 핵심 메시지입니다. 저자들은 소프트 코드워드 할당 자체는 새로운 아이디어가 아니지만, 높은 압축률을 달성하는 연속 토크나이저에 이를 적용한 것은 이 연구가 최초라고 명시합니다.

### 8-1. 모델의 일반화 성능 향상 가능성

- **긍정적 신호**: DINOv2-B로 인코더를 초기화하고 정렬했을 때 rFID 0.88, IS 103.4로 우수한 성능을 보였고, CLIP-B와 EVA-02-B에서도 재구성 성능이 추가로 개선됨은, SoftVQ의 잠재공간이 특정 사전학습 표현에 국한되지 않고 다양한 파운데이션 모델과 호환 가능함을 시사하며, 이는 다른 도메인/태스크로의 전이 가능성을 뒷받침함.
- **한계**: 다만 이 모든 실험은 ImageNet이라는 단일 벤치마크에 국한되어 있어, 진정한 "일반화(도메인 전이, 분포 이동에 대한 강건성)"를 검증했다고 보기는 어렵습니다. Linear probing 결과(Fig. 3)는 표현의 판별력을 보여주지만, 이는 어디까지나 ImageNet 클래스 분류라는 좁은 태스크 기준입니다.
- **구조적 일반화 잠재력**: soft quantization이 완전 미분 가능하다는 점은, 향후 다양한 정규화 기법(대조학습, self-distillation, 멀티모달 정렬 등)을 유연하게 결합할 수 있는 "플러그인 가능한(pluggable)" 아키텍처적 이점을 제공하며, 이는 일반화 성능을 개선할 여지가 구조적으로 열려 있음을 의미합니다.

### 8-2. 2020년 이후 관련 최신 연구 비교 분석 및 향후 연구 방향

**직접적 계보(2020년 이후 흐름)**:
- VQ-VAE/VQGAN(Esser et al., 2021) → Taming Transformers, RQ-Transformer 등 discrete 2D 토큰 계열 → **TiTok** (NeurIPS 2024, "An Image is Worth 32 Tokens")이 1D 토큰화의 개념을 제시하며 "using merely 32 tokens, TiTok-L-32 achieves a rFID of 2.21, comparable to the well trained VQGAN from MaskGIT"이라는 성과를 보고. 그러나 TiTok은 2단계 디코더와 사전학습된 토크나이저를 사용하는 반면, SoftVQ-VAE는 단일 디코더로 end-to-end 학습된다는 차이가 있어, SoftVQ는 TiTok의 압축 아이디어를 계승하면서 학습 파이프라인을 단순화·통합한 것으로 평가됩니다.
- **REPA**(Yu et al., 2024)의 표현 정렬 아이디어를 토크나이저 레벨로 옮겨온 것이 SoftVQ의 또 다른 핵심 기여이며, 이는 "Inspired by the recent work of REPA, we propose aligning the representations of latent codes with pre-trained vision encoders"로 명시되어 있습니다.
- **후속 연구들과의 비교**: SoftVQ-VAE 발표 이후 등장한 **MacTok**(2026)은 "It outperforms SoftVQ-VAE by 0.69 gFID using 64 tokens"이라고 보고하며, **Bootstrapped Tokenization**(2026)은 "our approach achieves a state-of-the-art gFID score of 1.56 using only 64 tokens"을 보고하는 등, SoftVQ-VAE가 제시한 "1D 연속 토크나이저 + 극한 압축" 패러다임이 이후 다수의 후속 연구(FlowTok, TA-TiTok, FlexTok, OneDPiece, GigaTok, MacTok 등)로 빠르게 확장되었음을 확인할 수 있습니다. "Subsequent works have explored various directions, including Flowtok and TA-Titok for text-to-image generation, Flextok and OneDpiece for variable-length sequences, and GigaTok for combining 1D and 2D structures."
- 이는 SoftVQ-VAE가 "미분 가능한 soft quantization + 1D 토큰 + 표현 정렬"이라는 조합을 하나의 표준적 설계 패턴으로 정착시키는 데 기여했음을 보여줍니다.

**향후 연구 시 고려할 점(제안)**:
1. **평가지표의 다각화**: rFID/gFID만으로는 잠재공간의 질을 온전히 평가할 수 없으므로("a better rFID does not necessarily translate to a better gFID"), CKNNA, linear probing, 다운스트림 태스크 성능 등을 종합적으로 보고하는 벤치마크 표준화가 필요합니다.
2. **도메인 다양성 검증**: ImageNet 중심 평가를 넘어, 텍스트-이미지 생성, 비디오, 의료/위성 이미지 등 이질적 도메인에서의 강건성 검증이 요구됩니다(실제로 8-11 검색 결과에서 "video autoencoders...fail to efficiently model spatio-temporal redundancies...resulting in suboptimal compression factors"이라는 후속 문제 제기가 이미 나타나고 있음).
3. **공정한 비교 프로토콜**: 모델 크기, 훈련 스텝, CFG 사용 여부가 서로 다른 조건에서의 비교(위 5번 항목 참조)를 통일하는 벤치마크가 커뮤니티 차원에서 필요합니다.
4. **온도 파라미터 및 코드북 크기의 이론적 분석**: softmax 온도 스케줄링과 코드북 크기의 최적 설계 원칙에 대한 이론적 뒷받침이 후속 연구에서 보완될 필요가 있습니다.
5. **효율성-품질 파레토 프론티어의 지속적 갱신**: MacTok, Bootstrapped Tokenization 등 후속 연구들이 이미 SoftVQ-VAE를 능가하는 결과를 보고하고 있으므로, 이 분야는 매우 빠르게 진화하고 있어 향후 연구자는 "현재 SOTA"보다 "설계 원칙(미분가능성, 표현 정렬, 압축률-품질 트레이드오프 관리)"에 주목하는 것이 더 지속가능한 접근일 것입니다.

---

## 참고 자료 (출처)

1. **SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer** — arXiv:2412.10958 (Hao Chen et al., CVPR 2025) — https://arxiv.org/abs/2412.10958 / https://arxiv.org/pdf/2412.10958 / https://arxiv.org/html/2412.10958v2
2. **CVPR 2025 Open Access 버전** — https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_SoftVQ-VAE_Efficient_1-Dimensional_Continuous_Tokenizer_CVPR_2025_paper.pdf
3. **IEEE Xplore 게재본** — https://ieeexplore.ieee.org/iel8/11091818/11091608/11093077.pdf
4. **ResearchGate PDF** — https://www.researchgate.net/publication/387105877_SoftVQ-VAE_Efficient_1-Dimensional_Continuous_Tokenizer
5. **CVPR 2025 Poster 페이지** — https://cvpr.thecvf.com/virtual/2025/poster/32526
6. **Semantic Scholar 논문 페이지** — https://www.semanticscholar.org/paper/SoftVQ-VAE:-Efficient-1-Dimensional-Continuous-Chen-Wang/a77849fd0cbf85818f953e490a8e0385e79ec2ef
7. **Moonlight 문헌 리뷰** — https://www.themoonlight.io/en/review/softvq-vae-efficient-1-dimensional-continuous-tokenizer
8. **Emergent Mind 요약** — https://www.emergentmind.com/topics/softvq-vae
9. **AIModels.fyi 요약** — https://www.aimodels.fyi/papers/arxiv/softvq-vae-efficient-1-dimensional-continuous-tokenizer
10. **An Image is Worth 32 Tokens for Reconstruction and Generation (TiTok)** — NeurIPS 2024 — https://proceedings.neurips.cc/paper_files/paper/2024/file/e91bf7dfba0477554994c6d64833e9d8-Paper-Conference.pdf
11. **MacTok: Robust Continuous Tokenization for Image Generation** — arXiv:2603.29634
12. **Balancing Image Compression and Generation with Bootstrapped Tokenization** — arXiv:2606.05552
13. **Representation Alignment for Generation (REPA)** — https://www.researchgate.net/publication/384770224_Representation_Alignment_for_Generation_Training_Diffusion_Transformers_Is_Easier_Than_You_Think
14. 한국어 논문 리뷰 블로그 — https://kimjy99.github.io/논문리뷰/softvq-vae/
