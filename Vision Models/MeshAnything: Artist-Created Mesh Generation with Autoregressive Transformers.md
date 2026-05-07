# MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers

## 1. 핵심 주장과 주요 기여 (요약)

**MeshAnything** (Chen et al., 2024, arXiv:2406.10163)은 자동 생성된 3D 자산을 인간 아티스트가 만든 메시(Artist-Created Mesh, **AM**)로 변환하는 문제를 **"메시 추출(Mesh Extraction)"이 아닌 "메시 생성(Mesh Generation)" 문제**로 재정의한 최초의 연구입니다.

**핵심 주장:**
- 기존 Marching Cubes, FlexiCubes 등의 mesh extraction 방법은 **재구성 방식(reconstruction manner)** 이기 때문에 본질적으로 효율적인 토폴로지를 만들 수 없음.
- 따라서 mesh extraction을 **shape-conditioned generation 문제**로 재정의하여, 주어진 형상에 정렬된 AM을 생성함.

**주요 기여:**
1. **Shape-Conditioned AM Generation**이라는 새로운 문제 설정 제안
2. **MeshAnything** 모델 아키텍처 (VQ-VAE + Shape-conditioned Decoder-only Transformer) 제시
3. **Noise-Resistant Decoder** 도입 — Transformer가 생성한 불완전한 토큰 시퀀스를 형상 조건을 보조 정보로 활용해 견고하게 디코딩
4. 기존 방식 대비 **수백 배 적은 face 수**로 비슷한 형상 정확도 달성

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

3D 산업(게임, 영화, 메타버스)은 mesh 기반 파이프라인을 표준으로 사용하지만, NeRF, 3D Gaussian Splatting, 생성 모델 등은 다른 3D 표현을 사용합니다. 이를 mesh로 변환할 때 발생하는 **세 가지 주요 문제**:

1. **저장·렌더링·시뮬레이션 비효율성**: AM 대비 수십~수백 배 많은 face
2. **후처리·다운스트림 작업의 복잡성 증가**
3. **날카로운 모서리(sharp edges)와 평면(flat surfaces) 표현의 한계** — over-smoothing, bumpy artifact 발생

### 2.2 수식적 정식화

기존 AM generation 연구는 다음 분포를 직접 추정했습니다:

$$p(\mathcal{M} \mid \mathcal{C})$$

여기서 $\mathcal{C}$는 image, text, 또는 unconditional의 경우 빈 집합. 그러나 이는 형상 분포 $\mathcal{S}$와 토폴로지 분포 $\mathcal{M}$를 동시에 학습해야 합니다.

저자들은 다음과 같이 분해합니다:

$$p(\mathcal{M} \mid \mathcal{C}) \approx p(\mathcal{M}, \mathcal{S} \mid \mathcal{C}) \tag{1}$$

체인 룰에 의해:

$$p(\mathcal{M}, \mathcal{S} \mid \mathcal{C}) = p(\mathcal{M} \mid \mathcal{S}, \mathcal{C}) \cdot p(\mathcal{S} \mid \mathcal{C}) \tag{2}$$

$\mathcal{S}$가 $\mathcal{C}$보다 훨씬 강한 조건이므로:

$$p(\mathcal{M} \mid \mathcal{S}, \mathcal{C}) \approx p(\mathcal{M} \mid \mathcal{S}) \tag{3}$$

따라서 최종 근사:

$$p(\mathcal{M} \mid \mathcal{C}) \approx p(\mathcal{M} \mid \mathcal{S}) \cdot p(\mathcal{S} \mid \mathcal{C}) \tag{4}$$

이때 $p(\mathcal{S} \mid \mathcal{C})$는 NeRF, 3D GS, LRM 등 기존 3D 생성 모델이 이미 잘 추정하므로, MeshAnything은 **$p(\mathcal{M} \mid \mathcal{S})$만 학습**하면 됩니다.

### 2.3 모델 구조

**(a) VQ-VAE 단계 (Mesh Tokenization)**

메시는 face 시퀀스로 이산화됩니다:

$$\mathcal{M} := (f_1, f_2, f_3, \ldots, f_N)$$

Encoder $E$가 face별 feature 추출:

$$\mathcal{Z} = (z_1, z_2, \ldots, z_N) = E(\mathcal{M})$$

Residual VQ로 양자화:

$$\mathcal{T} = \text{RQ}(\mathcal{Z}; \mathcal{B})$$

Decoder $D$가 vertex 좌표 logits를 예측:

$$\hat{\mathcal{M}} = D(\mathcal{Z})$$

코드북 크기 8,192, RVQ depth 3. Encoder/Decoder 모두 BERT 기반 transformer 사용.

**(b) Shape-Conditioned Autoregressive Transformer**

Point cloud encoder $P$ (Michelangelo, Zhao et al. 2024)의 출력을 mesh token 앞에 concat:

$$\mathcal{T}' = \text{concat}(P(\mathcal{S}), \mathcal{T})$$

Backbone은 **OPT-350M**. 추론 시 표준 next-token prediction:

$$\hat{\mathcal{M}} = D(\hat{\mathcal{T}})$$

**(c) Noise-Resistant Decoder (논문의 차별점)**

VQ-VAE decoder는 GT 토큰만 보고 학습되므로, transformer가 생성한 불완전한 시퀀스에 취약합니다. 저자들은 codebook sampling logits에 **Gumbel noise**를 추가하여 imperfect 시퀀스를 시뮬레이션하고, shape condition을 decoder에 주입하여 fine-tuning합니다. 이로써 noise가 커져도 ECD/CD/NC가 크게 떨어지지 않음을 Tab. 3, 4에서 입증.

**(d) 학습 데이터 도메인 갭 해결 전략**

AM에서 직접 point cloud를 샘플링하면 너무 정밀 → 추론 시 도메인 갭 발생. 저자들은:
1. AM에서 SDF 추출 (Wang et al. 2022)
2. Marching Cubes로 **의도적으로 거친 mesh로 변환**
3. 이 거친 mesh에서 point cloud 샘플링

→ 추론 시 NeRF/3D GS 등에서 추출한 거친 point cloud와 도메인을 일치시킴.

### 2.4 성능 향상

**Table 2 (Objaverse, Mesh Generation 비교):**
- COV: PolyGen 23.2 → MeshGPT 41.7 → **MeshAnything 53.1** (↑)
- MMD: 6.22 → 3.83 → **2.72** (↓)
- 1-NNA: 88.2 → 67.3 → **55.7** (↓)
- FID: 48.8 → 25.1 → **14.5** (↓)

**Table 5 (Mesh Extraction 비교):**
MeshAnything은 face 수 **318개**(0.318k)로 Marching Cubes의 146,000개와 유사한 형상 정확도를 달성 — 약 **460배 face 절감**.

**Fig. 3 (Perplexity 비교):**
Shape-conditioned PPL이 unconditional/image-conditioned 대비 현저히 낮음 → 학습 부담 대폭 감소 입증.

### 2.5 한계 (논문 A.3)

1. **최대 face 수 제한** (~800 face) — 큰 장면이나 복잡한 객체 처리 불가
2. **생성 모델의 본질적 불안정성** — Marching Cubes 같은 결정론적 reconstruction 대비 robust하지 않음
3. **Point cloud 품질에 영향** — 노이즈가 매우 클 때(Tab. 6 (c)) 성능 저하

---

## 3. 모델의 일반화 성능 향상 가능성

이 부분은 MeshAnything의 **가장 핵심적인 기여**와 직결됩니다.

### 3.1 학습 부담 감소가 일반화를 가능케 함

기존 MeshGPT는 ShapeNet의 일부 카테고리에 한정. MeshAnything은 **51k Objaverse + 5k ShapeNet**의 **카테고리 무제한** 데이터에서 학습 가능. 이는 $p(\mathcal{M} \mid \mathcal{S})$만 학습하면 되므로 형상 분포를 별도로 학습할 필요가 없기 때문입니다(Fig. 3의 PPL 비교가 이를 정량적으로 증명).

### 3.2 다양한 3D 표현과의 호환성

Point cloud를 shape condition으로 선택한 이유는:
1. **추출 용이성** — NeRF, 3D GS, voxel, mesh 어디서든 샘플링 가능
2. **데이터 증강 가능성** — 연속 표현이라 scaling/rotation augmentation 적용
3. **효율적 인코딩** — 사전학습된 point encoder(Michelangelo) 활용

### 3.3 Fig. 6 (c)의 Disentanglement 증거

저자들은 Fig. 6 (c)에서 GT mesh의 dense version에서 추출한 point cloud를 입력으로, 모델이 **GT와 완전히 다른 토폴로지를 가지면서도 같은 형상**을 가진 mesh를 생성함을 보였습니다. 이는 모델이 단순 overfitting이 아니라 **"효율적 토폴로지 구성 원리"를 학습**했음을 시사합니다.

### 3.4 Robustness 검증 (Table 6)

- Gaussian noise scale 0.005, 0.020에서 성능 거의 유지
- Rodin이 생성한 point cloud에서도 작동 → **End-to-end 3D 생성 파이프라인과 통합 가능성** 입증

### 3.5 일반화의 한계

- **800 face 상한** — 복잡한 객체로 일반화에 본질적 제약
- **Out-of-distribution 형상**(예: 매우 얇은 구조, 매우 큰 장면)에서의 동작 미검증
- Point encoder가 frozen되어 있어 새로운 형상 분포에 대한 적응성 제한 (V2에서는 이를 update하는 방식으로 변경됨)

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 후속 연구 동향 (2024년 후반 ~ 2025년)

MeshAnything은 발표 이후 짧은 기간 내 **autoregressive AM generation 패러다임**의 표준 reference가 되었습니다. 주요 후속 연구:

| 후속 연구 | 핵심 기여 | MeshAnything 대비 개선 |
|---|---|---|
| **MeshAnything V2** (Chen et al., 2024, arXiv:2408.02555) | Adjacent Mesh Tokenization (AMT) | 토큰 길이 절반, face 한도 1,600까지 확장 |
| **EdgeRunner** (Tang et al., 2024, arXiv:2409.18114) | Auto-regressive auto-encoder, edge-based tokenization | 더 긴 시퀀스, manifold 보장 |
| **PivotMesh** (Weng et al., 2024, arXiv:2405.16890) | Pivot vertices guidance | 거친 형상 가이드를 통한 품질 향상 |
| **MeshXL** (Chen et al., 2024, arXiv:2405.20853) | Neural coordinate field, 모델/데이터 스케일업 | foundation model 접근 |
| **BPT** (Weng et al., 2024) | Compressive tokenization | 8k face까지 확장 |
| **Meshtron** (Hao et al., 2024, arXiv:2412.09548) | Hourglass transformer, cross-attention conditioning | 수만 face의 high-fidelity 생성 |
| **TreeMeshGPT** (Lionar et al., 2025, arXiv:2503.11629) | DFS tree sequencing, 2 tokens per face | 시퀀스 길이 22%로 단축, manifold 보장 |
| **LLaMA-Mesh** (2024, arXiv:2411.09595) | LLaMA를 mesh generation에 활용 | 자연어와 통합된 mesh 생성 |
| **QuadGPT** (2025, arXiv:2509.21420) | Native quadrilateral mesh 생성 | Triangle이 아닌 quad mesh 직접 생성 |
| **VertexRegen** (2025, arXiv:2508.09062) | Continuous level of detail | 점진적 해상도 메시 생성 |

이 흐름에서 **MeshAnything의 가장 큰 학문적 기여는 "shape-conditioned AM generation" 패러다임을 제시한 것**입니다. 이후 거의 모든 후속 연구가 point cloud를 condition으로 사용합니다.

### 4.2 향후 연구 시 고려할 점

**(1) 토크나이저 효율성이 곧 스케일업 한계**
MeshAnything의 800 face 한계는 토크나이저의 비효율성 때문입니다. AMT(V2), EdgeRunner, BPT, TreeMeshGPT 등이 모두 이 지점을 공략한 것은 시사적입니다. 새 연구는 **압축률, 시퀀스 규칙성, manifold 보장**의 trade-off를 설계 첫 단계에서 고려해야 합니다.

**(2) Point cloud 외 더 강력한 shape condition 탐색**
Point cloud는 추출은 쉽지만 위상 정보가 없습니다. SDF, latent shape embedding(예: CLAY), multi-view feature 등 더 풍부한 condition으로의 확장이 유망합니다.

**(3) Manifold/non-manifold topology 보장**
MeshAnything은 생성된 메시의 manifold 성질을 보장하지 않습니다. 산업 적용을 위해서는 **rigging, UV unwrapping, subdivision** 등 다운스트림 호환성 검증이 필요합니다. QuadGPT가 quad mesh로 간 것은 이 맥락입니다.

**(4) 평가 지표의 한계**
저자들도 인정하듯, Chamfer Distance/Normal Consistency는 **형상 정합도만 측정**할 뿐 토폴로지의 "예술적/실용적 우수성"을 직접 평가하지 못합니다. User study에 의존하는 현 상황을 넘어, **edge flow, valence distribution, deformation stability** 등 토폴로지 전용 지표 연구가 필요합니다.

**(5) 생성 모델로서의 불안정성**
A.3에서 인정된 실패 사례(failure cases)는 autoregressive 생성의 본질적 문제입니다. **rejection sampling, RL fine-tuning(QuadGPT의 tDPO), classifier-free guidance** 등 LLM 분야 기법의 mesh generation 적용이 유망한 방향입니다.

**(6) 사회적 영향 (A.4)**
저자들이 명시한 대로, 비용 절감은 게임/영화/메타버스 산업을 가속화하는 한편 **저작권·위조·악용** 가능성도 동반합니다. 향후 연구는 출처 추적, 워터마킹 같은 책임 있는 배포 메커니즘을 함께 고려해야 합니다.

---

## 참고자료

논문 본문에서 인용한 자료:
- 본 분석의 근거 논문: Chen, Y. et al. (2024). *MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers.* arXiv:2406.10163v2 (사용자 업로드 PDF)
- 프로젝트 페이지: https://buaacyw.github.io/mesh-anything/

웹 검색으로 확인한 후속/관련 연구:
- *MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization* — arXiv:2408.02555 (https://buaacyw.github.io/meshanything-v2/)
- *EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation* — arXiv:2409.18114 (https://arxiv.org/abs/2409.18114)
- *MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers* (Siddiqui et al., CVPR 2024)
- *MeshXL: Neural Coordinate Field for Generative 3D Foundation Models* — arXiv:2405.20853
- *PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance* — arXiv:2405.16890
- *Scaling Mesh Generation via Compressive Tokenization (BPT)* — arXiv:2411.07025
- *Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale* — arXiv:2412.09548
- *TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing* — arXiv:2503.11629
- *LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models* — arXiv:2411.09595
- *QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models* — arXiv:2509.21420
- *VertexRegen: Mesh Generation with Continuous Level of Detail* — arXiv:2508.09062
- 출처 검색 페이지: emergentmind.com/topics/meshgpt, Hugging Face Papers

---

**참고로 답변의 정확도에 관한 사항**: 본 답변에서 MeshAnything 원논문의 내용(수식, 표 수치, 아키텍처 세부)은 업로드된 PDF에서 직접 확인한 것입니다. 후속 연구에 대한 비교는 웹 검색 결과를 기반으로 했으며, 각 후속 연구의 세부 수치는 해당 논문 원문 확인이 필요한 경우가 있습니다. 특히 face 한도, 학습 시간 등 일부 정량 비교는 후속 연구의 주장에 의존하므로, 정밀한 벤치마크가 필요하다면 각 원논문의 실험 섹션을 직접 참조하시기를 권장드립니다.
