# MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers

## 1. 핵심 주장 및 주요 기여 (요약)

**MeshGPT** (Siddiqui et al., CVPR 2024)는 3D 삼각형 메쉬를 **언어 모델 방식의 자기회귀(autoregressive) 시퀀스 생성** 문제로 재정의한 연구입니다. 핵심 주장은 다음과 같습니다.

- 신경장(neural field) 기반 방법(예: GET3D)에서 Marching Cubes로 추출된 **조밀한(dense)·과삼각화된 메쉬**와 달리, 사람이 만든 메쉬처럼 **간결(compact)하고 날카로운 모서리**를 갖는 메쉬를 직접 생성한다.
- **그래프 컨볼루션 인코더 + 잔차 벡터 양자화(Residual Vector Quantization, RQ) + GPT-2 디코더**를 결합한 2단계 학습 파이프라인을 제안한다.
- ShapeNet의 Chair·Table·Bench·Lamp 카테고리에서 기존 SOTA 대비 **Coverage 평균 +9%, FID 약 30점 개선**을 보고한다.

주요 기여는 (i) 메쉬를 삼각형 시퀀스로 보는 GPT 스타일 생성 패러다임 확립, (ii) 정점 단위(per-vertex) 잔차 양자화를 통한 시퀀스 압축 및 학습 친화적 토큰화, 두 가지로 요약됩니다.

---

## 2. 문제 정의·제안 방법·모델 구조·성능·한계

### 2.1 해결하고자 하는 문제

기존 3D 생성 모델은 voxel·point cloud·neural field 등 **간접 표현**을 학습한 뒤 후처리(iso-surfacing)로 메쉬를 추출하기 때문에:
1. 표면이 과도하게 조밀하고(dense), 미끈하지 못한 bumpy 아티팩트가 발생한다.
2. 데시메이션(decimation)으로 단순화하면 미세 구조가 소실된다.
3. 아티스트가 만든 메쉬의 **불규칙·효율적 삼각화 패턴**을 재현하지 못한다.

또한 PolyGen(2020)처럼 정점·면을 **별개 네트워크**로 자기회귀 생성하면, 면 생성기가 학습 시 GT 정점만 보고 추론 시 생성 정점에는 노출되지 않아 **누적 오차**가 발생합니다.

### 2.2 제안 방법 — 메쉬의 시퀀스화

메쉬 $\mathcal{M}$을 면(삼각형)의 시퀀스로 정의합니다.

$$\mathcal{M} := (f_1, f_2, f_3, \ldots, f_N), \quad f_i \in \mathbb{R}^{n_{\text{in}}}$$

순서는 PolyGen 규약을 따라 **z-y-x 정렬된 정점의 최저 인덱스 기준**으로 면을 정렬하고, 각 면 내에서는 최저 정점을 첫 번째로 두는 사이클릭 순열을 사용합니다.

**(1) 단계 1 — 기하 코드북 학습 (Vocabulary Learning)**

좌표를 그대로 토큰화하면 시퀀스 길이가 $9N$으로 폭증하고 이웃 정보가 사라집니다. 이를 해결하기 위해 그래프 컨볼루션 인코더 $E$로 면 단위 임베딩을 추출합니다.

$$\mathbf{Z} = (z_1, z_2, \ldots, z_N) = E(\mathcal{M})$$

각 면은 그래프의 노드, 인접 면은 무방향 간선으로 연결되며, 노드 입력 특징은 **위치 인코딩된 9 좌표 + 면 법선 + 모서리 사이 각 + 면적**입니다. 인코더는 SAGEConv [Hamilton et al., 2017] 층의 스택입니다.

추출된 특징은 **잔차 벡터 양자화(RQ)**로 양자화됩니다. 깊이 $D$의 RQ는 잔차를 재귀적으로 양자화합니다.

$$t^d = Q(r^{d-1}; \mathcal{C}), \qquad r^d = r^{d-1} - e(t^d)$$

$$\hat{z}^{(d)} = \sum_{d'=1}^{d} e(t^{d'}), \qquad \mathcal{L}_{\text{commit}}(z, \hat{z}) = \sum_{d=1}^{D} \|z - \text{sg}[\hat{z}^{(d)}]\|_2^2$$

핵심 트릭은 **per-vertex 양자화**입니다. 576차원 면 특징 $z_i$를 3개의 192차원 정점 특징 $(z_i^1, z_i^2, z_i^3)$으로 분할하고, 공유 정점에 대해 평균(aggregation)을 취한 뒤 각 정점에 $D/3$개의 토큰을 할당합니다.

$$\text{RQ}(z_i; \mathcal{C}, D) = \big(\text{RQ}(z_i^1; \mathcal{C}, D/3), \ldots, \text{RQ}(z_i^3; \mathcal{C}, D/3)\big)$$

이렇게 하면 인접한 두 면이 같은 정점을 공유할 때 시퀀스에 **반복 토큰**이 등장하여 트랜스포머가 학습하기 쉬워집니다 (논문 Fig. 16). 디코더는 1D ResNet-34로 다음 합성 특징을 입력받아 9개 좌표를 $128^3$ 이산 분포로 예측합니다.

$$\hat{\mathbf{Z}} = (\hat{z}_1, \ldots, \hat{z}_N), \quad \hat{z}_i = \bigoplus_{v=0}^{2} \sum_{d=1}^{D/3} e(t_i^{3v+d})$$

좌표를 **연속값 회귀가 아닌 이산 분포 분류**로 예측하는 점이 floating-face 아티팩트를 줄이는 데 결정적이라고 보고되었습니다.

**(2) 단계 2 — GPT-2 트랜스포머로 다음 토큰 예측**

학습된 $\mathbf{T} = (t_1, \ldots, t_N)$ (총 길이 $|\mathbf{T}| = DN$)을 GPT-2 medium에 입력하고, 다음 코드북 인덱스의 로그 확률을 최대화합니다.

$$\prod_{i=1}^{N} \prod_{d=1}^{D} p\big(t_i^d \mid e(t_{ < i}^d), e(t_i^{ < d}); \theta\big)$$

학습 손실은 표준 cross-entropy:

$$\mathcal{L}_{\text{recon}} = \sum_{i=1}^{N} \sum_{j=1}^{D} \sum_{k=1}^{|\mathcal{C}|} \log p(s_i^k = t_i^j)$$

추론 시에는 SOS 토큰부터 빔 샘플링으로 EOS까지 자동 회귀 생성하고, 디코더로 'triangle soup'을 만든 뒤 MeshLab으로 인접 정점 병합 후처리만 수행합니다.

### 2.3 모델 구조 요약

| 구성요소 | 사양 |
|---|---|
| 인코더 | SAGEConv 스택, 면 그래프 입력, 출력 차원 576 |
| 양자화 | RQ depth $D=2$ (정점당 2 토큰, 면당 6 토큰), 코드북 크기 16384, 임베딩 차원 192 |
| 디코더 | 1D ResNet-34, 9 좌표× $128$ 클래스 출력 |
| 트랜스포머 | GPT-2 medium (24 layer, 16 head, 768 width), context length 4608 |
| 학습 자원 | 인코더-디코더 2 A100×2일, 트랜스포머 4 A100×5일 |

### 2.4 성능 향상

ShapeNetV2 4개 카테고리에서 (Chair, Table, Bench, Lamp), Polygen·BSPNet·AtlasNet·GET3D 대비 **모든 지표에서 우위**입니다. 예시(Chair):

- COV ↑: 43.28 (vs. GET3D 40.85, Polygen 31.22)
- MMD ↓: 3.29 (× $10^3$ )
- 1-NNA: 75.51 (50%에 가까울수록 좋음)
- FID ↓: 18.46 (vs. GET3D 81.45, Polygen 61.10)
- 평균 면 수: 228 (vs. GET3D 27,457)

49명 사용자 연구에서 형상 품질 68%, 삼각화 품질 73%로 GET3D 대비 선호되었고, AtlasNet·BSPNet·Polygen 대비 80%대 압도적 선호를 받았습니다.

### 2.5 한계 (논문이 명시한 항목)

1. **자기회귀 샘플링 속도** — 메쉬 1개 생성에 30~90초.
2. **컨텍스트 윈도우** — 4608 토큰 한계로 800 면 이상 메쉬는 학습 제외. **씬(scene) 스케일에는 부적합**.
3. **모델 크기** — GPT-2 medium에 머무름. Llama-2급 확장이 막혀 있음.
4. 이산 좌표 해상도 $128^3$로 묶여 있어 미세 디테일에 한계.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

논문 자체가 일반화에 대해 중요한 단서를 제공합니다.

**(1) 대규모 사전학습이 카테고리 일반화에 결정적**
Tab. 3의 'w/o Pretraining' ablation에서 ShapeNet 단일 카테고리만으로 학습 시 COV 36.97→43.28(전체 카테고리 사전학습 후 fine-tuning)로 큰 격차가 나타났습니다. 이는 **언어 모델과 동일하게 데이터 다양성·규모가 직접적으로 일반화 성능을 끌어올린다**는 강한 신호입니다. 저자들도 limitations에서 "larger language models benefit from increased data and computational power, expanding these resources could significantly boost MeshGPT's performance"라고 명시합니다.

**(2) 새로운 형상 생성(Novelty) 능력**
Chamfer Distance 기반 nearest-neighbor 분석(Fig. 8)에서 50번째 백분위수 생성물조차 학습셋과 시각적으로 명확히 구분되는 형상을 만들었습니다. 즉 **단순 검색이 아니라 진짜 분포 학습이 일어났음**을 보여주며, 이는 더 큰 데이터로 확장 시 일반화 잠재력이 충분함을 시사합니다.

**(3) Shape Completion 능력**
Fig. 9에서 부분 메쉬를 입력으로 다중 가능 완성을 자기회귀로 추론하는데, 이는 GPT 계열에서 prefix-conditioning이 자연스럽게 작동하는 것과 같은 원리입니다. 추가 학습 없이 **부분→전체 일반화**가 작동한다는 점은 향후 텍스트·이미지·포인트클라우드 컨디셔닝으로의 확장이 용이함을 의미합니다.

**(4) per-vertex 양자화의 일반화 친화성**
Fig. 16과 Tab. 3가 보여주듯, per-face 양자화는 재구성 정확도(98.64%)는 더 높지만 트랜스포머 학습이 어려워 최종 생성 품질(COV 23.57)은 크게 떨어집니다. 인접 면 간 **공유 정점 토큰 반복 패턴**이 시퀀스 모델링의 inductive bias를 제공한다는 통찰은 이후 후속 연구(MeshAnything V2의 Adjacent Mesh Tokenization 등)에 직접 영감을 주었습니다.

**(5) 일반화의 본질적 한계 요인**
- 컨텍스트 4608 토큰 한계 → 800면 초과 메쉬·복잡한 토폴로지에 대한 일반화 차단.
- ShapeNet은 가구·실내 객체 중심이라 인체·캐릭터·기계부품 같이 다른 위상(topology) 분포에는 검증되지 않음.
- 카테고리별 fine-tuning이 여전히 필요 → 진정한 zero-shot 일반화는 미달성.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 후속 연구에 미친 영향 (2024 이후 비교 분석)

MeshGPT는 **"Native Mesh Generation"이라는 새로운 연구 라인**을 열었습니다. 이후 출판된 주요 후속 연구를 비교하면 다음과 같습니다.

| 연구 | 출판 | MeshGPT 대비 핵심 개선 |
|---|---|---|
| **PivotMesh** (Weng et al., arXiv:2405.16890, 2024) | 2024.05 | "pivot vertices"를 먼저 생성한 뒤 전체 메쉬를 채우는 2단계 자기회귀로 토폴로지 모델링 난이도를 낮춤. ShapeNet뿐 아니라 **Objaverse, Objaverse-XL** 같은 대규모 데이터로 확장. |
| **MeshAnything** (Chen et al., arXiv:2406.10163, ICLR 2025) | 2024.06 | 형상(point cloud)을 **조건**으로 받아 토폴로지 분포만 학습. MeshGPT 아키텍처를 그대로 이어받되 분포 학습 부담을 줄임. |
| **MeshAnything V2** (arXiv:2408.02555, 2024) | 2024.08 | **Adjacent Mesh Tokenization (AMT)** 도입으로 토큰 길이 약 50% 압축, 면 수 한도 두 배. |
| **MeshXL** (Chen et al., arXiv:2405.20853, 2024) | 2024.05 | **VQ-VAE를 폐기**하고 좌표 레벨에서 직접 자기회귀 — "Neural Coordinate Field". MeshGPT의 코드북 의존성을 회피. |
| **EdgeRunner** (Tang et al., arXiv:2409.18114, 2024) | 2024.09 | Edge 기반 토큰화 + auto-regressive auto-encoder로 시퀀스 압축률 추가 향상. |
| **FreeMesh** (arXiv:2505.13573, 2025) | 2025.05 | Coordinates Merging으로 9000 토큰 컨텍스트까지 확장, 4000면 메쉬 생성. |
| **MeshArt** (arXiv:2412.11596, 2024) | 2024.12 | 구조 가이드 트랜스포머로 **관절(articulated) 메쉬** 생성으로 도메인 확장. |
| **MeshMosaic** | 2025 | Local-to-Global Assembly로 고삼각수 메쉬 스케일링 한계 해결 시도. |

흐름을 종합하면, MeshGPT 이후 연구들은 다음 세 축에서 **MeshGPT의 한계를 직접 공략**해 왔습니다.
1. **시퀀스 압축**(MeshAnything V2, EdgeRunner, FreeMesh) — context window 문제 해소.
2. **조건부 생성**(MeshAnything 계열, PivotMesh) — 형상 분포와 토폴로지 분포를 분리.
3. **데이터 스케일업**(PivotMesh, MeshXL) — Objaverse-XL 같은 대규모 데이터로 확장.

### 4.2 향후 연구 시 고려할 점

1. **토큰화의 학습 친화성과 압축률 사이의 trade-off**
   per-vertex 양자화처럼 inductive bias를 제공하는 토큰 설계가 단순한 압축률보다 중요합니다. AMT, edge tokenization, coordinate merging 모두 이 통찰의 변주입니다.

2. **분포 분리 (decoupling shape vs. topology)**
   MeshGPT가 두 분포를 동시에 배우려 한 것은 학습 비효율의 큰 원인이었습니다. 후속 연구는 형상은 조건으로 받고 토폴로지만 학습하는 방향으로 수렴하고 있어, 새 연구는 **어떤 조건 신호(텍스트·이미지·포인트클라우드·SDF)가 가장 효율적인가**를 비교 분석해야 합니다.

3. **고면수·씬 스케일 메쉬**
   800면 한계는 산업 적용에 결정적 장벽입니다. local-to-global 어셈블리, 위계적(hierarchical) 표현, 압축 효율적 토큰화 중 어느 조합이 확장성에 유리한지 후속 검증이 필요합니다.

4. **평가 지표의 한계**
   COV·MMD·1-NNA·FID는 모두 점군이나 렌더링에서 계산되어 **실제 토폴로지 품질**을 직접 측정하지 못합니다. Edge length 분포, dihedral angle 통계, manifoldness, watertightness 등 **메쉬-네이티브 메트릭**의 표준화가 필요합니다.

5. **샘플링 속도와 디코딩 효율**
   30~90초의 샘플링 시간은 실용적 배포의 장벽입니다. speculative decoding, parallel sampling, diffusion-AR 하이브리드 같은 기법이 향후 연구의 자연스러운 후보입니다.

6. **편향(bias)·도메인 갭**
   ShapeNet 위주로 검증된 모델은 인체·캐릭터·자연물·CAD 부품 등 다른 도메인에 일반화되지 않을 수 있습니다. Objaverse-XL 규모 학습이 이미 후속 연구에서 시도되고 있으나, 도메인별 토폴로지 사전(prior)의 차이는 여전히 미해결입니다.

7. **재현성·라이선스**
   AUDI AG·TUM이 공개한 GitHub(https://github.com/audi/MeshGPT)는 비상업 연구용 라이선스로 배포되어 있어, 산업 응용 시 라이선스를 면밀히 확인해야 합니다.

---

## 참고 자료

1. Siddiqui et al., **"MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers"**, CVPR 2024 / arXiv:2311.15475 (사용자 업로드 PDF, 본문 핵심 분석의 1차 출처)
2. 프로젝트 페이지: https://nihalsid.github.io/mesh-gpt/
3. CVPR 2024 Poster: https://cvpr.thecvf.com/virtual/2024/poster/30751
4. 공식 GitHub (AUDI): https://github.com/audi/MeshGPT
5. Chen et al., **"MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers"**, arXiv:2406.10163, 2024 (https://buaacyw.github.io/mesh-anything/, https://arxiv.org/html/2406.10163v2)
6. Chen et al., **"MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization"**, arXiv:2408.02555, 2024 (https://arxiv.org/html/2408.02555v1)
7. Weng et al., **"PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance"**, arXiv:2405.16890, 2024 (https://arxiv.org/abs/2405.16890)
8. Chen et al., **"MeshXL: Neural Coordinate Field for Generative 3D Foundation Models"**, arXiv:2405.20853, 2024
9. Tang et al., **"EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation"**, arXiv:2409.18114, 2024 (https://arxiv.org/html/2409.18114v1)
10. **"FreeMesh: Boosting Mesh Generation with Coordinates Merging"**, arXiv:2505.13573, 2025 (https://arxiv.org/html/2505.13573)
11. **"MeshArt: Generating Articulated Meshes with Structure-Guided Transformers"**, arXiv:2412.11596, 2024
12. Nash et al., **"PolyGen: An Autoregressive Generative Model of 3D Meshes"**, ICML 2020 (논문 내 baseline)
13. Gao et al., **"GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images"**, NeurIPS 2022 (논문 내 baseline)
14. Hamilton et al., **"Inductive Representation Learning on Large Graphs (GraphSAGE)"**, NeurIPS 2017 (인코더 SAGEConv 출처)

> 주: 2.4의 정량 수치, 모델 하이퍼파라미터, ablation 결과는 모두 업로드된 논문 본문 및 부록(특히 Tab. 1·3·4, Sec. 3.3, Sec. B)에서 직접 인용한 것입니다. 후속 연구 비교(섹션 4.1)는 위 6~11번 출처에서 확인했으며, 각 후속 연구의 정확한 정량 비교 수치(예: 면 수 한계, 카테고리별 FID)는 본문에서 일관되게 제공되지 않아 정성적 차별점만 기술했습니다. 더 정밀한 수치 비교가 필요하시면 각 후속 논문의 실험 섹션을 직접 확인하시길 권장합니다.
