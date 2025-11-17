# Linear Regression

선형 회귀(Linear Regression)는 머신러닝 및 통계학에서 가장 기본적이면서도 중요한 지도 학습 알고리즘 중 하나입니다. 이 개념은 종속 변수와 하나 이상의 독립 변수 간의 선형 관계를 모델링하는 통계적 방법으로, 데이터 과학 분야에서 예측 모델링과 관계 분석의 핵심 도구로 활용되고 있습니다.

## 선형 회귀의 핵심 주장

### 기본 개념과 원리

선형 회귀의 핵심은 **입력 변수(독립 변수)와 출력 변수(종속 변수) 간의 선형 관계를 모델링**하는 것입니다[1][2]. 이 알고리즘은 주어진 데이터에 가장 적합한 직선(또는 고차원에서는 평면)을 찾아 새로운 입력 값에 대한 출력을 예측합니다[3][4].

선형 회귀는 지도 학습 알고리즘으로, 레이블된 훈련 데이터를 사용하여 입력과 출력 간의 관계를 학습합니다[5]. 이 과정에서 프로그램은 답을 알려주는 방식으로 학습하기 때문에 지도 학습 범주에 속합니다[6].

### 수학적 표현

단순 선형 회귀의 수학적 표현은 다음과 같습니다[1][7]:

$$ y = \theta_0 + \theta_1 x $$

여기서:
- $$y $$: 종속 변수(예측하고자 하는 값)
- $$x $$: 독립 변수(입력 특성)
- $$\theta_0 $$: 절편(bias)
- $$\theta_1 $$: 기울기(가중치, weight)

다중 선형 회귀의 경우 다음과 같이 확장됩니다[1][8]:

$$ y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n $$

## 해결하고자 하는 문제

### 회귀 문제의 본질

선형 회귀는 **연속형 값을 예측하는 회귀 문제**를 해결합니다[9][10]. 이는 분류 문제와 달리 무한한 범위의 연속적인 값을 예측하는 것이 특징입니다[11].

구체적으로 다음과 같은 문제들을 해결할 수 있습니다:

| 문제 유형 | 예시 |
|-----------|------|
| 예측 문제 | 집값 예측, 주식 가격 예측[12][13] |
| 관계 분석 | 광고비와 매출 간의 관계 분석[14] |
| 추세 파악 | 시간에 따른 매출 변화 추세[15] |

### 실제 적용 분야

선형 회귀는 다양한 분야에서 활용됩니다[14]:

- **시장 분석**: 가격과 판매량 간의 관계 분석
- **금융 분석**: 투자 수익률 예측, 위험 평가
- **스포츠 분석**: 선수 성능 예측, 경기 결과 분석
- **환경 보건**: 환경 요인과 건강 지표 간의 관계 분석
- **의료 분야**: 질병 위험 인자 분석, 환자 결과 예측

## 제안하는 방법: 최소 제곱법

### 비용 함수 (Cost Function)

선형 회귀의 핵심 방법론은 **최소 제곱법(Least Squares Method)**입니다[16][17]. 이 방법은 실제 값과 예측 값 간의 오차를 최소화하는 최적의 매개변수를 찾습니다.

평균 제곱 오차(MSE)를 비용 함수로 사용합니다[6][18]:

$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$

여기서:
- $$m $$ : 훈련 데이터의 개수
- $$h_\theta(x^{(i)}) $$ : i번째 데이터의 예측값
- $$y^{(i)} $$ : i번째 데이터의 실제값

### 매개변수 추정

최소 제곱법을 통해 다음과 같이 매개변수를 추정할 수 있습니다[19][7]:

**기울기 (slope):**

$$ b = \frac{n\sum xy - (\sum x)(\sum y)}{n\sum x^2 - (\sum x)^2} $$

**절편 (intercept):**

$$ a = \frac{\sum y - b\sum x}{n} $$

## 모델 구조

### 선형 모델의 특징

선형 회귀 모델은 다음과 같은 구조적 특징을 가집니다[20][21]:

1. **선형성**: 매개변수에 대해 선형적인 관계를 가짐
2. **단순성**: 해석이 용이하고 구현이 간단
3. **효율성**: 계산 복잡도가 낮음
4. **확장성**: 다중 변수로 쉽게 확장 가능

### 가정 사항

선형 회귀가 제대로 작동하기 위해서는 다음 가정들이 만족되어야 합니다[22][21][23]:

| 가정 | 설명 |
|------|------|
| **선형성** | 독립 변수와 종속 변수 간의 선형 관계[22] |
| **독립성** | 오차항들이 서로 독립적임[22] |
| **등분산성** | 오차의 분산이 일정함[22] |
| **정규성** | 오차가 정규분포를 따름[22] |
| **다중공선성 부재** | 독립 변수들 간의 강한 상관관계 없음[21] |

## 성능 향상 방법

### 정규화 기법

모델의 일반화 성능을 향상시키기 위해 다음과 같은 정규화 기법을 사용할 수 있습니다[24]:

- **Ridge 회귀 (L2 정규화)**: 계수의 크기를 제한하여 과적합 방지
- **Lasso 회귀 (L1 정규화)**: 일부 계수를 0으로 만들어 특성 선택 효과

### 특성 선택 및 전처리

모델 성능 향상을 위한 방법들[24]:

1. **특성 선택**: 중요한 변수만 선별하여 모델 복잡도 감소
2. **특성 스케일링**: 변수들의 크기를 정규화하여 수렴 속도 향상
3. **특성 공학**: 기존 변수를 조합하여 새로운 특성 생성

## 한계점

### 모델의 제약사항

선형 회귀는 다음과 같은 한계점을 가집니다[22][21][25]:

| 한계점 | 설명 |
|--------|------|
| **선형성 가정** | 비선형 관계 모델링 어려움[22] |
| **이상치 민감성** | 이상치에 크게 영향받음[10] |
| **다중공선성** | 독립 변수 간 상관관계 시 문제 발생[21] |
| **과소적합 위험** | 복잡한 관계 포착 어려움[21] |

### 실제 적용 시 고려사항

1. **데이터 전처리**: 결측값 처리, 이상치 제거 필요
2. **모델 검증**: 교차 검증을 통한 성능 평가
3. **가정 검증**: 모델 가정 위반 여부 확인

## 일반화 성능 향상 방안

### 모델 앙상블

단일 선형 회귀 모델의 한계를 극복하기 위해 **모델 앙상블 기법**을 활용할 수 있습니다[24]:

- **배깅(Bagging)**: 여러 모델의 예측을 평균하여 분산 감소
- **부스팅(Boosting)**: 순차적으로 모델을 학습하여 편향 감소

### 교차 검증

모델의 일반화 성능을 정확히 평가하기 위해 **교차 검증(Cross-Validation)**을 사용합니다[26]:

- **k-폴드 교차 검증**: 데이터를 k개 부분으로 나누어 검증
- **홀드아웃 검증**: 훈련/검증/테스트 데이터 분리

### 정규화와 일반화

정규화 기법은 모델의 일반화 성능 향상에 핵심적인 역할을 합니다[27]:

- **편향-분산 트레이드오프**: 정규화를 통해 분산 감소
- **적응적 정규화**: 데이터 특성에 따른 동적 정규화

## 앞으로의 연구에 미치는 영향

### 현재 연구 동향

선형 회귀는 현재 다음과 같은 방향으로 발전하고 있습니다[28][29][24]:

1. **베이지안 선형 회귀**: 사전 지식 활용한 모델 개선[29]
2. **양자 기반 알고리즘**: 양자 컴퓨팅을 활용한 선형 회귀 가속화[30]
3. **하이브리드 모델**: 다른 머신러닝 기법과의 결합[31]

### 미래 연구 방향

앞으로 고려해야 할 연구 방향들[24][32]:

#### 알고리즘 개선
- **적응적 학습률**: 데이터 특성에 따른 동적 학습률 조정[33]
- **강화 학습 통합**: 강화 학습과 회귀 분석의 결합
- **딥러닝 융합**: 신경망과 선형 회귀의 하이브리드 모델

#### 빅데이터 처리
- **분산 처리**: 대용량 데이터에 대한 효율적 처리 방법
- **스트리밍 데이터**: 실시간 데이터 처리 능력 향상
- **고차원 데이터**: 차원의 저주 해결 방안

#### 응용 분야 확장
- **IoT 데이터 분석**: 센서 데이터 기반 예측 모델
- **실시간 의사결정**: 빠른 예측이 필요한 시스템
- **개인화 서비스**: 사용자 맞춤형 예측 모델

## 앞으로 연구 시 고려할 점

### 기술적 고려사항

1. **계산 효율성**: 대용량 데이터에 대한 효율적 처리 방법 개발
2. **모델 해석성**: 복잡한 모델에서도 해석 가능한 결과 제공
3. **강건성**: 노이즈와 이상치에 대한 내성 향상

### 실무적 고려사항

1. **도메인 지식 통합**: 전문 분야 지식을 모델에 효과적으로 반영
2. **자동화**: 모델 선택과 하이퍼파라미터 튜닝의 자동화
3. **배포 및 운영**: 실제 운영 환경에서의 모델 관리 방안

### 윤리적 고려사항

1. **편향성 제거**: 데이터와 모델의 편향성 최소화
2. **개인정보 보호**: 프라이버시 보호 기술 적용
3. **투명성**: 모델 결정 과정의 투명성 확보

## 결론

선형 회귀는 머신러닝의 기초가 되는 핵심 알고리즘으로, 단순하면서도 강력한 예측 능력을 제공합니다. 비록 선형성 가정이라는 제약이 있지만, 적절한 전처리와 정규화 기법을 통해 실무에서 여전히 광범위하게 활용되고 있습니다.

향후 연구에서는 빅데이터 처리, 실시간 분석, 그리고 다른 머신러닝 기법과의 융합을 통해 선형 회귀의 활용도를 더욱 높일 수 있을 것으로 예상됩니다. 특히 해석 가능한 AI의 중요성이 증가하는 현 시점에서, 선형 회귀의 단순성과 해석 용이성은 더욱 중요한 가치를 가질 것입니다.

연구자들은 선형 회귀를 단순한 기초 알고리즘으로만 여기지 않고, 현대적 문제 해결을 위한 강력한 도구로 발전시켜 나가야 할 것입니다.

[1] https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/
[2] https://velog.io/@minch121/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%A7%80%EB%8F%84%ED%95%99%EC%8A%B5%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80
[3] https://www.scribbr.com/statistics/simple-linear-regression/
[4] https://www.w3schools.com/python/python_ml_linear_regression.asp
[5] https://ieeexplore.ieee.org/document/10434301/
[6] https://justweon-dev.tistory.com/12
[7] https://byjus.com/linear-regression-formula/
[8] https://mozenworld.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-%EC%86%8C%EA%B0%9C-1-%EC%84%A0%ED%98%95-%ED%9A%8C%EA%B7%80-%EB%AA%A8%EB%8D%B8-Linear-Regression
[9] https://ai-creator.tistory.com/585
[10] https://towardsdatascience.com/supervised-learning-basics-of-linear-regression-1cbab48d0eba/
[11] https://dev.to/mohiyaddeen7/linear-regression-from-theory-to-practice-4lli
[12] https://ieeexplore.ieee.org/document/10455550/
[13] https://www.ijitee.org/portfolio-item/D1110029420/
[14] https://sg.indeed.com/career-advice/career-development/linear-regression
[15] https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-023-02282-z
[16] http://demonstrations.wolfram.com/LeastSquares/
[17] https://www.accountingverse.com/managerial-accounting/cost-behavior/least-squares-method.html
[18] https://www.geeksforgeeks.org/machine-learning/what-is-the-cost-function-in-linear-regression/
[19] https://www.geeksforgeeks.org/maths/linear-regression-formula/
[20] https://developers.google.com/machine-learning/crash-course/linear-regression
[21] https://www.interactivebrokers.com/campus/ibkr-quant-news/linear-regression-assumptions-and-limitations-part-i/
[22] https://www.techladder.in/article/limitations-linear-regression
[23] https://blog.quantinsti.com/linear-regression-assumptions-limitations/
[24] https://www.matec-conferences.org/articles/matecconf/pdf/2024/07/matecconf_icpcm2023_01046.pdf
[25] https://www.rittmanmead.com/blog/2023/03/breaking-the-assumptions-of-linear-regression/
[26] https://d2l.ai/chapter_linear-regression/generalization.html
[27] https://arxiv.org/abs/2005.00695
[28] https://www.mdpi.com/2071-1050/17/7/3063
[29] https://drpress.org/ojs/index.php/ajst/article/view/23570
[30] https://quantum-journal.org/papers/q-2022-06-30-754/
[31] https://www.scitepress.org/Papers/2024/128717/128717.pdf
[32] https://fastercapital.com/topics/conclusion-and-future-directions-in-regression-analysis-with-r.html
[33] https://www.techrxiv.org/users/765374/articles/826576-momentum-enhanced-linear-regression-for-faster-convergence-in-real-world-predictions
[34] https://www.sec.gov/Archives/edgar/data/27673/000155837024016174/jdcc-20241027x10k.htm
[35] https://www.sec.gov/Archives/edgar/data/315189/000155837024016169/de-20241027x10k.htm
[36] https://www.sec.gov/Archives/edgar/data/1795815/000162828025015957/bcal-20241231.htm
[37] https://www.sec.gov/Archives/edgar/data/700564/000070056425000014/fult-20241231.htm
[38] https://www.sec.gov/Archives/edgar/data/40987/000004098725000034/gpc-20250227.htm
[39] https://www.sec.gov/Archives/edgar/data/1795815/000179581525000010/bcal-20250331.htm
[40] https://www.sec.gov/Archives/edgar/data/1050441/000105044125000088/egbn-20250331.htm
[41] https://www.mdpi.com/2071-1050/14/18/11674
[42] https://www.mdpi.com/2073-4433/13/8/1334
[43] https://journals.sagepub.com/doi/full/10.3233/JIFS-189208
[44] https://ieeexplore.ieee.org/document/9853185/
[45] https://www.iieta.org/journals/jesa/paper/10.18280/jesa.580217
[46] https://www.skillcamper.com/blog/a-beginners-guide-to-linear-regression-understanding-the-fundamentals
[47] https://www.machinelearningmastery.com/linear-regression-for-machine-learning/
[48] https://www.datacamp.com/tutorial/simple-linear-regression
[49] https://minjoo-happy-blog.tistory.com/38
[50] https://bookdown.org/content/b472c7b3-ede5-40f0-9677-75c3704c7e5c/fundamentals-of-linear-regression.html
[51] https://www.digitalocean.com/resources/articles/what-is-linear-regression
[52] https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-linear-regression/
[53] https://danawalab.github.io/machinelearning/2022/09/13/MachineLearning-LinearRegression.html
[54] https://www.sec.gov/Archives/edgar/data/1331421/000164117225000860/form10-k.htm
[55] https://www.sec.gov/Archives/edgar/data/1331421/000149315224039074/form8-k.htm
[56] https://www.sec.gov/Archives/edgar/data/1969302/000141057825000895/pony-20241231x20f.htm
[57] https://www.sec.gov/Archives/edgar/data/1763950/000164117225000926/form10-k.htm
[58] https://www.semanticscholar.org/paper/1580aecb6aafbf5303ee458d352739641d60f58a
[59] https://iopscience.iop.org/article/10.1088/1755-1315/1419/1/012032
[60] http://www.science-gate.com/IJAAS/2023/V10I8/1021833ijaas202308012.html
[61] https://ieeexplore.ieee.org/document/10314182/
[62] https://onepetro.org/SPEOKOG/proceedings/23OKOG/23OKOG/D031S011R002/518823
[63] https://ieeexplore.ieee.org/document/10170440/
[64] https://ieeexplore.ieee.org/document/10736623/
[65] https://pro.arcgis.com/en/pro-app/latest/tool-reference/geoai/how-linear-regression-works.htm
[66] https://dev.to/ahikmah/understanding-supervised-learning-the-basics-of-linear-regression-33ek
[67] https://www.reddit.com/r/learnmachinelearning/comments/9t048f/is_linear_regression_a_machine_learning_technique/
[68] https://www.youtube.com/watch?v=qxo8p8PtFeA
[69] https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/assumptions-of-linear-regression/
[70] https://www.linkedin.com/pulse/from-data-predictions-understanding-supervised-learning-pratik-shinde-lwh5f
[71] https://docs.tibco.com/pub/stat/14.0.0/doc/html/UsersGuide/GUID-F8407C18-73ED-44F3-98E9-B276184BD265.html
[72] https://www.linedata.com/main-supervised-regression-learning-algorithms
[73] https://stats.stackexchange.com/questions/486672/why-dont-linear-regression-assumptions-matter-in-machine-learning
[74] https://mlu-explain.github.io/linear-regression/
[75] https://www.sec.gov/Archives/edgar/data/1533998/000155837025002527/drio-20241231x10k.htm
[76] https://www.sec.gov/Archives/edgar/data/1977303/000197730325000006/lthm-20241231.htm
[77] https://www.sec.gov/Archives/edgar/data/39263/000003926325000017/cfr-20241231.htm
[78] http://idnrs.khpi.edu.ua/article/view/249768
[79] https://www.cureus.com/articles/313512-an-mri-derived-formula-for-estimating-the-native-joint-line-position-in-the-presence-of-distal-femoral-bone-loss
[80] https://www.tandfonline.com/doi/full/10.1080/10705511.2024.2375736
[81] http://www.veterinaryworld.org/Vol.13/March-2020/12.html
[82] https://www.semanticscholar.org/paper/016b0c8021f2c16188f2ed94342d31bde885dd69
[83] https://www.semanticscholar.org/paper/b6220999c73479a16039552ae3cc89c753e1677f
[84] http://ogscience.org/journal/view.php?doi=10.5468/ogs.2016.59.2.91
[85] https://arxiv.org/abs/2309.16846
[86] https://magnimetrics.com/least-squares-method-cost-function/
[87] https://statisticsbyjim.com/regression/linear-regression-equation/
[88] https://kenndanielso.github.io/mlrefined/blog_posts/8_Linear_regression/8_1_Least_squares_regression.html
[89] https://www.reddit.com/r/datascience/comments/15cskh6/how_to_improve_linear_regressionmodel_performance/
[90] https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/simple-linear-regression.html
[91] https://machine-learning-tutorial-abi.readthedocs.io/en/latest/content/overview/linear-regression.html
[92] https://www.cs.toronto.edu/~rgrosse/courses/csc311_f20/readings/notes_on_linear_regression.pdf
[93] https://byjus.com/maths/linear-regression/
[94] https://en.wikipedia.org/wiki/Linear_regression
[95] https://www.sec.gov/Archives/edgar/data/2000640/000121390025008966/ea0229409-s1a1_damon.htm
[96] https://link.springer.com/10.2991/artres.k.191224.042
[97] https://www.rsisinternational.org/journals/ijriss/articles/predictive-modeling-of-powerschool-usage-comparative-analysis-of-linear-regression-and-data-mining-techniques-using-student-attributes/
[98] https://www.dovepress.com/critical-appraisal-and-future-directions-for-the-association-between-a-peer-reviewed-fulltext-article-JIR
[99] https://dl.acm.org/doi/10.1145/3550074
[100] https://bmcresnotes.biomedcentral.com/articles/10.1186/s13104-020-05345-2
[101] https://academic.oup.com/book/39874/chapter/340060471
[102] https://www.sciencedirect.com/science/article/pii/S1877050923019750
[103] https://www.sciencedirect.com/science/article/pii/S1877050923019750/pdf?md5=a68089401ac3f749846eefd367a137d6&pid=1-s2.0-S1877050923019750-main.pdf
[104] https://www.iejme.com/download/linear-regression-model-to-predict-the-use-of-artificial-intelligence-in-experimental-science-15736.pdf
[105] https://dl.acm.org/doi/10.1145/3512290.3528750
[106] https://dl.acm.org/doi/abs/10.1016/j.procs.2023.11.144
[107] https://www.sciencedirect.com/science/article/abs/pii/S0021999124009161
[108] https://www.usaii.org/ai-insights/what-is-linear-regression-its-types-challenges-and-applications
[109] https://www.nature.com/articles/s41598-023-49899-0
[110] https://www.mdpi.com/1996-1073/15/14/5097
[111] https://pubs.aip.org/aip/acp/article-lookup/doi/10.1063/5.0259534
[112] https://radiopaedia.org/articles/57103
[113] https://www.matec-conferences.org/articles/matecconf/pdf/2018/35/matecconf_ifid2018_01033.pdf
[114] https://www.cambridge.org/core/services/aop-cambridge-core/content/view/52F06EF68EB20670B6CD1919C3C04D25/S0003055422001022a.pdf/div-class-title-relaxing-assumptions-improving-inference-integrating-machine-learning-and-the-linear-regression-div.pdf
[115] https://arxiv.org/html/2402.15213v1
[116] https://turcomat.org/index.php/turkbilmat/article/download/1092/874
[117] https://arxiv.org/abs/2105.04240
[118] https://arxiv.org/abs/2307.05189
[119] http://arxiv.org/pdf/2401.00186.pdf
[120] https://arxiv.org/pdf/2412.15633.pdf
[121] https://www.frontiersin.org/articles/10.3389/feart.2021.666444/pdf
[122] https://wikidocs.net/21670
[123] https://nittaku.tistory.com/284
[124] https://bcpublication.org/index.php/BM/article/view/3720
[125] https://www.ijraset.com/best-journal/comparative-study-on-supervised-machine-learning-algorithm
[126] https://arxiv.org/abs/1811.01586
[127] https://www.frontiersin.org/articles/10.3389/fonc.2023.1130229/pdf
[128] https://thescipub.com/pdf/jcssp.2020.1150.1162.pdf
[129] https://arxiv.org/pdf/2203.17193.pdf
[130] https://arxiv.org/pdf/2106.05526.pdf
[131] http://arxiv.org/pdf/2408.12308.pdf
[132] https://www.frontiersin.org/article/10.3389/fmolb.2020.00013/full
[133] https://pmc.ncbi.nlm.nih.gov/articles/PMC9949554/
[134] https://arxiv.org/pdf/2207.03336.pdf
[135] https://www.semanticscholar.org/paper/36a1699455aab1fd0cb394969eec6e826c10078d
[136] https://www.semanticscholar.org/paper/aba66666f12e4bf6faffd186d24b1e1a2d4aacda
[137] https://pmc.ncbi.nlm.nih.gov/articles/PMC6103548/
[138] https://pmc.ncbi.nlm.nih.gov/articles/PMC5122272/
[139] https://www.hindawi.com/journals/jfs/2019/1526920/
[140] https://openresearchlibrary.org/ext/api/media/1ffe902a-e3e8-45c3-b420-845483cee8ce/assets/external_content.pdf
[141] http://www.hrpub.org/download/20200130/MS1-13414594.pdf
[142] https://www.mdpi.com/1099-4300/25/4/611/pdf?version=1680518541
[143] https://www.mdpi.com/2227-7390/11/18/3957/pdf?version=1695102181
[144] http://www.scirp.org/journal/PaperDownload.aspx?paperID=61997
[145] https://tns.ewapublishing.org/media/1472974197554137bcf7e13d72c9e5fe.marked.pdf
[146] https://velog.io/@albert0811/ML-Linear-Regression-Model2
[147] https://math.stackexchange.com/questions/2774106/why-is-the-least-square-cost-function-for-linear-regression-convex
[148] https://koreascience.kr/article/JAKO202219559292353.page?lang=en
[149] https://www.colorado.edu/amath/sites/default/files/attached-files/ch12_0.pdf
[150] https://en.wikipedia.org/wiki/Least_squares
[151] https://ieeexplore.ieee.org/document/10522442/
[152] http://ieeexplore.ieee.org/document/6691313/
[153] https://eajournals.org/ejcsit/wp-content/uploads/sites/21/2024/06/An-Intelligent-Analytic-Framework.pdf
[154] https://arxiv.org/abs/1306.0114
[155] https://pmc.ncbi.nlm.nih.gov/articles/PMC7751867/
[156] https://pmc.ncbi.nlm.nih.gov/articles/PMC3810665/
[157] http://www.ccsenet.org/journal/index.php/mas/article/download/0/0/42533/44378
[158] http://medrxiv.org/cgi/content/short/2024.05.28.24308029v1?rss=1
[159] https://arxiv.org/pdf/1809.04838.pdf
[160] https://pmc.ncbi.nlm.nih.gov/articles/PMC11925299/

선형 회귀(Linear Regression)는 통계학 및 머신러닝에서 두 변수 또는 여러 변수 간의 관계를 선형 방정식으로 모델링하는 기법입니다. 독립 변수(설명 변수)와 종속 변수(반응 변수) 사이의 관계를 직선 형태로 나타내어, 종속 변수의 값을 예측하거나 변수 간 상관관계를 설명하는 데 사용됩니다.

주요 개념
목적:
예측: 관측된 설명 변수 값으로부터 종속 변수 값을 예측
설명: 설명 변수들이 종속 변수 변화에 미치는 영향 및 변수들 간의 관계 분석
수식(단순 선형 회귀):
( Y = mX + b )
여기서 (Y)는 종속 변수, (X)는 독립 변수, (m)은 기울기(변수 간 영향력), (b)는 절편(기준값)입니다.
특성 및 가정
설명 변수와 반응 변수 간의 선형 관계 존재
오차의 분포가 정규 분포를 따르고, 등분산성(오차들의 분산이 일정) 유지
독립 관측치(데이터 포인트 간 독립성)
예측 변수에 불확실성 없음
변수는 실제 변수여야 하며, 주성분 등 변형된 값이 아님
활용
통계 분석: 변수 간 상관관계 파악, 영향력 평가
머신러닝: 지도학습 알고리즘으로, 학습된 직선 모델을 이용해 새로운 데이터 예측
다양한 실무 분야에서 폭넓게 활용(경제, 사회과학, 공학 등)
기술적 구현 예
Python의 scikit-learn 라이브러리의 LinearRegression 클래스는 잔차 제곱합을 최소화하는 선형 모델 계수를 찾음
절편 계산 여부, 데이터 복사 여부 등의 하이퍼파라미터 조정 가능
이처럼 선형 회귀는 단순하면서도 직관적인 모델로, 데이터 내 변수 간 관계 분석과 예측을 위한 기본적인 도구로 널리 쓰입니다.

# Linear Regression — Car Price Prediction and Data Analysis

# Reference
- https://medium.com/@diellorhoxhaj/linear-regression-car-price-prediction-and-data-analysis-112883cdd39b
- https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/code
