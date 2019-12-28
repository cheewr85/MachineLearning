## 공정성
- ML의 일반적인 과정
  - 학습 데이터를 수집하고 주석을 추가 -> 모델을 학습 -> 콘텐츠를 필터링, 집계 또는 생성하거나 콘텐츠의 순위를 지정 -> 사용자에게 출력이 표시
- 학습 데이터를 수집하고 주석을 추가
  - 데이터에서 사람이 갖는 편향
    - 보고 편향
    - 표본 선택 편향
    - 과잉 일반화
    - 외부 집단 동질화 편향
    - 세상 데이터의 양식을 사용할 때 ML에 반영할 수 있는 '세상'의 무의식적 편향
    - 확증 편향
    - 자동화 편향
    - ML에 반영할 수 있는 절차의 무의식적 편향
- 공정성을 위한 설계
  - 1.문제 살펴보기
  - 2.전문가에게 문의
  - 3.모델이 편향을 고려하도록 학습시킴
  - 4.결과 해석
  - 5. 맥락과 함께 게시

### 편향의 유형
- 머신러닝 모델이라고 해서 본질적으로 객관적인 것은 아님/데이터 사전준비와 선정에 사람이 관여하기 때문에 모델의 예측이 편향되기 쉬움
- 일반적인 사람들의 편향을 인식하여 영향을 최소화할 수 있도록 미리 조치하는 것이 중요함 
- 아래의 유형은 모든 편향을 다른 것은 아님/참조:https://en.wikipedia.org/wiki/List_of_cognitive_biases/잠재적인 편향 원인도 생각해야함

- 보고 편향
  - 데이트 세트에 수집된 이벤트, 속성 및 결과의 빈도가 실제 빈도를 정확하게 반영하지 않을 때 나타남
  - 이 편향은 사람들이 '말할 필요도 없다고 느끼는' 일반적인 상황은 언급하지 않고 특별히 기억할 만하거나 특이항 상황만을 기록하려는 경향이 있어 발생
  - 예시
    - 인기 웹사이트에서 사용자가 제출하는 말뭉치를 바탕으로 도서 리뷰가 긍정인지 부정인지 예측하는 모델
    - 도서에 관해 별다른 의견이 없는 사람들은 리뷰를 제출할 가능성이 작기 때문에 리뷰 대다수는 극단적인 의견이 됨 
    - 이 모델은 좀 더 미묘한 어휘를 사용한 도서 리뷰의 감정을 정확히 예측할 가능성이 작음
- 자동화 편향
  - 두 시스템의 오류율과 관계없이 자동화 시스템이 생성한 결과를 비자동화 시스템이 생성한 결과보다 선호하는 경향
  - 예시
    - 톱니바퀴 제조업체에서 근무하는 소프트웨어 엔지니어가 톱니 결함을 파악하도록 학습한 새로운 '혁신적인' 모델을 배포하고 싶어함
    - 공장 감독자는 이 모델의 정밀도와 재현율이 사람 검사자보다 모두 15% 낮다고 지적함 
- 표본 선택 편향
  - 데이터 세트의 사례가 실제 분포를 반영하지 않는 방식으로 선정된 경우 발생함
  - 포함 편향/선택된 데이터가 대표성을 갖지 않음
  - 예시
    - 자사 제품을 구매한 소비자층을 대상으로 전화 설문조사를 하고 그 결과를 토대로 미래의 신제품 판매량을 예측하도록 학습된 모델
    - 경쟁업체 제품을 선택한 소비자는 설문조사 대상이 아니었기 때문에 결과적으로 이 소비자 그룹은 학습 데이터에 반영 안됨
  - 무응답 편향(응답 참여 편향)/데이터 수집 시 참여도의 격차로 인해 데이터가 대표성을 갖지 못함
  - 예시
    - 제품을 구매한 소비자 샘플과 경쟁업체 제품을 구매한 소비자 샘플을 대상으로 한 전화 설문자소를 통해 미래의 신제품 판매량 예측 모델
    - 경쟁업체 제품을 구매한 소비자는 설문조사를 완료하지 않을 가능성이 80% 높았기 때문에 이들의 데이터가 샘플에서 실제보다 작게 표현됨
  - 표본 추출 편향/데이터 수집 과정에서 적절한 무작위선택이 적용되지 않았음
  - 예시
    - 제품을 구매한 소비자 샘플과 경쟁업체 제품을 구매한 소비자 샘플을 대상으로 한 전화 설문조사를 통해 미래의 신제품 판매량을 예측하는 모델
    - 설문에서 소비자를 임의로 타겟팅하지 않고 선착순으로 이메일에 응답한 200명의 소비자를 선정함
    - 평균 구매자보다 제품에 관심이 많은 소비자가 다수 포함되었을 가능성이 높음
- 그룹 귀인 편향
  - 개인의 특성을 개인이 속한 그룹 전체의 특성으로 일반화하려는 경향을 말함
  - 내집단 편향/자신이 소속된 그룹 또는 본인도 공유하는 특성을 가진 그룹의 구성원을 선호하는 경향
  - 예시
    - 소프트웨어 개발자 모집을 위한 이력서 심사 모델을 학습시키는 두 명의 엔지니어는 자신들과 같은 컴퓨터 공학 아카데미에 다녔던 지원자가 더 직무에 적합하다고 믿음 
  - 외부 집단 동질화 편향/자신이 속하지 않은 그룹의 개별 구성원에 관해 고정 관념을 갖거나 그들이 모두 동일한 특징을 가진다고 판단하는 경향
  - 예시
    - 소프트웨어 개발자 모집을 위한 이력서 심사 모델을 학습시키는 두 명의 엔지니어는 자신들과 같은 컴퓨터 공학 아카데미에 다니지 않는 지원자가 직무에 필요한 전문지식이 부족하다고 믿음 
- 내재적 편향
  - 일반적으로 적용할 필요가 없는 자신의 정신적 모델과 개인적 경험을 바탕으로 가정할 때 발생함 
  - 예시
    - 동작 인식 모델을 학습시키는 중인 엔지니어가 '아니요'라는 단어를 나타내는 고개 가로 젖기를 기능으로 사용하려고 함
    - 일부 지역에서 고개를 가로 젓는 것은 반대로 '예'를 의미하기도 함
  - 내재적 편향의 일반적인 형태는 확증 편향으로 모델을 만드는 사람이 자기도 모르게 이미 가지고 있는 믿음이나 가설을 긍정하는 방향으로 데이터를 처리하는 것을 말함 
  - 경우에 따라 모델을 만드는 사람이 자신의 원래 가설과 일치할 때까지 반복해서 모델을 학습시키기도 하는데 이를 실험자 편향이라고 함 
  - 예시
    - 엔지니어가 다양한 특징(키,체중,종,환경)을 바탕으로 개의 공격성을 예측하는 모델을 만들고 있음
    - 이 엔지니어는 어릴 때 활동성이 강한 토이 푸들로 인해 불쾌한 일이 있었음
    - 이후 항상 토이푸들을 공격적인 종이라고 생각
    - 학습된 모델이 대부분의 토이 푸들이 상대적으로 유순하다고 예측했을 때 엔지니어는 크기가 작은 푸들이 더 공격적이라는 결과가 나올 때까지 모델을 여러번 다시 학습시킴 
    
### 편향 식별하기
- 모델에서 데이터를 가장 잘 표현할 방법을 찾기 위해 데이터를 살펴볼 때 공정성 문제를 염두에 두고 편향의 원인이 될 수 있는 요소를 사전에 점검하는 것이 중요 
- 특성 값 누락
  - 데이터 세트의 다수의 예에서 값이 누락된 특성이 하나 이상 있는 경우 데이터 세트의 주요 특성 중 일부가 제대로 표현되지 않았음을 나타내는 지표일 수 있음 
  - 예시
  - 아래의 사진을 통해서 모든 특성의 count가 17000이라는 것은 누락된 값이 없음을 나타냄
  <img src="https://user-images.githubusercontent.com/32586985/71537168-7af71f80-295b-11ea-9982-f6306986117c.PNG">
  
  - 아래의 사진을 보게 된다면 3가지 특성의 count가 3000이라고 가정할 시 각 특성에 14000개의 누락된 값이 있는것
  - 14000개의 값이 누락되어 가구의 평균 소득과 주택 가격의 중앙값을 정확히 연관시키기 훨씬 어려워졌음
  - 이 데이터 모델을 학습하기 전에 누락된 값의 원인을 신중하게 조사하여 소득 및 인구 데이터 누락의 원인이 될 수 있는 잠재적인 편향이 없는지 확인하는 것이 좋음 
  <img src="https://user-images.githubusercontent.com/32586985/71537194-d2958b00-295b-11ea-8e01-6d41f5c75a4c.PNG">
  
- 예기치 않은 특성 값
  - 데이터를 살펴볼 때 특이하거나 비정상적인 특성 값을 포함하는 예가 있는지 확인해 보아야 함
  - 이와 같이 예기치 않는 특성 값이 있다는 것은 데이터 수집 중에 문제가 발생했거나 편향을 일으킬 수 있는 기타 부정확성이 있음을 나타낼 수 있음 
  - 예시
  <img src="https://user-images.githubusercontent.com/32586985/71537216-1c7e7100-295c-11ea-9036-0971e4380c7a.PNG">
  
  - 예기치 않는 특성 값은 4번의 예인데 이 좌표는 미국 캘리포니아 주에 속하지 않음
  <img src="https://user-images.githubusercontent.com/32586985/71537224-4172e400-295c-11ea-8237-882a970567bf.PNG">
  
- 데이터 격차
  - 특정 그룹이나 특성이 실제보다 과소 또는 과대 표현되는 모든 종류의 데이터 격차로 인해 모델에 편향이 생길 수 있음
  - 검증 프로그래밍 실습에서 학습 세트와 검증 세트로 나누기 전에 캘리포니아 주택 데이터 세트를 무작위로 섞지 않아서 확연한 데이터 격차가 생긴것을 알 수 있음 
  - 그림 1은 전체 데이터 세트에서 추출한 데이터의 하위 집합을 캘리포니아 북서부 지역만 나타내도록 시각화한 것임 
  - 캘리포니아주 지도 위에 캘리포니아 주택 데이터 세트의 데이터를 오버레이한 그림
  - 각각의 점은 주택 단지를 나타내며, 파란색은 주택 가격 중앙값이 낮은 곳을, 빨간색은 주택 가격 중앙값이 높은 곳임을 나타냄
  <img src="https://user-images.githubusercontent.com/32586985/71537257-be05c280-295c-11ea-8d1f-3ea21f4b341c.PNG">
  
  - 이와 같이 전체를 잘 대표하지 못하는 샘플을 캘리포니아주 전체의 주택 가격 예측을 위한 모델을 학습하는 데 사용했다면 캘리포니아주 남부의 주택 데이터가 없는것이 문제가 됨
  - 모델에 인코딩된 지리적 편향은 데이터에 표현되지 않은 커뮤니티의 주택 구매자에게 부정적인 영향을 미칠 수 있음 
  
  ### 편향 평가
  - 모델을 평가할 때 전체 테스트 또는 검증세트를 기준으로 계산된 지표가 모델의 공정성에 관해 항상 정확한 모습을 보여주는 것은 아님 
  - 환자의 의료 기록 1000개로 구성된 검증세트가 있다고 가정/그를 바탕으로 종양의 유무를 예측하는 새로운 모델 개발
  - 500개 기록은 여성환자의 기록이고 나머지 500개는 남성 환자의 기록임
  - 다음 혼동행렬은 전체 1000개 사례의 결과를 요약한 것임
  <img src="https://user-images.githubusercontent.com/32586985/71537293-4e440780-295d-11ea-85ae-acb549186ad9.PNG">
  
  - 정밀도가 80%이고 재현율이 72.7%이므로 유먕한 결과로 보임/하지만 각 환자 세트에 관한 결과를 따로 계산한다면?
  - 각 그룹 모델의 성과가 크게 다른 것을 알 수 있음 
  <img src="https://user-images.githubusercontent.com/32586985/71537304-82b7c380-295d-11ea-9eae-4e1bf873e2e0.PNG">
  
  - 여성 환자
    - 실제로 종양이 있었던 여성 환자 11명에 관해 모델은 10명의 환자를 양성으로 정확하게 예측하여 90.9% 재현율을 보임
    - 이 모델은 여성 환자 사례의 9.1%에 관해서는 종양 진단을 놓친 것임
  - 남성 환자
    - 하지만 실제로 종양이 있었던 남성 환자 11명에 관해 모델은 6명의 환자만을 양성으로 정확하게 예측함(재현율 54.5%)
    - 이 모델은 남성 환자 사례의 45.5%에 관해서는 종양 진단을 놓친 것임
    - 모델이 남성 환자의 종양에 양성을 반환할 때 9개 중 6개 사례에서만 정확함(정밀도 66.7%)
    - 다시 말해 이 모델은 남성 환자 사례의 33.3%에 관해서는 종양을 잘못 예측한 것임 


## 프로그래밍 실습
- 사용할 데이터는 성인 인구의 수입 조사 데이터를 쓸 것임/1994년 인구조사 기반
- 예측 과제는 매년 5만달러 이상을 버는 사람을 예측하는 것임
- income_bracket:매년 5만달러 이상 버는 사람
- 설정
```python
   import os
   import numpy as np
   import matplotlib.pyplot as plt
   import pandas as pd
   import tensorflow as tf
   import tempfile
   !pip install seaborn==0.8.1
   import seaborn as sns
   import itertools
   from sklearn.metrics import confusion_matrix
   from sklearn.metrics import roc_curve, roc_auc_score
   from sklearn.metrics import precision_recall_curve
   from google.colab import widgets
   # For facets
   from IPython.core.display import display, HTML
   import base64
   !pip install facets-overview==1.0.0
   from facets_overview.feature_statistics_generator import FeatureStatisticsGenerator
   
   print('Modules are imported.')
```
<img src="https://user-images.githubusercontent.com/32586985/71538407-bf8bb680-296d-11ea-93e3-df73ac8cbe51.PNG">


- 데이터 불러오기
```python
   COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
              "marital_status", "occupation", "relationship", "race", "gender",
              "capital_gain", "capital_loss", "hours_per_week", "native_country",
              "income_bracket"]
   
   train_df = pd.read_csv(
       "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
       names=COLUMNS,
       sep=r'\s*,\s*',
       engine='python',
       na_calues="?")
   test_df = pd.read_csv(
       "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
       names=COLUMNS,
       sep=r'\s*,\s*',
       skiprows=[0],
       engine='python',
       na_values="?")
       
   # Drop rows with missing values
   train_df = train_df.dropna(how="any", axis=0)
   test_df = test_df.dropna(how="any", axis=0)
   
   print('UCI Adult Census Income dataset loaded.')
```
<img src="https://user-images.githubusercontent.com/32586985/71538492-42614100-296f-11ea-88ff-c854ee22b308.PNG">

- 데이터세트 분석
- 모델을 예측하기 이전에 몇가지 데이터 세트에 대한 확인을 한 뒤 모델을 사용해야한다
<img src="https://user-images.githubusercontent.com/32586985/71538535-067aab80-2970-11ea-99f4-4180fbe32fb0.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71538530-f236ae80-296f-11ea-84fd-09dc4473b90a.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71538532-fbc01680-296f-11ea-9412-b10356a02498.PNG">

### 과제1
- 위의 실행한 통계자료와 히스토그램을 바탕으로 Show Raw Data 버튼을 눌러 카테고리 별로 값의 분포를 확인해봐라
- 아래의 질문에 대한 답변을 해바라
  - 1.관측한 많은 수들 중 잃어버린 특성 값이 있는가?
  - 2.다른 특성에 영향을 줄 수 있는 잃어버린 특성 값이 있는가?
  - 3.예측하지 못한 특성 값이 있는가?
  - 4.데이터가 한 쪽으로 쏠린 현상을 볼 수 있는가?
- numeric과 categorical 특성을 봤을 때 별도의 잃어버린 특성이 있진 않았음
- 데이터를 본다면 hours_per_week 같은 경우 최소값이 1이지만 최대값은 99인 큰 격차가 보임을 알 수 있음
- capital_gain과 capital_loss 같은 경우 90%이상의 값이 0이고 10% 미만의 경우가 0이 아닌 값인 것을 알 수 있음/그로 인해 이 값에 대해 좀 더 자세히 보고 이 값을 증명하거나 이 특성이 유효한지 확인을 해봐야함
- gender의 히스토그램을 보게 되면 3분의 2의 해당하는 예시가 남자를 가르키고 있는데 이것은 충분히 데이터가 쏠린 현상으로 볼 수 있기 때문에 성비를 50/50의 가깝게 맞춰야함

### 과제2
- 데이터 세트를 좀 더 깊게 탐구하기 위해 Facets Dive를 사용할 것임
- 해당 툴은 데이터 포인터로 나타내는 것을 시각화하여 각각의 특성들이 상호작용하는 것을 볼 수 있음
<img src="https://user-images.githubusercontent.com/32586985/71538663-0f6c7c80-2972-11ea-98a0-0877bf0cd5bb.PNG">

- 해당 툴을 메뉴를 이용하여 시각화의 변화를 알아볼 예정
  - 1.X-Axis 메뉴,Color를 education을 고르고 Type 메뉴의 경우 income_bracket을 고른 후 그 둘 사이의 관계를 확인
    - 이 데이터 세트에서는 높은 학력을 가진 것이 많은 수입을 가지고 있다는 것과 연관되어 있음
    - 수입이 5만달러 이상일 경우 학력 수준이 높은 것으로 나옴
  <img src="https://user-images.githubusercontent.com/32586985/71538711-07f9a300-2973-11ea-9f0e-99ad104a7d43.PNG">
  
  - 2.X-Axis 메뉴,Color를 marital_status로 고르고 Type 메뉴의 경우 gender를 고른 후 주목할 만한 관찰은 무엇인지 확인
    - marital-status 카테고리에서 남성과 여성의 비율이 거의 1:1에 가까움/남성과 여성의 비율이 5:1인 married-civ-spouse를 제외하고는
    - 과제 1에서 알 수 있듯이, 남성의 비율이 많은 이 데이터 세트에서는 결혼한 여성의 비율이 특히 낮다는 것을 추론할 수 있음
  <img src="https://user-images.githubusercontent.com/32586985/71538732-822a2780-2973-11ea-9419-4ceee4de72bd.PNG">
  
### 과제3
- 공정성에 관하여 문제를 일으킬수 있는 특성에 대하여 알아보자
- 해당 특성들이 모델의 예측의 어떠한 영향을 주는지 확인
<img src="https://user-images.githubusercontent.com/32586985/71538867-dcc48300-2975-11ea-89a2-4b028733f458.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71538869-e0f0a080-2975-11ea-8f7e-395ef53e891c.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71538871-e3eb9100-2975-11ea-91b9-56a1026b0e46.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71538873-e6e68180-2975-11ea-9d8b-48a9fcc76b55.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71538875-e9e17200-2975-11ea-8014-780d5db1eb51.PNG">

- TensorFlow Estimators를 사용하여 예측하기
  - neural network를 통해서 수입을 예측할 것이고, TensorFlow's Estimator API를 통해 DNNClassifier에 접근할 것임
  - 우선 tensors에 있는 input 함수를 사용할 것
  ```python
     def csv_to_pandas_input_fn(data, batch_size=100, num_epochs=1, shuffle=False):
       return tf.estimator.inputs.pandas_input_fn(
           x=data.drop('income_bracket', axis=1),
           y=data['income_bracket'].apply(lambda x: ">50K" in x).astype(int),
           batch_size=batch_size,
           num_epochs=num_epochs,
           shuffle=shuffle,
           num_threads=1)
           
     print('csv_to_pandas_input_fn() defined.')      
  ```
  <img src="https://user-images.githubusercontent.com/32586985/71538960-9bcd6e00-2977-11ea-8ce4-90755410517c.PNG">
  
- TensorFlow에서의 특성 나타내기
- 모델의 data maps를 충족시키기 위함
 <img src="https://user-images.githubusercontent.com/32586985/71538975-d800ce80-2977-11ea-9382-a054942eda7a.PNG">
 
- 과제3을 하는동안 age를 선택했다면 age가 bucketing을 하고 서로 다른 그룹안에서 비슷한 나이끼리 묶는게 수월함
- 이것은 모델이 나이를 좀 더 일반화하기 좋음/카테고리 특성에 대해 age라는 숫자 특성을 변환하는
```python
   age_buckets = tf.feature_column.bucketized_column(
       age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```
- 주요한 서브그룹 역시 고려해야함 
- 모델의 특성 정의하기
  - gender를 subgroup으로 고려하고 이것을 별도의 subgroup_variables 리스트로 분리한후 필요로하는 특별한 요소를 추가할 것임
```python
   # List of variables, with special handling for gender subgroup.
   variables = [native_country, education, occupation, workclass,
                relationship, age_buckets]
   subgroup_variables = [gender]
   feature_columns = variables + subgroup_variables
```
- Adult 데이터세트에 Deep Neural Net Model 학습시키기
  - 딥러닝을 사용하여 수입을 예측할 것임
  - 두 개의 히든 레이어가 있는 간단한 구조로 할 것임
  - 그 전에 고차원의 카테고리 특성을 낮은 차원의 밀집된 실제 벡터 값으로 전환해야함(임베딩)
  ```python
     deep_columns = [
         tf.feature_column.indicator_column(workclass),
         tf.feature_column.indicator_column(education),
         tf.feature_column.indicator_column(age_buckets),
         tf.feature_column.indicator_column(gender),
         tf.feature_column.indicator_column(relationship),
         tf.feature_column.indicator_column(native_country, dimension=8),
         tf.feature_column.indicator_column(occupation, dimension=8),
     ]
     
     print(deep_columns)
     print('Deep columns created.') 
     
     [IndicatorColumn(categorical_column=VocabularyListCategoricalColumn(key='workclass', vocabulary_list=('Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'), dtype=tf.string, default_value=-1, num_oov_buckets=0)), IndicatorColumn(categorical_column=VocabularyListCategoricalColumn(key='education', vocabulary_list=('Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'), dtype=tf.string, default_value=-1, num_oov_buckets=0)), IndicatorColumn(categorical_column=BucketizedColumn(source_column=NumericColumn(key='age', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), boundaries=(18, 25, 30, 35, 40, 45, 50, 55, 60, 65))), IndicatorColumn(categorical_column=VocabularyListCategoricalColumn(key='gender', vocabulary_list=('Female', 'Male'), dtype=tf.string, default_value=-1, num_oov_buckets=0)), IndicatorColumn(categorical_column=VocabularyListCategoricalColumn(key='relationship', vocabulary_list=('Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'), dtype=tf.string, default_value=-1, num_oov_buckets=0)), EmbeddingColumn(categorical_column=HashedCategoricalColumn(key='native_country', hash_bucket_size=1000, dtype=tf.string), dimension=8, combiner='mean', initializer=<tensorflow.python.ops.init_ops.TruncatedNormal object at 0x7f5381d39630>, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True), EmbeddingColumn(categorical_column=HashedCategoricalColumn(key='occupation', hash_bucket_size=1000, dtype=tf.string), dimension=8, combiner='mean', initializer=<tensorflow.python.ops.init_ops.TruncatedNormal object at 0x7f5381d394a8>, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True)]
     
Deep columns created.
  
- 데이터 처리과정이 된 이후 deep neural net model을 정의할 수 있음
- 아래의 예시와 같이 정의함
<img src="https://user-images.githubusercontent.com/32586985/71539067-fec00480-2979-11ea-99af-974d777de17f.PNG">

- INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_model_dir': '/tmp/tmpjy2lakmf', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
graph_options {
  rewrite_options {
    meta_optimizer_iterations: ONE
  }
}
, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f5381d39710>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
Deep neural net model defined.

- 위의 일련의 과정을 거친후 1000번의 과정을 통해 학습시킬 것임
<img src="https://user-images.githubusercontent.com/32586985/71539086-58283380-297a-11ea-80f4-29942ed47616.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71539088-5bbbba80-297a-11ea-98c0-63d87d2d0cfc.PNG">

- 그리고 이 모델의 전체적인 성능을 평가함
- 다른 매개변수를 이용하여 모델을 다시 학습시킬 수 있음
- 각각의 서브그룹에 대해서 평가를 하지 못한 것이 하나 놓친부분이긴 함
<img src="https://user-images.githubusercontent.com/32586985/71539099-86a60e80-297a-11ea-8dc0-337ba2f70943.PNG">
  
### 과제4  
- Confusion 행렬을 사용하여 공정성 평가하기
- 모델에 대한 전반적인 성능에 대해서는 평가하지만 얼마나 잘 모델이 서로다른 subgroup에 대해서 작동했는지는 알 수 없음
- 모델의 공정성을 평가하는데 있어 다른 것들에 대한 특정한 예측 오류라던지 subgroup을 넘은 적합성에 대한 예측 오류등을 결정하는 것이 중요함
- 이것의 중요한 도구가 confusion matrix임/이를 바탕으로 모델이 얼마나 정확히 예측하는지 혹은 잘못 예측하는지 알 수 있음
- 우선 라벨에 대해서 두가지 가능한 값으로 binary 표현을 할 것이고 >50k는 positive, <50k는 negative임 
- 이 라벨은 단순히 값을 판단하는 용도가 아닌 두 가지 가능한 예측을 분류하는 용도로 쓰임 
- 제대로 된 예측을 한 것은 true로 잘못된 예측을 한 것은 false로 나타냄
  - true positive:모델이 >50k를 예측하고 true일 경우
  - true negative:모델이 <50k를 예측하고 true일 경우
  - false positive:모델이 >50k를 예측하고 false일 경우
  - false negative:모델이 <50k를 예측하고 false일 경우
- 아래의 실행은 우리의 이진 confusion 행렬과 평가 metrics에 필요한 것을 계산하는 실행임
<img src="https://user-images.githubusercontent.com/32586985/71539116-fcaa7580-297a-11ea-87d6-e47f9fbd34ef.PNG">

- 아래의 실행은 이진 confusion matrix를 시각화하는데 필요한 실행
<img src="https://user-images.githubusercontent.com/32586985/71539117-003dfc80-297b-11ea-8ab9-36564d255219.PNG">

- 이제 필요한 함수는 모두 정의를 하였고 이진 confusion 행렬과 평가 metrics를 deep neural net model을 바탕으로 결과를 냄
- gender의 subgroup을 선택하여 여성과 남성을 가지고 confusion matrics를 생성하고
- 각각의 값에 대해서 비교를 하고 확인을 해보자
- 다른 subgroup에 비해 모델의 오류의 비율이 더 나은 부분을 제시할 만한 부분이 있는지?
<img src="https://user-images.githubusercontent.com/32586985/71539118-02a05680-297b-11ea-8e1f-78efcf11a870.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71539119-07650a80-297b-11ea-9ad3-bae545be7d37.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71539316-cde1ce80-297d-11ea-80ea-6e70c7ac963f.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71539317-d1755580-297d-11ea-87f9-87941bfecdb6.PNG">

- 모델은 남성의 데이터가 여성의 데이터보다 더 나은 결과를 수행함
- 이 결과를 바탕으로 subgroup을 통해서 모델의 수행능력을 평가하는것이 중요함을 찾음
- 좋은 결정을 위해 위의 4가지 특성을 균형있게 하는것이 중요함/예를 들면 낮은 false positive 비율이면 높은 true positive 비율과 같이
- 혹은 높은 precision을 원하면 낮은 recall인 것
- 이러한 원하는 tradeoffs를 위해 평가 metrics를 
  
