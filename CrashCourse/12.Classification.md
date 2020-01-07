## 분류 및 회귀(Video Lecture)
- 확률 결과에 로지스틱 회귀를 사용하기도 하는데, 이 회귀의 형태는(0,1)임
- 다른 경우 별개의 이진 분류 값에 임계값을 설정함
- 임계값 선택은 중요하며 값을 조정할 수 있음 
### 평가 측정항목: 정확성
- 분류 모델을 어떻게 평가해야 할까?
  - 가능한 한 가지 측정 방법:정확성 
    - 올바른 예측의 비율
- 오해하기 쉬운 정확성
  - 대부분의 경우 정확성은 잘못되거나 오해하기 쉬운 측정항목임 
    - 다양한 종류의 실수에 여러 비용이 소요되는 경우가 많음
    - 대표적인 경우로 양성이나 음성이 거의 없는 클래스 불균형을 들 수 있음 
- TP 및 FP
  - 클래스 불균형 문제의 경우 다양한 종류의 오류를 분리하는데 유용함
  - 2X2 그리드 여러 종류의 성공과 실패(예 양치기소년)
    - TP(True Positive):우리는 늑대가 나타났다고 올바로 외쳤습니다.우리는 마을을 구했습니다 | FP:오류-우리는 늑대가 나타났다고 허위로 외쳤습니다. 모두 우리에게 화가 났습니다.
    - FN(False Negative):늑대가 있었지만 우리는 늑대를 발견하지 못했습니다.늑대가 닭을 모두 잡아먹었습니다. | TN:늑대도 없고 경고도 없었습니다. 모두 괜찮습니다.
- 평가 측정항목: 정밀도와 재현율
  - 정밀도:(참 양성(TP)) / (모든 양성 예측)
    - 모델이 '양성' 클래스로 나타났을 때 결과가 옳았나?
    - 직관: 모델이 '늑대다'라고 너무 자주 외쳤나?
  - 재현율:(참 양성(TP)) / (실제 양성 모두)
    - 모든 양성 가능성 중에서 모델이 몇 개나 정확히 식별했나?
    - 직관: 모델에서 놓친 늑대가 있나?
### ROC 곡선(수신자 조작 특성 곡선)
- 각 점은 하나의 결정 임계값에 있는 참양성(TP)과 거짓양성(FP)비율임 
<img src="https://user-images.githubusercontent.com/32586985/70700711-7fda9300-1d0e-11ea-9eef-e158ef3bf37a.png">

- 평가 측정항목:AUC
  - 해석: 임의의 양성 예측과 임의의 음성 예측을 선택할 때 내 모델이 정확한 순서로 순위를 매길 확률이 얼마나 될까?
  - 직관: 가능한 모든 분류 임계값에서 집계된 성능의 집계 측정값을 제공함 
- 예측 편향
  - 로지스틱 회귀 예측은 편향되지 않아야함
    - 예측 평균 == 관찰 평균
  - 편향을 보면 시스템의 상태를 판단할 수 있음
    - 편향이 0이라고 해서 시스템 전체가 완벽하다는 것을 의미하지 않음
    - 하지만 상태를 확인하는 데 매우 유용함 
  - 편향이 있으면 문제가 발생함 
    - 특성 세트가 불완전한가?
    - 파이프라인에 결함이 있나?
    - 학습 샘플이 편향적인가?
  - 편향은 캘리브레이션 레이어에서 수정하지 말고 모델에서 수정해라
  - 데이터 내에서 편향을 찾아라 그러면 개선할 수 있음
- 바케팅 편향을 나타내는 캘리브레이션 플롯
<img src="https://user-images.githubusercontent.com/32586985/70701432-b2d15680-1d0f-11ea-9b35-70209588719a.png">


## 분류:임계값
- 로지스틱 회귀는 확률을 반환함/반환된 확률을 '있는 그대로' 사용하거나,이진 값으로 변환하여 사용함 
- 로지스틱 회귀 값을 이진 카테고리에 매핑하려면 분류 임계값(결정 임계값)을 정의해야함
  - 임계값보다 높은 값은 스팸을 나타내고 임계값보다 낮은 값은 스팸아님을 나타냄 
  - 임계값은 문제에 따라 달라지므로 값을 조정해야함 

## 분류:참 대 허위, 양성 대 음성
- 예시
<img src="https://user-images.githubusercontent.com/32586985/70701932-9255cc00-1d10-11ea-89e4-2a2deabaaeae.png">

- '늑대다'는 양성 클래스임
- '늑대가 없다'는 네거티브 클래스임 
- '늑대 예측' 모델에서 발생할 수 있는 4가지 결과를 요약하면 다음과 같이 2X2 혼동 행렬을 사용해 나타낼 수 있음 
<img src="https://user-images.githubusercontent.com/32586985/70702048-c29d6a80-1d10-11ea-9f71-1a73e754fecf.png">

- 참양성은 모델에서 포지티브 클래스를 정확하게 평가하는 결과임
- 참음성은 모델에서 네거티브 클래스를 정확하게 평가하는 결과임 
- 허위양성은 모델에서 포지티브 클래스를 잘못 예측한 결과임
- 허위음성은 모델에서 네거티브 클래스를 잘못 예측한 결과임 

## 분류:정확성
- 정확성은 분류 모델 평가를 위한 측정항목 중 하나임/비공식적으로 모델의 예측이 얼마나 정확한가를 보여줌
- 공식적으로 정확성의 정의는 다음과 같음 
<img src="https://user-images.githubusercontent.com/32586985/70702371-4c4d3800-1d11-11ea-970e-be7304b8594c.png">

- 이진 분류에서는 다음과 같이 양성과 음성을 기준으로 정확성을 계산할 수도 있음 
- 여기에서 TP=참 양성(True Positives),TN=참 음성(True Negatives),FP=허위 양성(False Positives),FN=허위 음성(False Negatives)
<img src="https://user-images.githubusercontent.com/32586985/70702434-69820680-1d11-11ea-9df2-df657a0b818b.png">

- 다음과 같이 악성으로 분류된 종양 또는 양성으로 분류된 종양 모델 100개의 정확성을 계산해봄
<img src="https://user-images.githubusercontent.com/32586985/70702599-c2ea3580-1d11-11ea-9e99-03bff1f463e6.png">

- 정확성은 0.91 또는 91%(총 100개의 예제 중 정확한 예측 91개)로 나타남/이는 종양 분류자가 악성 종양을 제대로 식별했음을 의미함
- 더 면밀하게 분석시 모델의 성능을 자세히 파악 할 수 있음 
  - 종양 예제 100개 중 91개는 양성(참 음성 90개 허위 양성 1개)이고, 9개는 악성(참 양성 1개와 허위 음성 8개)임 
  - 모델은 양성 종양 91개 중 90개를 양성으로 정확히 식별함/뛰어난 예측 능력임 
  - 악성 종양 9개 가운데 1개만 악성으로 식별/8개가 미확진 상태로 남았음/형편없음 
  - 91% 정확성이 좋아 보여도 이 예제에선 항상 양성으로 예측하는 다른 종양 분류자 모델도 정확히 동일한 정확성을 달성할 것임 
  - 악성과 양성을 구분하는 예측 능력이 0인 모델과 비교해 나을 바가 없음
- 이와 같이 클래스 불균형 데이터 세트를 사용하면 양성 라벨수와 음성 라벨수가 상당히 다르므로 정확성만으로는 모든 것을 평가할 수 없음 

## 분류:정밀도와 재현율
- 정밀도
  - 양성으로 식별된 사례 중 실제로 양성이었던 사례의 비율은 어느 정도인가요?
  - 정밀도는 다음과 같이 정의됨 
  <img src="https://user-images.githubusercontent.com/32586985/70702988-85d27300-1d12-11ea-9de4-322fb33e6966.png">
  
  - ML 모델의 정밀도를 계산해 봄
  - 이 모델의 정밀도는 0.5임/평가가 정확할 확률은 50%임 
  <img src="https://user-images.githubusercontent.com/32586985/70703087-b0243080-1d12-11ea-8a3d-3845b831f89f.png">
  
- 재현율
  - 실제 양성 중 정확히 양성이라고 식별된 사례의 비율은 어느정도인가?
  - 재현율은 다음과 같이 정의됨 
  <img src="https://user-images.githubusercontent.com/32586985/70703182-e5c91980-1d12-11ea-9e43-eec152924c7a.png">
  
  - 재현율 계산
  - 이 모델은 재현율이 0.11임/모든 악성 종양 중 11%가 정확하게 식별됨 
  <img src="https://user-images.githubusercontent.com/32586985/70703235-ff6a6100-1d12-11ea-9f79-bd0214f6501d.png">
  
- 정밀도 및 재현율:줄다리기
  - 모델의 효과를 완전히 평가하려면 정밀도와 재현율을 모두 검사해야함 
  - 하지만 정밀도와 재현율은 상충되는 관계에 있는 경우가 많음/정밀도가 향상되면 재현율이 감소되거나 반대의 경우도 생김 
  - 예시:이메일 분류 모델로 만든 30개 예측/분류 임계값 오른쪽에 있는 메일은 '스팸'으로 분류되는 반면 왼쪽에 있는 메일은 '스팸 아님'으로 분류
  <img src="https://user-images.githubusercontent.com/32586985/70703422-7bfd3f80-1d13-11ea-9c59-ffd1ff1efd29.png">
  
  - 정밀도는 정확하게 분류된 스팸으로 신고된 이메일의 비율/임계값 선 오른쪽에 있으며 초록색으로 표시된 점의 비율 측정
  <img src="https://user-images.githubusercontent.com/32586985/70703534-b1099200-1d13-11ea-80c4-3022ab1789c9.png">
  
  - 재현율은 정확하게 분류된 실제 스팸 이메일의 비율/임계값 선 오른쪽에 있는 초록색 점의 비율을 측정
  <img src="https://user-images.githubusercontent.com/32586985/70703540-b4048280-1d13-11ea-9a0e-6d7ec4b3db4e.png">
  
  - 분류 임계값 증가의 효과
  <img src="https://user-images.githubusercontent.com/32586985/70703702-f4640080-1d13-11ea-985b-955b8b1cbe1f.png">
  
  - 허위 양성(FP)수는 감소하지만 허위 음성(FN)수는 증가함/정밀도는 증가하는 반면 재현율은 감소함 
  <img src="https://user-images.githubusercontent.com/32586985/70703707-f6c65a80-1d13-11ea-9d91-0680d8d950b0.png">
  
  - 분류 임계값 감소의 효과
  <img src="https://user-images.githubusercontent.com/32586985/70703801-283f2600-1d14-11ea-82a6-3783d2892087.png">
  
  - 허위 양성(FP)이 증가하고 허위 음성(FN)은 감소함/정밀도가 감소하고 재현율이 증가함 
  <img src="https://user-images.githubusercontent.com/32586985/70703829-355c1500-1d14-11ea-82c1-12e567aed122.png">
  
  
## 분류:ROC 및 AUC
- ROC 곡선 
  - ROC 곡선(수신자 조작 특성 곡선)은 모든 분류 임계값에서 분류 모델의 성능을 보여주는 그래프임 
  - 참 양성 비율(TPR)
    - 재현율의 동의어
    <img src="https://user-images.githubusercontent.com/32586985/70839429-195d8e00-1e50-11ea-9036-262212c8855f.PNG">
    
  - 허위 양성 비율(FPR)
    <img src="https://user-images.githubusercontent.com/32586985/70839444-2d08f480-1e50-11ea-802b-38bc86ce4937.PNG">
    
  - 다양한 분류 임계값의 TPR 및 FPR을 나타냄/분류 임계값을 낮추면 더 많은 항목이 양성으로 분류되므로 거짓양성과 참양성이 모두 증가함 
  - 밑의 그림은 일반 ROC곡선을 보여줌
  <img src="https://user-images.githubusercontent.com/32586985/70839523-66d9fb00-1e50-11ea-9d50-36eddb1b429e.PNG">
  
- ROC 곡선의 점을 계산하기 위해 분류 임계값이 다른 로지스틱 회귀 모형을 여러 번 평가할 수 있지만 효율적이지 못함 
- 이 정보를 제공하는 효율적인 정렬 기반 알고리즘이 있는데 이를 AUC라고 함
- AUC:ROC 곡선 아래 영역
  - AUC는 ROC 곡선 아래 영역을 의미함/(0,0)에서 (1,1)까지 전체 ROC 곡선 아래에 있는 전체 2차원 영역을 측정함 
  <img src="https://user-images.githubusercontent.com/32586985/70839584-c33d1a80-1e50-11ea-95e5-3966d2f292e2.PNG">
  
  - AUC는 가능한 모든 분류 임계값에서 성능의 집계 측정값을 제공함
  - 해석하는 한 가지 방법은 모델이 임의 양성 예제를 임의 음성 예제보다 더 높게 평가할 확률임 
  - 다음 예는 로지스틱 회귀 예측의 오름차순으로 왼쪽에서 오른쪽으로 정렬되어 있음 
  <img src="https://user-images.githubusercontent.com/32586985/70839655-0bf4d380-1e51-11ea-8ff1-b6a292764158.PNG">
  
  - AUC는 임의의 양성(초록색)예제가 임의의 음성(빨간색)예제의 오른쪽에 배치되는 확률을 나타냄 
  - AUC 값의 범위는 0~1임/예측이 100% 잘못된 모델의 AUC는 0.0/예측이 100% 정확한 모델의 AUC는 1.0임 
  - 다음의 두 가지 이유로 이상적임 
    - AUC는 척도 불변임/AUC는 절대값이 아니라 예측이 얼마나 잘 평가되는지 측정함 
    - AUC는 분류 임계값 불변임/AUC는 어떤 분류 임계값이 선택되었는지와 상관없이 모델의 예측 품질을 측정함 
  - 하지만 이런 두 이유는 특정 사용 사례에서 AUC의 유용성을 제한 할 수 있음 
    - 척도 불변이 항상 이상적인 것은 아님/잘 보정된 확률 결과가 필요한 경우가 있는데 AUC로는 이 정보를 알 수 없음 
    - 분류 임계값 불변이 항상 이상적인 것은 아님
      - 허위 음성(FN)비용과 허위 양성(FP)비용에 큰 차이가 있는 경우 한 가지 유형의 분류 오류를 최소화하는 것은 위험할 수 있음 
      - 예를 들어 이메일 스팸 감지를 실행할 때 허위 양성(FP)의 최소화로 인해 허위 음성(FN)이 크게 증가한다고 해도 허위 양성(FP)최소화를 우선시하고 싶을 수 있음 
      - AUC는 이런 유형의 최적화에 유용한 측정항목이 아님 

## 분류:예측 편향 
- 로지스틱 회귀 예측은 편향되지 않아야함/'예측평균'≈'관찰평균'
- 예측 편향은 두 평균이 서로 얼마나 멀리 떨어져 있는지 측정하는 수량임 
- 예측 편향 = 예측 평균 - 이 데이터 세트의 라벨 평균 
- 0이 아닌 유의미한 예측 편향은 모델 어딘가에 버그가 있다는 의미임/양성인 라벨이 발생하는 빈도에 있어서 모델이 잘못되었음을 보여줌 
- 예를 들어 전체 이메일의 1%가 스팸이라고 할 때 어떤 이메일의 대해 전혀 모르는 경우 메일의 1%는 스팸일 수 있다고 예측해야함
- 평균적으로 1%가 스팸일 수 있다고 예측해야함/각 이메일이 스팸으로 예측될 가능성의 평균을 내면 결과가 1%여야함 
- 모델이 스팸 가능성을 평균 20%로 예측한다면 예측 편향을 드러내는 것으로 결론을 내릴 수 있음 
- 예측 편향이 나타날 수 있는 근본 원인
  - 불완전한 특성 세트
  - 노이즈 데이터 세트
  - 결함이 있는 파이프라인
  - 편향된 학습 샘플
  - 지나치게 강한 정규화
- 예측 편향을 줄이기 위해 모델의 결과를 조정하는 캘리브레이션 레이어를 추가하여 학습한 모델을 사후 처리하는 방법으로 예측 편향을 수정하고자 할 수 있음 
- 예를 들어 모델의 편향이 +3%인 경우 평균 예측을 3% 낮추는 캘리브레이션 레이어를 추가할 수 있음
- 하지만 캘리브레이션 레이어 추가는 다음과 같은 이유로 바람직하지 않음 
  - 원인이 아니라 증상만을 수정함 
  - 최신 상태로 유지하기 어려운 불안정한 시스템이 구축됨 
- 가능하면 사용하지 않는 것이 좋음/캘리브레이션 레이어를 사용하므로써 이에 대한 의존도가 높아지는 경향이 있음/유지관리문제로 어려움을 겪을 수 있음 
- 적절한 모델의 편향은 대개 0에 가까움/예측 편향이 낮다고 해서 모델이 적합하다는 것을 증명하는 것은 아님 

### 바케팅 및 예측 편향
- 로지스틱 회귀는 0과 1사이의 값을 예측함/하지만 라벨이 있는 모든 예제는 정확히 0또는 정확히 1임
- 예측 편향을 검사할 때 예제 하나만을 토대로 해서는 예측을 정확하게 판단할 수 없음 
- 예제의 버킷을 대상으로 예측 편향을 검사해야함 
- 로지스틱 회귀의 예측 편향은 예제들을 충분히 모아서 예측된 값을 관찰된 값과 비교할 수 있는 경우에만 의미가 있음 
- 버킷 구성
  - 타겟 예측을 선형으로 분류
  - 분위 형성
- 특정 모델의 캘리브레이션 플롯을 가정해 봄/각 점은 1000개의 버킷을 나타냄
  - x축은 모델이 해당 버킷을 예측한 값의 평균을 나타냄 
  - y축은 해당 버킷의 데이터 세트에 있는 값의 실제 평균을 나타냄 
  - 두 축은 대수 척도임 
  <img src="https://user-images.githubusercontent.com/32586985/70840210-211f3180-1e54-11ea-93b4-53a189a545f0.PNG">
  
  - 모델의 일부에서만 예측이 저조한 이유는?
    - 학습 세트가 데이터 공간의 특정 하위 집합을 충분히 대표하지 않음 
    - 데이터 세트의 일부 하위 집합이 다른 하위 집합보다 노이즈가 많음 
    - 모델이 지나치게 정규화되어있음(람다 값을 줄여보자)


## 프로그래밍 실습 
- 로지스틱 회귀
- 특정 지역의 거주 비용이 높은지 여부를 예측하는 이진 분류 문제로 바꿈
- 이진 분류 문제로 전환
  - 데이터 세트의 타겟은 숫자(연속 값)특성인 median_house_value임/이 연속 값에 임계값을 적용하여 부울 라벨을 만들 수 있음 
  - 특정 지역을 나타내는 특성이 주어질 때 거주 비용이 높은 지역인지를 예측하려고 함 
  - 분류 임계값을 주택 가격 중앙값에 대한 75번째 백분위수(약 265,000)로 정의함 
  - 주택 가격이 임계값보다 높으면 라벨이 1로, 그렇지 않으면 0으로 지정됨 
```python
   from __future__ import print_function
   
   import math
   
   from IPython import display
   from matplotlib import cm
   from matplotlib import gridspec
   from matplotlib import pyplot as plt
   import numpy as np
   import pandas as pd
   from sklearn import metrics
   #tensorflow_version 1.x
   import tensorflow tf
   from tensorflow.python.data import Dataset
   
   tf.logging.set_verbosity(tf.logging.ERROR)
   pd.options.display.max_rows = 10
   pd.options.display.float_format = '{:.1f}'.format
   
   california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
   
   california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
   
   
   def preprocess_features(california_housing_dataframe):
     """Prepares input features from California housing data set.
     
     Args:
       california_housing_dataframe: A Pandas DataFrame expected to contain data
         from the California housing data set.
     Returns:
       A DataFrame that contains the features to be used for the model, including
       synthetic features.
     """
     selected_features = california_housing_dataframe[
       ["latitude",
        "longitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"]]
     processed_features = selected_features.copy()
     # Create a synthetic feature.
     processed_features["rooms_per_person"] = (
       california_housing_dataframe["total_rooms"] /
       california_housing_dataframe["population"])
     return processed_features
     
   def preprocess_targets(california_housing_dataframe):
     """Prepares target features (i.e., labels) from California housing data set.
     
     Args:
       california_housing_dataframe: A Pandas DataFrame expected to contain data
         from the California housing data set.
     Returns:
       A DataFrame that contains the target feature.
     """
     output_targets = pd.DataFrame()
     # Create a boolean categorical feature representing whether the 
     # median_house_value is above a set threshold.
     output_targets["median_house_value_is_high"] = (
       california_housing_dataframe["median_house_value"] > 265000).astype(float)
     return output_targets
     
     
   # Choose the first 12000 (out of 17000) examples for training.
   training_examples = preprocess_features(california_housing_dataframe.head(12000))
   training_targets = preprocess_targets(california_housing_dataframe.head(12000))
   
   # Choose the last 5000 (out of 17000) examples for validation.
   validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
   validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
   
   # Double-check that we've done the right thing.
   print("Training examples summary:")
   display.display(training_examples.describe())
   print("Validation examples summary:")
   display.display(validation_examples.describe())
   
   print("Training targets summary:")
   display.display(training_targets.describe())
   print("Validation targets summary:")
   display.display(validation_targets.describe())
   
   
   # 데이터 로드 및 특성 및 타겟 준비
```
- 선형 회귀의 성능 측정
- 로지스틱 회귀가 효과적인 이유를 확인하기 위해, 선형 회귀를 사용하는 단순 모델을 학습시켜 봄 
- 이 모델에서는 {0,1} 집합에 속하는 값을 갖는 라벨을 사용하며 0 또는 1에 최대한 가까운 연속 값을 예측하려고 시도함 
- 출력을 확률로 해석하려고 하므로 (0,1)범위 내에서 출력되는 것이 이상적임/그런 다음 임계값 0.5를 적용하여 라벨을 결정함 
```python
   def construct_feature_columns(input_features):
     """Construct the TensorFlow Feature Columns.
     
     Args:
       input_features: The names of the numerical input features to use.
     Returns:
       A set of feature columns
     """
     return set([tf.feature_column.numeric_column(my_feature)
                 for my_feature in input_features])
   
   def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
       """Trains a linear regression model.
       
       Args:
         features: pandas DataFrame of features
         targets: pandas DataFrame of targets
         batch_size: Size of batches to be passed to the model
         shuffle: True or False. Whether to shuffle the data.
         num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
       Returns:
         Tuple of (features, labels) for next data batch
       """
       
       # Convert pandas data into a dict of np arrays.
       features = {key:np.array(value) for key,value in dict(features).items()}
       
       # Construct a dataset, and configure batching /repeating.
       ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
       ds = ds.batch(batch_size).repeat(num_epochs)
       
       # Shuffle the data, if specified.
       if shuffle:
         ds = ds.shuffle(10000)
       
       # Return the next batch of data.
       features, labels = ds.make_one_shot_iterator().get_next()
       return features, labels
   
   def train_linear_regressor_model(
       learning_rate,
       steps,
       batch_size,
       training_examples,
       training_targets,
       validation_examples,
       validation_targets):
     """Trains a linear regression model.
     
     In addition to training, this function also prints training progress information,
     as well as a plot of the training and validation loss over time.
     
     Args:
       learning_rate: A 'float', the learning rate.
       steps: A non-zero 'int', the total number of training steps. A training step
         consists of a forward and backward pass using a single batch.
       batch_size: A non-zero 'int', the batch size.
       training_examples: A 'DataFrame' containing one or more columns from
         'california_housing_dataframe' to use as input features for training.
       training_targets: A 'DataFrame' containing exactly one column from
         'california_housing_dataframe' to use as target for training.
       validation_examples: A 'DataFrame' containing one or more columns from
         'california_housing_dataframe' to use as input features for validation.
       validation_targets: A 'DataFrame' containing exactly one column from 
         'california_housing_dataframe' to use as target for validation.
         
     Returns:
       A 'LinearRegressor' object trained on the training data.
     """
     
     periods = 10
     steps_per_period = steps / periods 
     
     # Create a linear regressor object.
     my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
     my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
     linear_regressor = tf.estimator.LinearRegressor(
         feature_columns=construct_feature_columns(training_examples),
         optimizer=my_optimizer
     )
     
     # Create input functions.
     training_input_fn = lambda: my_input_fn(training_examples,
                                             training_targets["median_house_value_is_high"],
                                             batch_size=batch_size)
     predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                     training_targets["median_house_value_is_high"],
                                                     num_epochs=1,
                                                     shuffle=False)
     predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                       validation_targets["median_house_value_is_high"],
                                                       num_epochs=1,
                                                       shuffle=False)
     
     # Train the model, but do so inside a loop so that we can periodically assess
     # loss metrics.
     print("Training model...")
     print("RMSE (on training data):")
     training_rmse = []
     validation_rmse = []
     for period in range(0, periods):
       # Train the model, starting from the prior state.
       linear_regressor.train(
           input_fn=training_input_fn,
           steps=steps_per_period
       )
       
       # Take a break and compute predictions.
       training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
       training_predictions = np.array([item['predictions'][0] for item in training_predictions])
       
       validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
       validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
       
       # Compute training and validation loss.
       training_root_mean_squared_error = math.sqrt(
           metrics.mean_squared_error(training_predictions, training_targets))
       validation_root_mean_squared_error = math.sqrt(
           metrics.mean_squared_error(validation_predictions, validation_targets))
       # Occasionally print the current loss.
       print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
       # Add the loss metrics from this period to our list.
       training_rmse.append(training_root_mean_squared_error)
       validation_rmse.append(validation_root_mean_squared_error)
     print("Model training finished")
     
     # Output a graph of loss metrics over periods.
     plt.ylabel("RMSE")
     plt.xlabel("Periods")
     plt.title("Root Mean Squared Error vs. Periods")
     plt.tight_layout()
     plt.plot(training_rmse, label="training")
     plt.plot(validation_rmse, label="validation")
     plt.legend()
     
     return linear_regressor
     
     linear_regressor = train_linear_regressor_model(
         learning_rate=0.000001,
         steps=200,
         batch_size=20,
         training_examples=training_examples,
         training_targets=training_targets,
         validation_examples=validation_examples,
         validation_targets=validation_targets)       
```
<img src="https://user-images.githubusercontent.com/32586985/70842347-dc9c9180-1e65-11ea-9f1c-6620cff7bda6.PNG">


### 작업1:예측의 LogLoss 계산 가능성 확인 
- LogLoss는 이러한 '신뢰 오차'에 훨씬 큰 페널티를 부여함
- LogLoss의 정의는 다음과 같음 
<img src="https://user-images.githubusercontent.com/32586985/70842956-c72b6580-1e6d-11ea-931b-c93de1108e1d.PNG">

- 예측 값을 가저오는것이 먼저임/예측과 타겟이 주어지면 LogLoss를 계산할 수 있는지 확인해봄 
```python
   predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                     validation_targets["median_house_value_is_high"],
                                                     num_epochs=1,
                                                     shuffle=False)
   validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
   validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
   
   _ = plt.hist(validation_predictions)
```
<img src="https://user-images.githubusercontent.com/32586985/70842997-3e60f980-1e6e-11ea-9237-b7da0fa3915f.PNG">


### 작업2:로지스틱 회귀 모델을 학습시키고 검증세트로 LogLoss 계산
- 로지스틱 회귀를 사용하려면 LinearRegressor 대신 LinearClassifier를 사용함 
```python
   def train_linear_classifier_model(
       learning_rate,
       steps,
       batch_size,
       training_examples,
       training_targets,
       validation_examples,
       validation_targets):
     """Trains a linear classification model.
     
     In addition to training, this function also prints training progress information,
     as well as a plot of the training and validation loss over time.
     
     Args:
       learning_rate: A 'float', the learning rate.
       steps: A non-zero 'int', the total number of training steps. A training step
         consists of a forward and backward pass using a single batch.
       batch_size: A non-zero 'int', the batch size.
       training_examples: A 'DataFrame' containing one or more columns from
         'california_housing_dataframe' to use as input features for training.
       training_targets: A 'DataFrame' containing exactly one column from
         'california_housing_dataframe' to use as target for training.
       validation_examples: A 'DataFrame' containing one or more columns from 
         'california_housing_dataframe' to use as input features for validation.
       validation_targets: A 'DataFrame' containing exactly one column from
         'california_housing_dataframe' to use as target for validation.
         
     Returns:
       A 'LinearClassifier' object trained on the training data.
     """
     
     periods = 10
     steps_per_period = steps / periods
     
     # Create a linear classifier object.
     my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
     my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
     linear_classifier = tf.estimator.LinearClassifier(
         feature_columns=construct_feature_columns(training_examples),
         optimizer=my_optimizer
     )
     
     # Create input functions.
     training_input_fn = lambda: my_input_fn(training_examples,
                                             training_targets["median_house_value_is_high"]
                                             batch_size=batch_size)
     predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                     training_targets["median_house_value_is_high"]
                                                     num_epochs=1,
                                                     shuffle=False)
     predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                       validation_targets["median_house_value_is_high"],
                                                       num_epochs=1,
                                                       shuffle=False)
     
     # Train the model, but do so inside a loop so that we can periodically assess
     # loss metrics.
     print("Training model...")
     print("LogLoss (on training data):")
     training_log_losses = []
     validation_log_losses = []
     for period in range (0, periods):
       # Train the model, starting from the prior state.
       linear_calssifier.train(
           input_fn=training_input_fn,
           steps=steps_per_period
       )
       # Take a break and compute predictions.
       training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
       training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
       
       validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
       validation_probabilities = np.array([item['probabilities] for item in validation_probabilities])
       
       training_log_loss = metrics.log_loss(training_targets, training_probabilities)
       validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
       # Occasionally print the current loss.
       print("  period %02d: %0.2f" % (period, training_log_loss))
       # Add the loss metrics from this period to our list.
       training_log_losses.append(training_log_loss)
       validation_log_losses.append(validation_log_loss)
     print("Model training finished.")
     
     # Output a graph of loss metrics over periods.
     plt.ylabel("LogLoss")
     plt.xlabel("Periods")
     plt.title("LogLoss vs. Periods")
     plt.tight_layout()
     plt.plot(training_log_losses, label="training")
     plt.plot(validation_log_losses, label="validation")
     plt.legend()
     
     return linear_classifier
     
     linear_classifier = train_linear_calssifier_model(
         learning_rate=0.000005
         steps=500,
         batch_size=20,
         training_examples=training_examples,
         training_targets=training_targets,
         validation_examples=validation_examples,
         validation_targets=validation_targets)         
```
<img src="https://user-images.githubusercontent.com/32586985/70843246-81bd6700-1e72-11ea-9627-b68481628a27.PNG">


### 작업3:검증 세트로 정확성 계산 및 ROC 곡선 도식화
- 분류에 유용한 몇 가지 측정항목은 모델 정확성,ROC곡선 및 AUC(ROC 곡선 아래 영역)이며 이러한 측정항목을 조사해 봄
- 과적합이 나타나지 않느 범위 내에서 더 오랫동안 학습하는 것/단계 수, 배치 크기 또는 둘 모두를 늘리면 됨 
- 모든 측정항목이 동시에 개선되므로 손실 측정항목은 AUC와 정확성 모두를 적절히 대변함 
- AUC를 몇 단위만 개선하려 해도 굉장히 많은 추가 반복이 필요함
```python
   linear_classifier = train_linear_classifier_model(
       learning_rate=0.000003,
       steps=20000,
       batch_size=500,
       training_examples=training_examples,
       training_targets=training_targets,
       validation_examples=validation_examples,
       validation_targets=validation_targets)
   
   evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
   
   print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
   print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
```
<img src="https://user-images.githubusercontent.com/32586985/70843337-df9e7e80-1e73-11ea-9d7f-aa1cd64835e7.PNG">

