## 표현 
- 데이터의 표현을 만들어 모델이 데이터의 핵심적인 특징을 들여다 볼 수 있는 유용한 관측지점 제공
- 데이터를 가장 잘 표현하는 특성 세트를 선택해야함

## 특성 추출 
### 원시 데이터를 특성에 매핑 
- 특성 추출이란 원시 데이터를 특성 벡터로 변환하는 과정/상당한 시간이 소요
- 특성 값에 모델 가중치를 곱해야 하므로 실수 벡터로 가중치를 표현함 
- 그림 1. 특성 추출을 통해 원시 데이터를 ML 특성에 매핑 
<img src="https://user-images.githubusercontent.com/32586985/69487267-0df2f480-0e9a-11ea-82ff-6e3b34895efb.PNG">

- 왼쪽 부분은 입력 데이터 소스의 원시 데이터
- 오른쪽 부분은 특성벡터, 데이터 세트의 예로 구성된 부동 소수점 값의 집합 

### 숫자 값 매핑 
- 정수와 부동 소수점 데이터에는 숫자 가중치를 곱할 수 있으므로 특수한 인코딩은 필요하지 않음 
- 그림 2. 정수 값을 부동 소수점 값에 매핑/원시 정수 값 6을 특성 값 6.0으로 변환하는 것은 큰 의미가 없음  
<img src="https://user-images.githubusercontent.com/32586985/69487276-59a59e00-0e9a-11ea-9789-c24d84992f61.PNG">

### 범주 값 매핑 
- 범주형 특성은 가능한 값의 이산 집합을 가짐/가능한 값의 불연속 집합을 갖는 특성 
```python
   {'Charleston Road', 'North Shoreline Boulevard', 'Shorebird Way', 'Rengstorff Avenue'}
   # street_name 이라는 특성이 있음 
```
- 모델은 학습된 가중치로 문자열을 곱할 수 없으므로 특성 추출을 통해 문자열을 숫자값으로 변환함 
- 가능한 값의 어휘로 지칭할 특성 값에서 정수로의 매핑을 정의함으로써 가능함 
- 다른 모든 거리를 포괄적인 '기타'범주(OOV(out-of-vocabulary)버킷이라고도 함)로 그룹화 할 수 있음 
- 다음과 같이 거리 이름을 숫자로 매핑 가능 
  - Charleston Road를 0으로 매핑
  - North Shoreline Boulevard를 1로 매핑 
  - Shorebird Way를 2로 매핑 
  - Rengstorff Avenue를 3으로 매핑 
  - 기타(OOV)를 모두 4로 매핑 
- 하지만 이러한 색인 번호를 모델에 직접 통합하면 몇 가지 제약으로 인한 문제가 생김 
  - 모든 거리에 적용되는 하나의 가중치를 학습/street_name에 적용되는 가중치를 각 곱함 
  - street_name을 특성으로 사용하여 주택 가격을 예측하는 모델이 있다고 가정 
  - 거리 이름을 바탕으로 가격을 선형 보정할 가능성은 없어 평균 주택 가격을 바탕으로 거리 순서를 결정했다고 가정 
  - 모델을 거리별로 적용되는 각 가중치를 유연하게 학습할 수 있어야 하며, 다른 특성을 사용하여 추정한 가격에 더해짐 
  - street_name이 여러 값을 가지는 사례는 고려하지 않음 
  - 두 거리가 만나는 모퉁이에 주택이 여러 채 있고 street_name값의 정보에 포함된 색인이 하나인 경우 가격정보 인코딩할 수 없음 
- 이러한 제약조건을 모두 제거하기 위해,대신 모델의 범주형 특성에 바이너리 벡터를 생성할 수 있음 
  - 예시에 적용되는 값의 경우 관련 벡터 요소를 1로 설정
  - 다른 요소는 모두 0으로 설정 
- 이 벡터의 길이는 어휘에 있는 요소의 수와 같음 
  - 이러한 표현은 단일 값이 1일때 원-핫 인코딩 이라함 
  - 여러 값이 1일 때 멀티-핫 인코딩 이라함 
- 그림 3. 원-핫 인코딩을 통한 거리 주소 매핑/Shorebird Way의 바이너리 벡터에 있는 요소의 값은 1이고 다른 모든 거리의 요소 값은 0임 
<img src="https://user-images.githubusercontent.com/32586985/69487392-5f9c7e80-0e9c-11ea-961e-dc7a845890ae.PNG">

- 이러한 방식을 통해 모든 특성 값에 대한 부울 변수를 효과적으로 만들 수 있음 
- 위에 예시에서는 집이 Shorebird Way에 있는 경우 Shorebird Way에만 해당하는 바이너리 값이 1임/모델은 Shorebird Way의 가중치만 사용
- 이와 유사하게 집이 두 거리가 만나는 모퉁이에 위치한 경우 두 바이너리 값은 1로 설정되며 모델은 각각의 가중치를 모두 사용 
- 원-핫 인코딩은 우편번호와 같이 가중치를 바로 곱하고 싶지 않은 숫자 데이터까지 확장됨 

### 희소 표현 
- 데이터세트에 street_name의 값으로 포함하고 싶은 거리 이름이 백만 개가 있다고 가정할 시 
- 1,2개 요소만 true인 요소 백만 개의 바이너리 벡터를 명시적으로 만드는 것은 매우 비효율적인 표현임 
- 이런 상황에서는 0이 아닌 값만 저장되는 희소 표현을 사용하는 것임/각 특성 값에 독립적인 모델 가중치가 학습됨 


## 좋은 특성의 조건 
- 어떠한 종류의 값이 실제로 좋은 특성이 되는가
### 거의 사용되지 않는 불연속 특성 값 배제 
- 좋은 특성 값은 데이터 세트에서 5회 이상 나타나야함 
- 모델에서 라벨과의 관계를 학습하기가 쉬움
- 동일한 이산 값을 갖는 예가 많으면 모델에서 다양한 설정으로 특성을 확인하여 라벨을 예측하는 좋은 지표인지를 판단할 수 있음 
```python
   house_type: victorian
   # 올바른 예시
```
- 특성의 값이 한 번만 나타나거나 매우 드물게 나타난다면 모델에서 해당 특성을 기반으로 예측할 수 없음 
```python
   unique_house_id: 8SK982ZZ1242Z
   # 각 값이 한 번만 사용되어 모델에서 어떠한 학습도 불가능하므로 좋은 특성이 아님 
```

### 가급적 분명하고 명확한 의미 부여 
- 각 특성은 명확하고 분명한 의미를 가져야함 
```python
   house_age: 27
   # 주택의 연령을 연 단위로 나타내는 것을 바로 알아볼 수 잇는 좋은 특성 
```
```python
   house_age: 851472000
   # 특성을 만든 엔지니어 이외에 다른 사람은 알아보기 어려움 
```
- 데이터 노이즈로 인해 값이 불명확해지는 경우도 있음 
```python
   user_age: 277
   # 적절한 값을 확인하지 않은 소스에서 비롯됨 
```

### '특수'값을 실제 데이터와 혼용하지 말 것 
- 좋은 부동 소수점 특성은 특이한 범위 외 불연속성 또는 '특수'값을 포함하지 않음 
```python
   quality_rating: 0.82
   quality_rating: 0.37
   # 어떤 특성이 0~1 범위의 부동 소수점 값을 갖는다고 가정해보면 다음과 같은 값은 문제가 없음 
```
```python
   quality_rating: -1
   # 사용자가 quality_rating을 입력하지 않은 경우 데이터 세트에서 다음과 같은 특수 값으로 데이터가 없음을 표현했을 수 있음 
```
- 특수 값이 나타나지 않게 하려면 특성을 다음과 같은 두 특성으로 변환 
  - 특성 하나는 특수 값 없이 오로지 품질 등급만 낮음 
  - 특성 하나는 quality_rating이 입력되었는지 여부를 나타내는 부울 값을 가짐
    - 이 부울 특성에 is_quality_rating_defined와 같은 이름을 지정 
    
### 업스트림 불안정성 고려
- 특성의 정의는 시간이 지나도 변하지 않아야함 
```python
   city_id: "br/sao_paulo"
   # 도시의 이름은 일반적으로 바뀌지 않으므로 다음 값은 유용함 
   # br/sao_paulo와 같은 문자열을 원-핫 벡터로 변환할 필요는 있음 
```
- 다른 모델에서 추론한 값을 수집할 때는 또 다른 문제점이 있음 
```python
   inferred_city_cluster: "219"
   # 219라는 값이 현재 상파울루를 나타내고 있더라도, 다른 모델을 이후에 실행할 때는 이 표현이 변경될 수 있음 
```


## 데이터 정제 
### 특성 값 조정 
- 조정이란 부동 소수점 특성 값을 100~900 등의 자연 범위에서 0~1 또는 -1~+1등의 표준 범위로 변환하는 작업임 
- 특성 세트가 단일 특성으로만 구성된 경우 조정에 따르는 실질적인 이점은 거의 없음 
- 특성 세트가 여러 특성으로 구성되었다면 특성 조정으로 다음과 같은 이점을 누림 
  - 경사하강법이 더 빠르게 수렴됨 
  - 'NaN 트랩'이 방지됨/NaN 트랩이란 모델의 숫자 중 하나가 NaN이 된 후 수학연산 과정에서 모델의 다른 모든 숫자가 결국 NaN이 되는 상황 
  - 모델이 각 특성의 적절한 가중치를 익히는 데 도움이 됨/특성 조정을 수행하지 않으면 모델에서 범위가 더 넓은 특성을 과도하게 중시함 
- 모든 부동 소수점 특성에 동일한 척도를 부여할 필요는 없음 
- 특성 A는 -1~+1로, 특성 B는 -3~+3으로 조정해도 심각한 부작용은 없음 
- 그러나 특성 B를 5000~100000으로 조정하면 모델이 부정적으로 반응할 것임 
- 조정
  - 숫자 데이터를 조정하는 알기 쉬운 방법 중 하나는 [최소값,최대값]을 [-1,+1]등의 작은 척도로 선형 매핑하는 것임 
  - 각 값의 Z 점수를 계산하는 조정 방식도 널리 사용됨/Z 점수는 표준편차를 기준으로 평균에서 벗어난 정도를 계산함 
    - scaledvalue = (value-mean)/stddev.
    - 평균=100/표준편차=20/원래 값=130
    - scaled_value = (130 - 100) / 20
    - scaled_value = 1.5
    - Z 점수로 조정하면 대부분의 조정 값이 -3~+3 범위에 놓이지만, 이 범위보다 약간 높거나 낮은 값도 다소 존재함 
    
### 극단적 이상점 처리 
- 다음 예시는 캘리포니아 주택 데이터 세트에서 얻은 roomsPerPerson이라는 특성을 나타낸 플롯
- roomsPerPerson 값은 지역의 전체 방 수를 해당 지역의 인구로 나누어 계산함 
- 플롯을 보면 캘리포니아의 대부분 지역은 1인당 1~2개의 방을 갖추고 있음
- 그림 4. 매우 긴 꼬리
<img src="https://user-images.githubusercontent.com/32586985/69487718-0d5e5c00-0ea2-11ea-8f1e-875b1778565e.PNG">

- 모든 값의 로그를 취하여 극단적 이상점 영향을 조정해봄 
- 그림 5. 로그 조정을 거쳐도 꼬리가 남아 있음 
<img src="https://user-images.githubusercontent.com/32586985/69487728-3aab0a00-0ea2-11ea-8981-13fd4ab69c4e.PNG">

- 여전히 꼬리가 상당히 남아있음 
- 다른 접근법: roomsPerPerson의 최대값을 4.0 같은 임의의 지점에서 잘라내어 제한을 둠 
- 그림 6. 4.0에서 특성 값 잘라내기
<img src="https://user-images.githubusercontent.com/32586985/69487742-64fcc780-0ea2-11ea-9366-597f330d42cf.PNG">

- 특성 값을 4.0에서 잘라낸다는 말은 4.0보다 큰 값을 모두 무시한다는 의미가 아님
- 4.0보다 큰 값을 모두 4.0으로 인식하겠다는 의미/4.0 지점에서 부자연스러운 경사가 생김 
- 하지만 조정된 특성 세트는 원래 데이터보다 훨씬 유용해진 상태임 

### 비닝 
- 위도에 따른 상대적인 주택 분포를 보여주는 플롯
- 로스엔젤레스의 위도는 약 34도이고 샌프란시스코의 위도는 약 38도임 
- 그림 7. 위도별 주택 수 
<img src="https://user-images.githubusercontent.com/32586985/69487758-c755c800-0ea2-11ea-8a08-479deee30f5a.PNG">

- 데이터 세트에서 latitude는 부동 소수점 값임 
- 이 모델에서는 표현할 수 없음/위도와 주택 값 사이에 선형적 관계가 없기 때문임
- 예를 들면 위도 35도에 위치한 주택이 위도 34도에 위치한 주택보다 35/34만큼 싸거나 비싸지는 않음 
- 각각의 위도 값은 주택 가격을 예측하는 좋은 지표일 가능성이 높음 
- 위도를 여러 '빈(bin)'으로 나눔 
- 그림 8. 값 비닝 
<img src="https://user-images.githubusercontent.com/32586985/69487802-611d7500-0ea3-11ea-8fe5-5e626941f155.PNG">

- 부동 소수점 특성 하나가 아니라 11개의 개별 부울 특성(LatitudeBin1,LatitudeBin2,...,LatitudeBin11)이 생김 
- 11개의 개별 특성은 다소 번잡하므로 하나의 11원소 벡터로 통일해서 보임 
- 위도 37.4도를 다음과 같이 표현할 수 있음 
```python
   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] 
```
- 비닝을 사용하면 모델에서 각 위도에 대해 완전히 다른 가중치를 익힐 수 있음 
- 위의 예제는 빈 경계로 정수를 사용했지만 해상도를 더 높여야 한다면 빈 경계를 1/10도 간격으로 나눌 수 있음 
- 빈을 더 추가하면 모델에서 위도 37.4도와 37.5도에 대해 서로 다른 동작을 익힐 수 있지만 1/10도 간격으로 충분한 수의 예가 확보되어야 한다는 전제가 붙음       
- 분위를 기준으로 빈을 나누면 각 버킷에 같은 개수의 예가 포함됨/분위별 비닝 사용시 이상점에 대해 전혀 신경 쓸 필요가 없어짐 
  - 분위란? 확률 분포를 동등한 확률 구간으로 나누는 구분 눈금들
 
### 스크러빙   
- 학습 및 데이터에 사용되는 모든 데이터를 신뢰할 수 있다고 가정을 하였지만 실제로는 다음 같은 이유로 그렇지 못함 
  - 값 누락. 예를 들어 사용자가 주택의 연령을 실수로 입력하지 않았을 수 있음 
  - 중복 예. 예를 들어 서버에서 같은 로그를 실수로 두 번 업로드했을 수 있음 
  - 잘못된 라벨. 예를 들어 사용자가 참나무 사진에 실수로 단풍나무 라벨을 지정했을 수 있음 
  - 잘못된 특성 값. 예를 들어 사용자가 숫자를 실수로 입력했거나 온도계를 햇빛에 두었을 수 있음 
- 잘못된 예가 발견되면 일반적으로 데이터 세트에서 삭제하여 해당 예를 수정함 
- 값 누락이나 중복 예를 탐지하고자 간단한 프로그램을 작성할 수 있음/잘못된 특성 값 또는 라벨을 탐지하기는 훨씬 더 까다로울 수 있음 
- 잘못된 개별 예를 탐지하는 것 외에 집계에서도 잘못된 데이터를 탐지해야함
- 히스토그램은 집계 데이터를 시각화하는 유용한 메커니즘임 
- 다음과 같은 통계를 구하면 도움이 될 수 있음 
  - 최대 및 최소
  - 평균 및 중앙값
  - 표준편차
- 불연속 특성에서 가장 자주 나타나는 값으로 목록을 생성해보자

### 철저한 데이터 파악 
- 정상적인 데이터가 어떠한 모습이어야 하는지 항상 생각
  - 시각화: 출현 빈도 순으로 히스토그램을 그림 
- 데이터가 이러한 예상과 일치하는지 확인하고, 그렇지 않다면 그 이유를 파악 
  - 디버그: 중복된 예가 있나? 누락된 값이 있나? 이상점이 있나? 데이터가 대시보드와 일치하는가? 학습 데이터와 검증 데이터가 서로 비슷한가?
- 학습 데이터가 대시보드 등의 다른 소스와 일치하는지 재차 확인함 
  - 모니터링: 특성 분위 및 시간에 따른 예의 수 


### 특성 세트 프로그래밍 실습 
- 복잡한 특성 세트만큼 좋은 성능을 발휘하는 최소한의 특성 세트를 만듬 
- 모델에 포함된 특성이 적을수록 리소스 사용이 감소하며 유지보수도 쉬워짐 
- 주택 관련 특성을 최소한으로 사용하면서 데이터 세트의 모든 특성을 사용하는 모델과 동등한 성능을 발휘하는 모델을 만들어봄 

- 설정 
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
   %tensorflow_version 1.x
   import tensorflow as tf
   from tensorflow.python.data import Dataset
   
   tf.logging.set_verbosity(tf.logging.ERROR)
   pd.options.display.max_rows = 10
   pd.options.display.float_format = '{:.1f}'.format
   
   california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
   
   california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
   
   def preprocess_features(california_housing_dataframe):
     """Prepares input features from California housing data set.
     
     Args:
       california_housing_dataframe: A Pandas DataFrame expected tro contain data
         from the California housing data set.
     Returns:
       A DataFrame that contains the features to be used for the model, including 
       synthetic features.
     """
     selected_features = california_housing_dataframe[
       ["latitude",
        "longtitude",
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
     # Scale the target to be in units of thousands of dollars.
     output_targets["median_house_value"] = (
       california_housing_dataframe["median_house_value"] / 1000.0)
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
```
<img src="https://user-images.githubusercontent.com/32586985/69488182-4221e180-0ea9-11ea-8936-d664761577c4.PNG">
<img src="https://user-images.githubusercontent.com/32586985/69488183-4817c280-0ea9-11ea-9103-5e55bd47f617.PNG">

### 작업 1: 효율적인 특성 세트 개발 
- 특성을 2~3개만 사용하면서 성능을 어디까지 올릴 수 있을까
- 상관행렬은 각 특성을 타겟과 비교한 결과 및 각 특성을 서로 비교한 결과에 따라 쌍의 상관성을 보여줌 
- 상관성을 피어슨 상관계수로 정의함
   - 피어슨 상관계수는 두 변수 X와 Y간의 선형 상관 관계를 계량화한 수치임 
   - 상관성 값의 의미는 다음과 같음 
      - -1.0: 완벽한 음의 상관성 
      - 0.0: 상관성 없음 
      - 1.0: 완벽한 양의 상관성 
```python 
   correlation_dataframe = training_examples.copy()
   correlation_dataframe["target"] = training_targets["median_house_value"]
   
   correlation_dataframe.corr()
```
<img src="https://user-images.githubusercontent.com/32586985/69488247-2965fb80-0eaa-11ea-9560-d2ce91451240.PNG">

- 타겟과 상관성이 높은 특성을 찾아야함 
- 각 특성이 서로 독립적인 정보를 추가하도록 서로간의 상관성이 높지 않은 특성을 찾는 것이 좋음 
- 실습의 학습 코드
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
       
       # Construct a dataset, and configure batching/repeating.
       ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
       ds = ds.batch(batch_size).repeat(num_epochs)
       
       # Shuffle the data, if specified.
       if shuffle:
         ds = ds.shuffle(10000)
         
       # Return the next batch of data.
       features, labels = ds.make_one_shot_iterator().get_next()
       return features, labels
   def train_model(
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
         'california_housing-dataframe' to use as input features for training.
       training_targets: A 'DataFrame' containing exactly one columns from 
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
                                             training_targets["median_house_value"],
                                             batch_size=batch_size)
     predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                     training_targets["median_house_value"],
                                                     num_epochs=1,
                                                     shuffle=False)
     predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                       validation_targets["median_house_value"],
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
           steps=steps_per_period,
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
     print("Model training finished.")
     
     
     # Output a graph of loss metrics over periods.
     plt.ylabel("RMSE")
     plt.xlabel("Periods")
     plt.title("Root Mean Squared Error vs. Periods")
     plt.tight_layout()
     plt.plot(training_rmse, label="training")
     plt.plot(validation_rmse, label="validation")
     plt.legend()
     
     return linear_regressor
```
- 상관성을 다양한 데이터를 비교하여서 최적의 안을 아래의 값으로 찾아냄
```python
   minimal_features = [
     "median_income",
     "latitude",
   ]
   
   minimal_training_examples = training_examples[minimal_features]
   minimal_validation_examples = validation_examples[minimal_features]
   
   _ = train_model(
       learning_rate=0.01,
       steps=500,
       batch_size=5,
       training_examples=minimal_training_examples,
       training_targets=training_targets,
       validation_examples=minimal_validation_examples,
       validation_targets=validation_targets)
```
<img src="https://user-images.githubusercontent.com/32586985/69489245-1eb26300-0eb8-11ea-8653-cfacf0f28eba.PNG">


### 작업 2: 위도 활용 고도화
- latitude와 median_house_value로 그래프를 그리면 선형 관계가 없다는 점이 드러남 
- 로스앤젤레스 및 샌프란시스코에 해당하는 위치 부근에 마루가 나타남 
```python
   plt.scatter(training_examples["latitude"], training_targets["median_house_value"])
```
<img src="https://user-images.githubusercontent.com/32586985/69489337-caa87e00-0eb9-11ea-88b3-49eec4955a03.PNG">

- 위도를 더 잘 활용할 수 있는 합성 특성
- 공간을 10개의 버킷으로 나누어 latitude_32_to_33, latitude_33_to_34등의 특성을 만들고 latitude가 해당 버킷의 범위에 포함되면 1.0 값을 그렇지 않으면 0.0 값을 표시함 
- 상관행렬과 연관
```python
   LATITUDE_RANGES = zip(range(32, 44), range(33, 45))
   
   def select_and_transform_features(source_df):
     selected_examples = pd.DataFrame()
     selected_examples["median_income"] = source_df["median_income"]
     for r in LATITUDE_RANGES:
       selected_examples["latitude_%d_to_%d" % r] = source_df["latitude"].apply(
         lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
     return selected_examples
     
   selected_training_examples = select_and_transform_features(training_examples)
   selected_validation_examples = select_and_transform_features(validation_examples)
   
   _ = train_model(
       learning_rate=0.01,
       steps=500,
       batch_size=5,
       training_examples=selected_training_examples,
       training_targets=training_targets,
       validation_examples=selected_validation_examples,
       validation_targets=validation_targets)
```
<img src="https://user-images.githubusercontent.com/32586985/69489392-ea8c7180-0eba-11ea-8755-aaff2b421044.PNG">
