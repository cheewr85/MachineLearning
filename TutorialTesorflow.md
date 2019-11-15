## 텐서플로우 첫걸음 
- LinearRegressor 클래스를 사용하여 입력 특성 하나를 기반으로 지역별 주택 가격 중앙값을 예측함
- 평균 제곱급 오차(RMSE)를 사용하여 모델 예측의 정확성을 평가함
- 초매개변수를 조정하여 모델의 정확성을 개선함
- 데이터의 출처는 1990년 캘리포니아 인구조사 자료


## 설정 
- 필요한 라이브러리 로드
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
```

- 데이터 세트를 로드함
```python
   california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
```

- 확률적 경사하강법의 성능에 악영향을 줄 수 있는 의도치 않은 정렬 효과를 방지하고자 데이터를 무작위로 추출하겠음
- 일반적으로 사용하는 학습률 범위에서 보다 쉽게 학습할 수 있도록 median_house_value를 천 단위로 조정함
<img src="https://user-images.githubusercontent.com/32586985/68983035-86c7d000-084c-11ea-9a58-71e354f1530d.PNG">



## 데이터 조사
- 본격적으로 다루기 전에 살펴볼 내용임
- 각 열에 대해 예의 개수, 평균, 표준편차, 최대값, 최소값, 다양한 분위 등 몇 가지 유용한 통계를 간단히 요약하여 출력한 값
<img src="https://user-images.githubusercontent.com/32586985/68983098-d0181f80-084c-11ea-84b7-a049f1216f9f.PNG">


## 첫 번째 모델 만들기
