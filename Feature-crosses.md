## 특성 교차
- 방식의 이름/양식 [A X B]의 템플릿을 정의함
- 양식이 더 복잡할 수 있음 [A X B X C X D X E]
- A와 B가 빈과 같은 부울 특성인 경우 곱의 결과 범위가 매우 희소하게 나타날 수 있음 
- 예
  - 주택시장 가격 예측: [latitude X num_bedrooms]
  - 틱택토 예측: [pos1 x pos2 x ... x pos9]
### 특성 교차를 사용해야 하는 이유
- 선형 학습자는 대량의 데이터(예:Vowpal Wabbit, sofia-ml)에 맞게 적절히 확장됨 
- 특성 교차가 없으면 모델을 충분히 표현할 수 없음 
- 특성 교차와 대량의 데이터를 사용하면 매우 복잡한 모델을 효율적으로 학습할 수 있음 
  - 신경망을 사용할 수도 있음 


## 특성 교차:비선형성 인코딩 
- 파란색 점은 병든 나무를 나타냄
- 주황색 점은 건강한 나무를 나타냄 
- 선을 기준으로 나무의 상태를 적절히 예측할 수 있음 
<img src="https://user-images.githubusercontent.com/32586985/70212855-6ddf7a00-177b-11ea-8ac9-8bb7ca02be06.PNG">

- 병든 나무와 건강한 나무를 깔끔하게 구분하는 직선을 그릴 수 없음/어떤 선을 그려도 나무의 상태를 적절히 예측할 수 없음 
<img src="https://user-images.githubusercontent.com/32586985/70213052-cf9fe400-177b-11ea-8918-71ca935fae72.PNG">

- 비선형 문제 
<img src="https://user-images.githubusercontent.com/32586985/70212951-9d8e8200-177b-11ea-9c3f-a7b893e8aa6d.PNG">

- 그림 2에 표시된 비선형 문제를 해결하려면 특성 교차를 만들어야함 
- 특성 교차는 두 개 이상의 입력 특성을 곱하여 특성 공간에서 비선형성을 인코딩하는 합성 특성임 
- x1과 x2를 교차하여 x3이라는 특성 교차를 만듬 
<img src="https://user-images.githubusercontent.com/32586985/70213163-0ece3500-177c-11ea-96fa-6f872f7d50d3.PNG">

- 선형 수식은 다음과 같음 
- w3의 가중치를 학습할 수 있음/w3이 비선형 정보를 인코딩해도 w3의 값을 결정하기 위해 선형 모델의 학습 방식을 변경하지 않아도 됨 
<img src="https://user-images.githubusercontent.com/32586985/70213212-2a394000-177c-11ea-82f4-4a533b17a376.PNG">

### 특성 교차의 종류
- 여러 종류의 특성 교차를 만들 수 있음 
  - [A X B]: 두 특성의 값을 곱하여 구성되는 특성 교차
  - [A X B X C X D X E]: 다섯 개의 특성 값을 곱하여 구성되는 특성 교차
  - [A X A]: 단일 특성을 제곱하여 구성되는 특성 교차
- 확률적 경사하강법을 활용하여 선형 모델을 효율적으로 학습시킬 수 있음 
- 예전부터 조정된 선형 모델을 특성 교차로 보완하는 방법으로 모델을 대규모 데이터 세트에 효율적으로 학습시켜옴 


## 특성 교차:원-핫 벡터 교차
- 별개의 두 부동 소수점 특성의 특성 교차에 초점을 맞추었지만 실제로는 연속 특성을 교차하는 경우는 거의 없음 
- 원-핫 특성 벡터의 특성 교차는 논리적 결합이라고 할 수 있음 
- 예를 들어 국가와 언어, 두 특성이 있다고 가정
  - 각 특성을 원 핫 인코딩하면 country=USA,country=France 또는 language=English,language=Spanish로 가능
  - 위와 같이 해석할 수 있는 이진 특성이 포함된 벡터가 생성됨
  - 이러한 원-핫 인코딩의 특성을 교차하여 다음과 같이 논리적 결합으로 해석할 수 있는 이진 특성이 생성됨
  ```python
     country:usa AND language:spanish
  ```
- 위도와 경도를 비닝하여 다섯 개의 요소로 구성된 별도의 원-핫 특성 벡터를 만든다고 가정
  ```python
     binned_latitude = [0, 0, 0, 1, 0]
     binned_longitude = [0, 1, 0, 0, 0]
     # 다음과 같이 표현 가능 
  ```
  - 이 두 특성 벡터의 특성 교차를 만든다고 가정 
  ```python
     binned_latitude X binned_longitude
  ```
  - 이 특성 교차는 25개의 요소로 구성된 원-핫 벡터임(24개의 0과 1개의 1)
  - 교차에 있는 1개의 1은 위도와 경도의 특정 결합을 나타냄/해당 결합의 특정 연결을 학습할 수 있음 
- 위도와 경도를 훨씬 더 넓은 간격으로 비닝한다고 가정 
  ```python
     binned_latitude(lat) = [
       0  < lat <= 10
       10 < lat <= 20
       20 < lat <= 30
     ]
     
     binned_longitude(lon) = [
       0  < lon <= 15
       15 < lon <= 30
     ]  
  ```
    - 다음과 같은 의미의 합성 특성이 생성됨 
    ```python
       binned_latitude_X_longitude(lat, lon) = [
         0  < lat <= 10 AND 0  < lon <= 15
         0  < lat <= 10 AND 15 < lon <= 30
         10 < lat <= 20 AND 0  < lon <= 15
         10 < lat <= 20 AND 15 < lon <= 30
         20 < lat <= 30 AND 0  < lon <= 15 
         10 < lat <= 30 AND 15 < lon <= 30
       ]   
    ```
- 모델에서 두 특성을 기반으로 개 주인이 개에 만족하는 정도를 예측해야 한다고 가정 
  - 행동 유형(짖기,울기,달라붙기 등)
  - 시간 
- 두 특성의 특성 교차를 만들면 
```python
   [behavior type X time of day]
```
- 특성 하나만 사용하는 경우보다 훨씬 더 효과적으로 예측할 수 있음 
- 선형 학습자는 대량의 데이터에 적합하게 확장됨/대량의 데이터에 특성 교차를 사용하면 매우 복잡한 모델을 효율적으로 학습할 수 있음 
- 신경망을 통해 다른 전략을 사용할 수도 있음 


## 특성교차 실습
- 세 가지 입력 특성의 가중치를 두어 파란색 점과 주황색 점을 구분하는 모델을 만들어봄 
- x1,x2,x1x2의 값을 각각 0,0,1로 설정했을 시
<img src="https://user-images.githubusercontent.com/32586985/70382617-57ffce00-19a2-11ea-8166-9ef9f3f533e6.PNG">

- 더 복잡한 특성 교차/파란색 점이 데이터 세트 가운데 있고 바깥쪽 둘레에 주황색 점이 있음 
  - 시각화와 관련 사항 
    - 파란색 점은 한 데이터 클래스의 한 가지 예/주황색 점 역시 다른 데이터 클래스의 한 가지 예
    - 배경 색상은 모델이 예측한 해당 색상 예의 위치를 나타냄
    - 파란색 점 주위의 파란색 배경은 모델이 해당 예를 올바르게 예측함을 의미함
    - 이와 반대로 파란색 점 주위의 주황색 배경은 모델이 해당 예를 잘못 예측하고 있음을 의미함 
    - 파란색과 주황색 배경은 농도가 조정됨/색상 농도는 모델이 얼마나 확실하게 추측하는지를 나타내는 것으로 간주함 
    - 짙은 파란색은 모델이 매우 확실하게 추측한다는 것을 의미/옅은 파란색은 모델이 덜 확실하게 추측한다는 것을 의미함 
    
- 작업 1: 지정된 대로 이 모델을 실행할 경우 선형 모델이 데이터 세트와 관련하여 효과적인 결과를 제공하는지?
- 선형 모델이 이 데이터 세트를 효과적으로 모델링하지 않음/학습률을 낮추면 손실이 감소하지만 여전히 허용할 수 없을 정도로 높은 값으로 수렴함
<img src="https://user-images.githubusercontent.com/32586985/70382714-68647880-19a3-11ea-81af-ef73f1fa82ca.PNG">
<img src="https://user-images.githubusercontent.com/32586985/70382718-6d292c80-19a3-11ea-9629-cf97f6d520f4.PNG"> 

- 작업 2: x1x2와 같은 교차 곱 특성을 추가하여 성능을 최적화해보자
  - 어느 특성이 가장 도움이 되는가?
  - 어떤 최고의 성능을 얻을 수 있는가?
- 데이터 세트가 무작위로 생성되므로 답이 정확히 일치하지 않을 수 있음/다음과 같은 조치를 취해봄  
  - x1^2, x2^2를 모두 특성교차로 사용함/x1x2는 크게 도움이 되지 않음(사진과 같이 형성되지도 않음) 
  - 학습률을 0.001정도로 낮춤 
<img src="https://user-images.githubusercontent.com/32586985/70382746-2425a800-19a4-11ea-93df-a456f28693a2.PNG">

- 작업 3: 적절한 모델이 있는 경우 배경 색상으로 표시되는 모델 출력 표면을 확인 
  - 모델 출력 표면이 선형 모델인가?
  - 모델을 어떻게 설명할 것인가?
- 모델 출력 표면이 선형 모델처럼 보이지 않음/타원형으로 보임 
<img src="https://user-images.githubusercontent.com/32586985/70382778-939b9780-19a4-11ea-8a85-dd46799b277a.PNG">


## 프로그래밍 실습 
- 합성 특성을 추가하여 선형 회귀 모델을 개선함 
```python
   from __future__ import print_function
   
   import math
   
   from IPython import display
   from matplotlib import cm
   from matplotlib import gridspec
   from matplotlib import pyploy as plt
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
     # Scale the target to be in units of thousands of dollars.
     output_targets["median_house_value"] = (
       california_housing_dataframe["median_house_value"] / 1000.0)
     return output_targets
     
   # Choose the first 12000 (out of 17000) examples for training
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
       features, labelds = ds.make_one_shot_iterator().get_next()
       return features, labels
```


### FTRL 최적화 알고리즘 
- 고차원 선형 모델에서는 경사 기반 최적화의 일종인 FTRL이 유용함
- 여러 가지 계수의 학습률을 서로 다르게 조정함 
- 일부 특성이 0이 아닌 값을 거의 취하지 않는 경우에 유용할 수 있음 
```python
   def train_model(
       learning_rate,
       steps,
       batch_size,
       feature_columns,
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
       feature_columns: A 'set' specifying the input feature columns to use.
       training_examples: A 'DataFrame' containing one or more columns from 
         'california_housing_dataframe' to use as input features for training.
       training_targets: A 'DataFrame' containing exactly one columns from
         'california_housing_dataframe' to use as input features for training.
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
     my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
     my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
     linear_regressor = tf.estimator.LinearRegressor(
         feature_columns=feature_columns,
         optimizer=my_optimizer
     )
     
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
           steps=steps_per_period
       )
       # Take a break and compute predictions.
       training_predictions = linear_regressor.predict(input_fn=perdict_training_input_fn)
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
     
     _ = train_model(
         learning_rate=1.0,
         steps=500,
         batch_size=100,
         feature_columns=construct_feature_columns(training_examples),
         training_examples=training_examples,
         training_targets=training_targets,
         validation_examples=validation_examples,
         validation_targets=validation_targets)        
```
<img src="https://user-images.githubusercontent.com/32586985/70383244-964eba80-19ad-11ea-9a51-19fd9addbec2.PNG">

### 불연속 특성에 대한 원-핫 인코딩 
- 
