## 신경망 학습
- 역전파는 신경망의 가장 일반적인 학습 알고리즘임
- 다계층 신경망에서 경사하강법을 사용하려면 이 알고리즘이 필요함
- 다음 사항을 봐야함
  - 데이터가 그래프를 통과하는 방식
  - 동적 프로그래밍을 사용하면 기하급수적으로 증가하는 그래프 통과 경로를 일일이 계산할 필요가 없는 이유
  - 동적 프로그래밍은 정방향 및 역방향 전달에서 중간 결과를 기록함을 의미

### 역전파:숙지할 사항
- 경사의 중요성
  - 미분 가능하면 학습이 가능할 확률이 높음
- 경사의 소실 가능성
  - 레이어를 추가할수록 신호와 노이즈가 연속적으로 감소할 수 있음
  - ReLU의 유용성
- 경사의 발산 가능성
  - 학습률의 중요성
  - batch 정규화(유용한 노브)로 해결 가능
- ReLU 레이어의 소멸 가능성
  - 당황하지 말고 학습률 낮추기 
### 특성 값 정규화
- 특성에 합리적인 척도를 부여해야함
  - 0에 대략적인 중심을 둔 [-1,1] 범위가 일반적으로 유리함
  - 경사하강법이 더 빠르게 수렴되고 NaN 트랩이 방지됨
  - 이상점 값을 배제하는 방법도 도움이 됨 
- 몇 가지 표준 방법 사용 가능
  - 선형 조정
  - 최대값, 최소값 강제 제한(클리핑)
  - 로그 조정 
### 드롭아웃 정규화
- 드롭아웃:또 하나의 정규화 형태, NN에 유용
- 단일 경사 스텝에서 네트워크의 유닛을 무작위로 배제
  - 앙상블 모델과의 접점
- 드롭아웃이 많을수록 정규화가 강력해짐
  - 0.0 = 드롭아웃 정규화 없음
  - 1.0 = 전체 드롭아웃. 학습 중지
  - 중간 범위의 값이 유용함 

## 신경망 학습:권장사항
- 실패 사례
  - 몇 가지 일반적인 이유로 인해 역전파에서 문제가 나타날 수 있음 
  - 경사 소실
    - 입력 쪽에 가까운 하위 레이어의 경사가 매우 작아질 수 있음
    - 심층 네트워크에서 이러한 경사를 계산할 때는 많은 작은 항의 곱을 구하는 과정을 포함할 수 있음 
    - 하위 레이어의 경사가 0에 가깝게 소실되면 이러한 레이어에서 학습 속도가 크게 저하되거나 학습이 중지됨 
    - ReLU 활성화 함수를 통해 경사 소실을 방지할 수 있음 
  - 경사 발산
    - 네트워크에서 가중치가 매우 크면 하위 레이어의 경사에 많은 큰 항의 곱이 포함됨
    - 이러한 경우 경사가 너무 커져서 수렴하지 못하고 발산하는 현상이 나타남
    - batch 정규화를 사용하거나 학습률을 낮추면 경사 발산을 방지할 수 있음 
  - ReLU 유닛 소멸
    - ReLU 유닛의 가중 합이 0 미만으로 떨어지면 ReLU 유닛이 고착될 수 있음 
    - 이러한 경우 활동이 출력되지 않으므로 네트워크의 출력에 어떠한 영향도 없음
    - 역전파 과정에서 경사가 더 이상 통과할 수 없음 
    - 경사의 근원이 단절되므로 가중 합이 다시 0 이상으로 상승할 만큼 ReLU가 변화하지 못할 수도 있음 
    - 학습률을 낮추면 ReLU 유닛 소멸을 방지할 수 있음 
- 드롭아웃 정규화
  - 드롭아웃이라는 정규화 형태가 유용함 
  - 단일 경사 스텝에서 유닛 활동을 무작위로 배제하는 방식임 
  - 드롭아웃을 반복할수록 정규화가 강력해짐
    - 0.0 = 드롭아웃 정규화 없음
    - 1.0 = 전체 드롭아웃.모델에서 학습을 수행하지 않음
    - 0.0~1.0 범위 값 = 보다 유용함 


## 프로그래밍 실습  
- 신경망 성능 개선하기
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
   
   california_housing_dataframe = pd.read.csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
   
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
     
   # Choose the first 12000 (out of 17000) examples for training.
   training_examples = preprocess_features(california_housing_dataframe.head(12000))
   training_targets = preprocess_targets(california_housing_dataframe.head(12000))
   
   # Choose the last 5000 (out of 17000) examples for validation.
   validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
   validation_targets = preprocess_features(california_housing_dataframe.tail(5000))
   
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

- 신경망 학습
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
       """Trains a neural network model.
       
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
       
   def train_nn_regression_model(
       my_optimizer,
       steps,
       batch_size,
       hidden_units,
       training_examples,
       training_targets,
       validation_examples,
       validation_targets):
     """Trains a neural network regression model.
     
     In addition to training, this function also prints training progress information,
     as well as a plot of the training and validation loss over time.
     
     Args:
       my_optimizer: An instance of 'tf.train.Optimizer', the optimizer to use.
       steps: A non-zero 'int', the total number of training steps. A training step
         consists of a forward and backward pass using a single batch.
       batch_size: A non-zero 'int', the batch size.
       hidden_units: A 'list' of int values, specifying the number of neurons in each layer.
       training_examples: A 'DataFrame' containing one or more columns from
         'california_housing_dataframe' to use as input features for training.
       training_targets: A 'DataFrame' containing exactly one column from
         'california_housing_dataframe' to use as target for training.
       validation_examples: A 'DataFrame' containing one or more columns from 
         'california_housing_dataframe' to use as input features for validation.
       validation_targets: A 'DataFrame' containing exactly one column from
         'california_housing_dataframe' to use as target for validation.
         
     Returns:
       A tuple '(estimator, training_losses, validation_losses)':
         estimator: the trained 'DNNRegressor' object.
         training_losses: a 'list' containing the training loss values taken during training.
         validation_losses: a 'list' containing the validation loss values taken during training.
     """
     
     periods = 10
     steps_per_period = steps / periods
     
     # Create a DNNRegresor object.
     my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
     dnn_regressor = tf.estimator.DNNRegressor(
         features_columns=construct_feature_columns(training_examples),
         hidden_units=hidden_units,
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
     print("Trainin model...")
     print("RMSE (on training data):")
     training_rmse = []
     validation_rmse = []
     for period in range (0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        
        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
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
     
     print("Final RMSE (on training data):    %0.2f" % training_root_mean_squared_error)
     print("Final RMSE (on validation data):  %0.2f" % validation_root_mean_squared_error)
     
     return dnn_regressor, training_rmse, validation_rmse
     
     _ = train_nn_regression_model(
         my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
         steps=5000,
         batch_size=70,
         hidden_units=[10, 10],
         training_examples=training_examples,
         training_targets=training_targets,
         validation_examples=validation_examples,
         validation_targets=validation_targets)  
```

- 선형 조정
  - 입력값을 -1,1범위에 들어오도록 정규화하는것이 권장되는 표준 방식임
  - SGD에서 한 차원으로 너무 크거나 다른 차원으로 너무 작은 단계를 밟을 때 고착을 방지하는 데 도움이 됨
```python
   def linear_scale(series):
     min_val = series.min()
     max_val = series.max()
     scale = (max_val - min val) / 2.0
     return series.apply(lambda x:((x - min_val) / scale) - 1.0)     
```


### 작업1: 선형 조정을 사용하여 특성 정규화
- 입력값을 -1,1 척도로 정규화함
- 입력 특성이 대략 같은 척도일 때 NN의 학습 효율이 가장 높음 
- 정규화된 데이터의 상태를 확인해봐라
  - 정규화에 최소값과 최대값이 사용되므로 데이터 세트 전체에 한 번에 적용되도록 조치해야함 
- 데이터 세트가 여러 개인 경우에는 학습 세트에서 추출한 정규화 매개변수를 테스트 세트에 동일하게 적용하는 것이 좋음 
```python
   def normalize_linear_scale(examples_dataframe):
     """Returns a version of the input 'DataFrame' that has all its features normalized linearly."""
     processed_features = pd.DataFrame()
     processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
     processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
     processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
     processed_features["total_rooms"] = linear_scale(examples_dataframe["total_rooms"])
     processed_features["total_bedrooms"] = linear_scale(examples_dataframe["total_bedrooms"])
     processed_features["population"] = linear_scale(examples_dataframe["population"])
     processed_features["households"] = linear_scale(examples_dataframe["households"])
     processed_features["median_income"] = linear_scale(examples_dataframe["median_income"])
     processed_features["rooms_per_person"] = linear_scale(examples_dataframe["rooms_per_person"])
     return processed_features
     
   normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
   normalized_training_examples = normalized_dataframe.head(12000)
   normalized_validation_examples = normalized_dataframe.tail(5000)
   
   _ = train_nn_regression_model(
       my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.005),
       steps=2000,
       batch_size=50,
       hidden_units=[10, 10],
       training_examples=normalized_training_examples,
       training_targets=training_targets,
       validation_examples=normalized_validation_examples,
       validation_targets=validation_targets)
       
```
<img src="https://user-images.githubusercontent.com/32586985/71089983-4eb60300-21e5-11ea-944d-50d4f3c4a6f4.png">


### 작업2:다른 옵티마이저 사용해 보기
- Adagrad 옵티마이저/모델의 각 계수에 대해 학습률을 적응적으로 조정하여 유효 학습률을 단조적으로 낮춘다는 것
  - 볼록 문제에는 적합하지만 비볼록 문제 신경망 학습에는 이상적이지 않을 수 있음
  - Adagrad를 사용하려면 GradientDescentOptimizer 대신 AdagradOptimizer를 지정함
  - 학습률을 더 높여야 할 수 있음 
- Adam 옵티마이저
  - 비볼록 최적화 문제에 효율적임
  - tf.train.AdamOptimizer 메소드를 호출함 
  - 선택적으로 몇 가지 초매개변수를 인수로 취하지만 인수 중 하나만 지정함 
  - 프로덕션 설정에서는 선택적 초매개변수를 신중하게 지정하고 조정해야함 
  
```python
   _, adagrad_training_losses, adagrad_validation_losses = train_nn_regression_model(
       my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.5),
       steps=500,
       batch_size=100,
       hidden_units=[10, 10],
       training_examples=normalized_training_examples,
       training_targets=training_targets,
       validation_examples=normalized_validation_examples,
       validation_targets=validation_targets)
       
   # Adagrad 시험          
```
<img src="https://user-images.githubusercontent.com/32586985/71090502-807b9980-21e6-11ea-8260-14c06c036fad.png">

```python
   _, adam_training_losses, adam_validation_losses = train_nn_regression_model(
       my_optimizer=tf.train.AdamOptimizer(learning_rate=0.009),
       steps=500,
       batch_size=100,
       hidden_units=[10, 10],
       training_examples=normalized_training_examples,
       training_targets=training_targets,
       validation_examples=normalized_validation_examples,
       validation_targets=validation_targets)
       
   # Adam 시험    
```
<img src="https://user-images.githubusercontent.com/32586985/71090511-84a7b700-21e6-11ea-9509-4a562bc75afe.png">

```python
   plt.ylabel("RMSE")
   plt.xlabel("Periods")
   plt.title("Root Mean Squared Error vs. Periods")
   plt.plot(adagrad_training_losses, label='Adagrad training')
   plt.plot(adagrad_validation_losses, label='Adagrad validation')
   plt.plot(adam_training_losses, label='Adam training')
   plt.plot(adam_validation_losses, label='Adam validation')
   _ = plt.legend()
```
<img src="https://user-images.githubusercontent.com/32586985/71090520-8a050180-21e6-11ea-862a-3e3f4c1e355d.png">


### 작업3:대안적 정규화 방식 탐색
- 다양한 특성에 대안적인 정규화를 시도하여 성능을 더욱 높임
- 변환된 데이터의 요약 통계를 자세히 조사해 보면 선형 조정으로 인해 일부 특성이 -1에 가깝게 모이는 것을 알 수 있음 
- 여러 특성의 중앙값이 0.0이 아닌 -0.8근처임 
```python
   _ = training_examples.hist(bins=20, figsize=(18, 12), xlabelsize=2)
```
<img src="https://user-images.githubusercontent.com/32586985/71091151-ef0d2700-21e7-11ea-9c8c-70be6f699be0.png">

- 이러한 특성을 추가적인 방법으로 변환하면 성능이 더욱 향상될 수 있음 
- 로그 조정이 일부 특성에 도움이 될 수 있음/또는 극단값을 잘라내면 척도의 나머지 부분이 더 유용해질 수 있음 
```python
   def log_normalize(series):
     return series.apply(lambda x:math.log(x+1.0))
   
   def clip(series, clip_to_min, clip_to_max):
     return series.apply(lambda x:(
       min(max(x, clip_to_min), clip_to_max)))
   
   def z_score_normalize(series):
     mean = series.mean()
     std_dv = series.std()
     return series.apply(lambda x:(x - mean) / std_dv)
   
   def binary_threshold(series, threshold):
     return series.apply(lambda x:(1 if x > threshold else 0))
```
- 이러한 함수를 사용 직접 추가하여 작업 수행

- households, median_income, total_bedrooms는 모두 로그 공간에서 정규분포를 나타냄
- latitude, longitude, housing_median_age는 이전과 같이 선형 조정을 사용하는 방법이 더 좋을 수 있음 
- population, totalRooms, rooms_per_person에는 극단적인 이상점 존재,이런 점은 지나치게 극단적이므로 삭제하기로 함
```python
   def normalize(examples_dataframe):
     """Returns a version of the input 'DataFrame' that has all its features normalized."""
     processed_features = pd.DataFrame()
     
     processed_features["households"] = log_normalized(examples_dataframe["households"])
     processed_features["median_income"] = log_normalized(examples_dataframe["median_income"])
     processed_features["total_bedrooms"] = log_normalized(examples_dataframe["total_bedrooms"])
     
     processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
     processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
     processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
     
     processed_features["population"] = linear_scale(clip(examples_dataframe["population"], 0, 5000))
     processed_features["rooms_per_person"] = linear_scale(clip(examples_dataframe["rooms_per_person"], 0 ,5))
     processed_features["total_rooms"] = linear_scale(clip(examples_dataframe["total_rooms"], 0 , 10000))
     
     return processed_features
   
   normalized_dataframe = normalize(preprocess_features(califronia_housing_dataframe))
   normalized_training_examples = normalized_dataframe.head(12000)
   normalized_validation_examples = normalized_dataframe.tail(5000)
   
   _ = train_nn_regression_model(
       my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.15),
       steps=1000,
       batch_size=50,
       hidden_units=[10, 10],
       training_examples=normalized_training_examples,
       training_targets=training_targets,
       validation_examples=normalized_validation_examples,
       validation_targets=validation_targets)
```
<img src="https://user-images.githubusercontent.com/32586985/71092170-17962080-21ea-11ea-83e1-b6edf3297e56.png">


### 선택 과제:위도 및 경도 특성만 사용
- 특성으로 위도와 경도만 사용하는 NN 모델을 학습시킴
- 복잡한 비선형성을 학습할 수 있어야함 
```python
   def location_location_location(examples_dataframe):
     """Returns a version of the input 'DataFrame' that keeps only the latitude and longitude."""
     processed_features = pd.DataFrame()
     processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
     processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
     return processed_features
     
   lll_dataframe = location_location_location(preprocess_features(california_housing_dataframe))
   lll_training_examples = lll_dataframe.head(12000)
   lll_validation_examples = lll_dataframe.tail(5000)
   
   _ = train_nn_regression_model(
       my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.05),
       steps=500,
       batch_size=50,
       hidden_units=[10, 10, 5, 5, 5],
       training_examples=lll_training_examples,
       training_targets=training_targets,
       validation_examples=lll_validation_examples,
       validation_targets=validation_targets)
```
- 나쁘지 않은 결과 얻음/짧은 거리 내에서 속성 값이 크게 요동하는 경우가 있긴함 
<img src="https://user-images.githubusercontent.com/32586985/71092732-2204ea00-21eb-11ea-977d-0a48f222857e.png">

