## 검증
- 두 개의 분할을 사용하는 워크플로우
- 모델 조정이란 학습률 변경, 특성 추가 또는 삭제, 완전히 새로운 모델 설계와 같이 모델에서 가능한 모든 요소를 조정함을 의미
- 그림 1. 가능한 워크플로우
<img src="https://user-images.githubusercontent.com/32586985/69413033-64a1e680-0d53-11ea-81d9-50946ee305c5.PNG">

- 데이터 세트를 셋으로 나누어 과적합 가능성을 크게 낮출 수 있음 
- 그림 2. 데이터 세트 하나를 셋으로 분할
<img src="https://user-images.githubusercontent.com/32586985/69413201-b5194400-0d53-11ea-9b4a-1fc918f5eba5.PNG">

- 검증세트를 사용하여 학습 세트의 결과를 평가함
- 그런 다음 모델이 검증세트를 통과한 후 테스트 세트를 사용하여 다시 평가해 봄 
- 그림 3. 워크플로우 개선
<img src="https://user-images.githubusercontent.com/32586985/69413325-f3aefe80-0d53-11ea-844a-6bdea9c90f57.PNG">

- 1. 검증세트에서 가장 우수한 결과를 보이는 모델을 선택함 
- 2. 테스트 세트를 기준으로 해당 모델을 재차 확인함 
- 테스트 세트가 보다 적게 노출되므로 더 우수함 


### 검증 프로그래밍 실습
- 단일 특성이 아닌 여러 특성을 사용하여 모델의 효과를 더욱 높임 
- 모델 입력 데이터의 문제를 디버깅함 
- 테스트 데이터 세트를 사용하여 모델이 검증 데이터에 과적합되었는지 확인함 

### 설정 
- 데이터 로드 
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
   # california_housing_dataframe = california_housing_dataframe.reindex(
   #     np.random.permutation(california_housing_dataframe.index))
   
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
```
- 학습 세트로 총 17000개 중 처음 12000개를 선택함 
```python
   training_examples = preprocess_features(california_housing_dataframe.head(12000))
   training_examples.describe()
```
<img src="https://user-images.githubusercontent.com/32586985/69414910-fb23d700-0d56-11ea-93ff-ddf3f8ec4953.PNG">

```python
   training_targets = preprocess_targets(california_housing_dataframe.head(12000))
   training_targets.describe()
```
<img src="https://user-images.githubusercontent.com/32586985/69415027-36bea100-0d57-11ea-8987-145178ce652a.PNG">

- 검증 세트로 총 17000개 중에서 마지막 5000개를 선택함 
```python
   validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
   validation_examples.describe()
```
<img src="https://user-images.githubusercontent.com/32586985/69415151-72596b00-0d57-11ea-9b2e-e8d754904b54.PNG">

```python
   validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
   validation_targets.describe()
```
<img src="https://user-images.githubusercontent.com/32586985/69415244-a0d74600-0d57-11ea-8fcb-8f35951a1743.PNG">



### 작업 1: 데이터 조사
- 사용 가능한 입력 특성은 9개임 
  - latitude/longitude/housing_median_age
  - total_rooms/total_bedrooms/population
  - households/median_income/rooms_per_person
- median_income의 척도는 약 3~15 범위인데, 그 의미가 불분명함/일종의 로그 척도로 보이지만 별도로 설명된 곳이 없음/높은 값은 높은 소득에 해당한다는 것을 유추할 수 있음  있음 
- median_house_value의 최대값은 500,001임/이 값은 인위적인 한도로 보임 
- rooms_per_person 특성의 척도는 일반적으로 상식에 부합/75번째 백분위수 값이 약 2임/18이나 55 같은 매우 큰 값이 보이며 이는 데이터 손상일 수 있음 

### 작업 2: 위도/경도와 주택 가격 중앙값을 비교하여 도식화  
- latitude와 longitude라는 두 가지 특성을 중점적으로 살펴봄/이들은 특정 구역의 지리 좌표를 나타냄
- latitude와 longitude는 도식화하고 median_house_value를 색상으로 표현
```python
   plt.figure(figsize=(13,8))
   
   ax = plt.subplot(1, 2, 1)
   ax.set_title("Validation Data")
   
   ax.set_autoscaley_on(False)
   ax.set_ylim([32, 43])
   ax.set_autoscalex_on(False)
   ax.set_xlim([-126, -112])
   plt.scatter(validation_examples["longitude"],
               validation_examples["latitude"],
               cmap="coolwarm",
               c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())
               
   ax = plt.subplot(1,2,2)
   ax.set_title("Training Data")
   
   ax.set_autoscaley_on(False)
   ax.set_ylim([32, 43])
   ax.set_autoscalex_on(False)
   ax.set_xlim([-126, -112])
   plt.scatter(training_examples["longitude"],
               training_examples["latitude"],
               cmap="coolwarm",
               c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
   _ = plt.plot()
```
<img src="https://user-images.githubusercontent.com/32586985/69417288-b9e1f600-0d5b-11ea-8ddc-73fe36b1fd2f.PNG">

- 정상적으로 기대되는 결과는 캘리포니아주 지도가 그려지면서 샌프란시스코,로스앤젤레스 같이 주택 가격이 높은 지역이 빨간색으로 표시됨
- 학습 세트는 어느정도 기대에 부합하지만 검증 세트는 그렇지 못함 
### 특성 또는 열의 종류에 관계없이 학습 세트와 검증 세트에서 값의 분포가 대략적으로 같아야함 
### 값의 분포가 다르다면 심각한 문제이고 학습 세트와 검증 세트를 만드는 방법이 잘못되었을 가능성이 높다는 의미  

### 작업 3: 데이터 가져오기 및 전처리 코드로 돌아가서 버그가 있는지 확인  
- ML의 디버깅은 코드 디버깅이 아닌 데이터 디버깅인 경우가 많음 
- 데이터가 잘못되었다면 가장 발전한 최신 ML 코드라도 문제를 일으킬 수 밖에 없음
- 데이터를 읽을 때 무작위로 섞는 부분
  - 학습 세트와 검증 세트를 만들때 데이터를 무작위로 적절히 섞지 않으면, 데이터가 일정한 규칙으로 정렬된 경우 문제가 생김
- 즉 데이터를 무작위로 섞을 때의 값을 조정할 경우 도식화된 값도 다르게 나옴   
- 다양한 값을 수정해 본 결과 어느정도 균등한 형태가 나옴/단 자릿수 하나까지 디테일하게 수정하진 않았음 
- 해당 데이터에 관한 표 값은 생략하겠음 도식화만에 집중하기 위해서
  - 설명을 더하자면 데이터의 값을 조정한 결과 학습 세트와 검증 세트 사이의 분포가 비슷해짐 
```python
   training_examples = preprocess_features(california_housing_dataframe.head(10500))
   training_targets = preprocess_targets(california_housing_dataframe.head(10500))
   validation_examples = preprocess_features(california_housing_dataframe.tail(6500))
   validation_targets = preprocess_features(california_housing_dataframe.tail(6500))
```
```python
plt.figure(figsize=(13,8))
   
   ax = plt.subplot(1, 2, 1)
   ax.set_title("Validation Data")
   
   ax.set_autoscaley_on(False)
   ax.set_ylim([32, 43])
   ax.set_autoscalex_on(False)
   ax.set_xlim([-126, -112])
   plt.scatter(validation_examples["longitude"],
               validation_examples["latitude"],
               cmap="coolwarm",
               c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())
               
   ax = plt.subplot(1,2,2)
   ax.set_title("Training Data")
   
   ax.set_autoscaley_on(False)
   ax.set_ylim([32, 43])
   ax.set_autoscalex_on(False)
   ax.set_xlim([-126, -112])
   plt.scatter(training_examples["longitude"],
               training_examples["latitude"],
               cmap="coolwarm",
               c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
   _ = plt.plot()
```
- 10500/6500으로 설정했을때의 도식화 값 
- 이 도식화보다 더 좋은 결과값이 나올 수 있음/정말 디테일하게 수정을 가할 경우
<img src="https://user-images.githubusercontent.com/32586985/69419126-40e49d80-0d5f-11ea-9d27-487976d2456e.PNG">

### 작업 4: 모델 학습 및 평가
- 데이터 세트의 모든 특성을 사용하여 선형 회귀 모델을 학습시키고 성능을 확인 
- 입력 함수 정의
```python
   def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
       """Trains a linear regression model of multiple features.
       
       Args:
         features: pandas DataFrame of features
         targets: pandas DataFrame of taregets
         batch_size: Size of batches to be passed to the model
         shuffle: True or False. Whether to shuffle the data.
         num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
       Returns:
         Tuple of (features, labels) for next data batch
       """
       
       # Convert pandas data into a dict of np arrays.
       features = {key:np.array(value) for key,value in dict(features).items()}
       
       # Construct a dataset, and configure batching/repeating 
       ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
       ds = ds.batch(batch_size).repeat(num_epochs)
       
       # Shuffle the data, if specified.
       if shuffle:
         ds = ds.shuffle(10000)
       
       # Return the next batch of data.
       features, labels = ds.make_one_shot_iterator().get_next()
       return features, labels
```
- 특성 열을 구성하는 코드를 모듈화하여 별도의 함수 만듬 
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
                 
```
- 학습 데이터와 검증 데이터의 손실을 비교
- 원시 특성이 하나일 때는 가장 양호한 평균 제곱근 오차(RMSE)가 약 180이었음 
```python
   def train_model(
       learning_rate,
       steps,
       batch_size,
       training_examples,
       training_targets,
       validation_examples,
       validation_targets):
     """Trains a linear regression model of multiple features.
     
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
     my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnin_rate)
     my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
     linear_regressor = tf.estimator.LinearRegressor(
         feature_columns=construct_feature_columns(training_examples),
         optimizer=my_optimizer
     )
     
     # Create input functions.
     training_input_fn = lambda: my_input_fn(
         training_examples,
         training_targets["median_house_value"],
         batch_size=batch_size)
     predict_training_input_fn = lambda: my_input_fn(
         training_examples,
         training_targets["median_house_value"],
         num_epochs=1,
         shuffle=False)
     predict_validation_input_fn = lambda: my_input_fn(
         validation_examples, validation_targets["median_house_value"],
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
       # Ocassionally print the current loss.
       print("   period %02d : %0.2f" % (period, training_root_mean_squared_error))
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
     plt.plot(validation_rmse, label="vcalidation")
     plt.legend()
     
     return linear_regressor         
```
```python
   linear_regressor = train_model(
       learning_rate=0.00003,
       steps=500,
       batch_size=5,
       training_examples=training_examples,
       training_targets=training_targets,
       validation_examples=validation_examples,
       validation_targets=validation_targets)
```
- 초기 12000/5000 일 경우
<img src="https://user-images.githubusercontent.com/32586985/69421640-e6e6d680-0d64-11ea-9c41-dc94a1ad0a27.PNG">
<img src="https://user-images.githubusercontent.com/32586985/69421645-e9e1c700-0d64-11ea-9e2b-e466f507e98f.PNG">

- 10500/6500 일 경우
<img src="https://user-images.githubusercontent.com/32586985/69421920-a76cba00-0d65-11ea-8e36-0b23b6ead7c2.PNG">
<img src="https://user-images.githubusercontent.com/32586985/69421924-ab98d780-0d65-11ea-8e30-766add9cd38f.PNG">

### 작업 5: 테스트 데이터로 평가
- 테스트 데이터 세트를 로드하고 이를 기준으로 모델을 평가함 
- 해당 표본의 특이성에 대한 과적합이 발생하지 않았는지 확인해야함 
```python
   california_hosuing_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")
   test_examples = preprocess_features(california_housing_test_data)
   test_targets = preprocess_targets(california_housing_test_data)
   
   predict_test_input_fn = lambda: my_input_fn(
         test_examples,
         test_targets["median_house_value"],
         num_epochs=1,
         shuffle=False)
         
   test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
   test_predictions = np.array([item['predictions'][0] for item in test_predictions])
   
   root_mean_squared_error = math.sqrt(
       metrics.mean_squared_error(test_predictions, test_targets))
       
   print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)                  
```
- 12000/5000 인 경우
<img src="https://user-images.githubusercontent.com/32586985/69421689-ffef8780-0d64-11ea-9386-b0807fff0ba2.PNG">

- 10500/6500 인 경우
<img src="https://user-images.githubusercontent.com/32586985/69421926-ae93c800-0d65-11ea-83fa-f068bfc20f76.PNG">
