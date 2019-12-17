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
