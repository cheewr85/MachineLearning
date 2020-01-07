## 신경망 
- 비선형 분류 문제
  - 비선형/b+w1x1+w2x2형태의 모델로 라벨을 정확하게 예측할 수 없다는 의미임
  - 결정 표면은 선이 아님
  <img src="https://user-images.githubusercontent.com/32586985/70856668-306db000-1f24-11ea-903a-bd4410423072.PNG">
  
- 복잡한 비선형 분류 문제의 데이터 세트
<img src="https://user-images.githubusercontent.com/32586985/70856679-4ed3ab80-1f24-11ea-91a6-72de222aeb7d.PNG">

- 신경망이 비선형 문제를 해결하는 데 어떻게 도움이 되는지 알아보기 위해 선형 모델을 그래프로 나타냄
- 각 파란색 원은 입력 기능을 나타내고, 녹색 원은 입력의 가중합을 나타냄 
<img src="https://user-images.githubusercontent.com/32586985/70856703-d0c3d480-1f24-11ea-85f6-98516ea05fd1.PNG">

### 히든 레이어
- 중간값의 히든 레이어를 추가함/각 노란색 노드는 파란색 입력 노드 값의 가중합임/출력은 노란색 노드의 가중합임
- 여전히 입력의 선형 조합임 
<img src="https://user-images.githubusercontent.com/32586985/70856719-0799ea80-1f25-11ea-9feb-c88e6a56a863.PNG">

- 가중합의 두번째 히든 레이어를 추가함 
- 여전히 선형 모델임/출력을 입력의 함수로 표현하고 단순화하면 입력의 또 다른 가중합을 얻게됨
- 이 합계는 비선형 문제를 효과적으로 모델링하지 않음 
<img src="https://user-images.githubusercontent.com/32586985/70856736-65c6cd80-1f25-11ea-957e-df2136cf9cc5.PNG">

### 활성화 함수
- 비선형 문제를 모델링하기 위해 비선형성을 직접 도입할 수 있음/각 히든 레이어의 노드가 비선형 함수를 통과하도록 할 수 있음 
- 아래의 그래프로 나타낸 모델에서 히든 레이어1의 각 노드 값이 비선형 함수로 변환한 후에 다음 레이어의 가중 합으로 전달됨 
- 이 비선형 함수를 활성화 함수라고 함
- 활성화 함수를 추가하였으므로 레이어를 추가하면 효과가 더 큼
- 비선형성을 누적하면 입력과 예측 출력 간의 매우 복잡한 관계를 모델링 할 수 있음 
- 각 레이어는 원시 입력에 적용되는 더 높은 수준의 복잡한 함수를 효과적으로 학습함
<img src="https://user-images.githubusercontent.com/32586985/70856750-b76f5800-1f25-11ea-9121-2525ddb95a2e.PNG">

- 일반적인 활성화 함수
  - 시그모이드 활성화 함수는 가중합을 0과 1사이의 값으로 변환함
  <img src="https://user-images.githubusercontent.com/32586985/70856771-0e752d00-1f26-11ea-83c7-6a37064f7f69.PNG">
  
  - 구성은 다음과 같음
  <img src="https://user-images.githubusercontent.com/32586985/70856776-2c429200-1f26-11ea-85ab-51e5fa760b33.PNG">
  
- 정류 선형 유닛(ReLU)활성화 함수
  - 시그모이드와 같은 매끄러운 함수보다 조금 더 효과적이지만, 훨씬 쉽게 계산할 수 있음 
  <img src="https://user-images.githubusercontent.com/32586985/70856790-6875f280-1f26-11ea-8a4e-1498662db158.PNG">
  
  - ReLU의 반응성 범위가 더 유용함
  <img src="https://user-images.githubusercontent.com/32586985/70856791-6ad84c80-1f26-11ea-9dfd-3b82996579ab.PNG">
  
- 어떠한 수학 함수라도 활성화 함수의 역할을 할 수 있음
- σ가 활성화 함수(ReLU,시그모이드등)를 나타낸다고 가정한다면
- 결과적으로 네트워크 노드 값은 다음 수식으로 나타냄
<img src="https://user-images.githubusercontent.com/32586985/70856818-e0dcb380-1f26-11ea-9d57-96965155a5ae.PNG">


## 실습
- 처음 만들어보는 신경망
  - 과제1.주어진 모델은 두 개의 입력 특성을 하나의 뉴련으로 결합함/이 모델이 비선형성을 학습할 수 있을까?
  - 활성화가 선형으로 설정되어 있으므로 이 모델은 어떠한 비선형성도 학습할 수 없음/손실은 매우 높음 
  <img src="https://user-images.githubusercontent.com/32586985/70856905-0e762c80-1f28-11ea-9f43-8d5d358260bf.PNG">
  
  - 과제2.히든 레이어의 뉴런 수를 1개에서 2개로 늘려보고 선형 활성화에서 ReLU와 같은 비선형 활성화로 변경한다면 비선형성을 학습하나?
  - 비선형 활성화 함수는 비선형 모델을 학습할 수 있음/뉴런이 2개인 히든 레이어는 모델을 학습하는데 시간이 오래 걸림 
  - 이 실습은 비결정적이므로 일부 시도에서는 효과적인 모델을 학습하지 못함/다른 시도에서는 상당히 효과적으로 학습할 수 있음 
  <img src="https://user-images.githubusercontent.com/32586985/70856939-965c3680-1f28-11ea-9d95-2167e468b2a1.PNG">
  
  - 과제3.히든 레이어 및 레이어당 뉴런을 추가하거나 삭제하여 실험을 계속해본다/자유롭게 설정을 변경
  - 테스트 손실을 0.177이하로 얻는데 사용할 수 있는 가장 적은 노드 및 레이어수는 얼마인가?
  - 히든 레이어 3개의 테스트 손실이 매우 낮음
    - 1레이어에는 뉴런이 3개 있음/2레이어에는 뉴런이 3개 있음/3레이어에는 뉴런이 2개있음
  - 정규화도 L1정규화로 함
  <img src="https://user-images.githubusercontent.com/32586985/70856987-e687c880-1f29-11ea-90f7-773c71522f5b.PNG">
  
- 신경망 초기화
- XOR 데이터를 사용하여, 학습용 신경망의 반복성과 초기화의 중요성을 살펴봄
  - 과제1.주어진 모델을 4~5회 실행함/매번 시도하기 전에 네트워크 초기화 버튼을 눌러 임의로 새롭게 초기화함
  - 최소 500단계를 실행하도록 함/각 모델 출력이 어떤 형태로 수렴하나?/이 결과가 비볼록 최적화에서 초기화의 역할에 어떤 의미를 가지는가?
  - 각 시도마다 학습된 모델의 형태가 달라짐/테스트 손실 수렴 결과는 최저와 최고가 거으 2배까지 차이가 날 정도로 다양했음 
  <img src="https://user-images.githubusercontent.com/32586985/70857147-cd344b80-1f2c-11ea-977f-37b11b25342b.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857153-d9b8a400-1f2c-11ea-9d01-e1e46a3e0ec7.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857157-e3daa280-1f2c-11ea-88ac-0be4c2abb63a.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857162-efc66480-1f2c-11ea-8d4d-a01895213dd0.PNG">
  
  - 과제2.레이어 한 개와 추가 노드를 몇 개 더 추가하여 모델을 약간 더 복잡하게 만듬/과제1의 시도를 반복/결과에 안정성이 보강되나?
  - 레이어와 추가 노드를 추가하여 더 반복적인 결과를 얻음/매 시도마다 결과 모델은 거의 같은 형태였음
  - 테스트 손실 수렴 결과는 매 시도마다 변화가 적음 
  <img src="https://user-images.githubusercontent.com/32586985/70857209-b510fc00-1f2d-11ea-989e-35b3483907a8.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857223-c8bc6280-1f2d-11ea-9eb8-e8df883ce879.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857228-cce88000-1f2d-11ea-8b93-d5f7206d5354.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857231-d2de6100-1f2d-11ea-9733-e130503ae981.PNG">
  
- 나선형 신경망 
- 잡음이 있는 나선형임/선형 모델은 실패하고 직접 정의된 특성 교차도 구성이 어려울 수 있음 
  - 과제1.:x1과 x2만 사용하여 가능한 한 최고의 모델로 학습시켜 본다/자유롭게 설정 변경 가능
  - 얻을 수 있는 최고의 테스트 손실은 얼마인가?/모델 출력 표면은 얼마나 매끄러운가?
  - 레이어를 추가하고 노드를 많이 추가하면 좋지만 모델 속도가 느려지고 좋지 않음/모델을 해석하기도 어려움
  <img src="https://user-images.githubusercontent.com/32586985/70858950-c6b6cb80-1f4e-11ea-8d11-5e43d917f663.PNG">
  
  - 레이어, 노드를 줄여 조금은 부드러운 형태를 생성함
  <img src="https://user-images.githubusercontent.com/32586985/70858963-0e3d5780-1f4f-11ea-9653-fe74a0747966.PNG">
  
  - 학습률,정규화,정규화율,노이즈 등을 조정하여 조금 더 부드럽고 매끄러운 표면과 손실을 줄일 수 있음 
  
  
  - 과제2:신경망이라도 최고의 성능을 도달하기 위해서는 특성 추출이 일부 필요함
  - 추가 교차 특성이나 sin(X1),sin(X2)과 같은 기타 변환을 추가해보아라
  - 더 나은 모델이 도출되는가?/모델 표면이 더 매끄러워지는가?
  - 기타 변환을 추가할 경우 첫 번째 레이어에서 특성이 훨씬 복잡해짐/곡선 또한 복잡해지는 구조를 가지게 됨
  - 레이어와 뉴런의 수를 줄이고 학습률일 낮추어봄/데이터가 적합해지고 손실 모델도 더 나아짐
  - 단 하나의 히든 레이어와 5개의 뉴런만 있는 단순한 모델에서도 학습률과 정규화율, 활성화 함수 등 몇 가지 다른 매개변수를 변경함
  - 훨씬 나은 테스트 손실과 훨씬 부드러운 적합성 곡선을 얻을 수 있음 
    - 영상의 해설만큼 결과값이 안 나오므로 사진 첨부 생략
  
  
  ## 프로그래밍 실습 
  - 표준 회귀 작업에서부터 시작
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
       """Prepares targets features (i.e., labels) from California housing data set.
       
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
     training_targets = preprocess_features(california_housing_dataframe.head(12000))
     
     # Choose the last 5000 (out of 17000) examples for validation.
     validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
     validation_targets = preprocess_features(california_housing_datafrmae.tail(5000))
     
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
  
- 신경망 구축
- 신경망은 DNNRegressor 클래스에 의해 정의됨
- hidden_units를 사용하여 신경망의 구조를 정의함/hidden_units 인수는 정수의 목록을 제공
- 각 정수는 히든 레이어에 해당하고 포함된 노드의 수를 나타냄 
- hidden_units=[3,10]
  - 히든 레이어 2개를 갖는 신경망을 지정함
  - 1번 히든 레이어는 노드 3개를 포함함
  - 2번 히든 레이어는 노드 10개를 포함함
- 레이어를 늘리려면 목록에 정수를 더 추가하면 됨/기본적으로 모든 히든 레이어는 ReLU 활성화를 사용하며 완전 연결성을 가짐 
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
       """Trains a neural net  regression model.
       
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
       learning_rate,
       steps,
       batch_size,
       hidden_units,
       training_examples,
       training_targets,
       validation_examples,
       validation_targets):
     """Trains a neural network regression model.
     
     In addition to trainig, this function also prints training progress information,
     as well as a plot of the training and validation loss over time.
     
     Args:
       learning_rate: A 'float', the learning rate.
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
       A 'DNNRegressor' object trained on the training data.
     """
     
     periods = 10
     steps_per_period = steps / periods 
     
     # Create a DNNRegressor object.
     my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
     my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
     dnn_regressor = tf.estimator.DNNRegressor(
         feature_columns=construct_feature_columns(training_examples),
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
                                                       shuffel=False)
     
     # Train the model, but do so inside a loop so that we can periodically assess
     # loss metrics.
     print("Training model...")
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
           metrics.mean_squared_error(training_perdictions, training_targets))
       validation_root_mean_squared_error = math.sqrt(
           metrics.mand_squared_error(validation_perdictions, validation_targets))
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
     
     print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
     print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)
     
     return dnn_regressor                    
```

### 작업 1:NN 모델 학습 
- RMSE 110미만으로 낮추는 것을 목표로 초매개변수 조정함 
- 다양한 학습 설정을 수정하여 검증 데이터에 대한 정확성을 높이는 것
```python
   dnn_regressor = train_nn_regression_model(
       learning_rate=0.001,
       steps=2000,
       batch_size=100,
       hidden_units=[10, 10],
       training_examples=training_examples,
       training_targets=training_targets,
       validation_examples=validation_examples,
       validation_targets=validation_targets)
```
<img src="https://user-images.githubusercontent.com/32586985/70859621-15b62e00-1f5a-11ea-97c1-1a3154b4bbf5.PNG">

### 작업2:테스트 데이터로 평가
- 검증 성능 결과가 테스트 데이터에 대해서도 유지되는지 확인 
- 만족할 만한 모델이 만들어졌다면 테스트 데이터로 평가하고 검증 성능과 비교해봄 
- 적절한 데이터 파일을 로드하고 전처리한 후 predict 및 mean_squared_error를 호출함
- 모든 레코드를 사용할 것이므로 테스트 데이터를 무작위로 추출할 필요는 없음 
```python
   california_housing_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")
   
   test_examples = preprocess_features(california_housing_test_data)
   test_targets = preprocess_targets(california_housing_test_data)
   
   predict_testing_input_fn = lambda: my_input_fn(test_examples,
                                                  test_targets["median_house_value"],
                                                  num_epochs=1,
                                                  shuffle=False)
                                                  
   test_predictions = dnn_regressor.predict(input_fn=predict_testing_input_fn)
   test_predictions = np.array([item['predictions'][0] for item in test_predictions])
   
   root_mean_squared_error = math.sqrt(
       metrics.mean_squared_error(test_predictions, test_targets))
       
   print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)    
```
- Final RMSE (on test data): 104.66
