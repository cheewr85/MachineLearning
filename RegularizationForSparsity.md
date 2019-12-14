## 희소성을 위한 정규화
- 특성 교차로 돌아가기
  - 희소 특성 교차는 특성 공간을 크게 늘릴 수 있음
  - 가능한 문제
    - 모델 크기(RAM)가 매우 커질 수 있음
    - '노이즈'계수(과적합의 원인)
- L1 정규화
  - L0 가중치 기준에 페널티를 주고자 함
    - 볼록하지 않은 최적화,NP-난해
  - L1 정규화로의 완화:
    - 절대값(가중치)의 합에 페널티를 줌
    - 볼록 문제
    - L2와는 달리 희소성을 유도
    
## L1 정규화
- 희소 벡터는 종종 많은 차원을 포함함/이러한 고차원 특성 벡터가 주어지면 모델 크기가 커질 수 있고 엄청난 양의 RAM이 필요함
- 고차원의 희소 벡터에서는 가중치가 정확하게 0으로 떨어지도록 유도하는 것이 좋음
- 가중치가 정확하게 0일 경우 모델에서 해당 특성을 삭제함/특성을 없애면 RAM이 절약되고 모델의 노이즈가 줄어들 수 있음 
- 적절히 선택한 정규화 항을 추가함으로써, 학습 시 수행한 최적화 문제에 아이디어 적용 가능 
- L2 정규화는 가중치를 작은 값으로 유도하지만 정확히 0.0으로 만들지는 못함 
- L0 정규화는 볼록 최적화 문제를 NP-난해임 볼록하지 않은 최적화 문제로 바꿔버리는 단점이 있음/효과적으로 사용할 수 없음 
- L1 정규화를 사용하여 모델에서 유용하지 않은 많은 계수를 정확히 0이 되도록 유도하여 추론 단계에서 RAM을 절약할 수 있음 
### L1정규화와 L2정규화 비교
- L2와 L1은 서로 다른 방식으로 가중치에 페널티를 줌 
  - L2는 가중치^2에 페널티를 줌
  - L1은 |가중치|에 페널티를 줌
- 결과적으로 L2와 L1은 서로 다르게 미분됨 
  - L2의 미분계수는 2*가중치임
  - L1의 미분계수는 K(가중치와 무관한 값을 갖는 상수)임 
- L2의 미분계수는 매번 가중치의 x%만큼 제거한다고 생각하면 됨/무한히 제거해도 그 값은 절대 0이 되지 않음/가중치를 0으로 유도하지 않음
- L1의 미분계수는 매번 가중치에서 일정 상수를 빼는 것으로 생각하면 됨/L1은 0에서 불연속성을 가지며 이로 인해 0을 지나는 빼기 결과값은 0이 됨
- L1을 통해 가중치가 제거됨/모든 가중치의 절대값에 페널티를 줌
<img src="https://user-images.githubusercontent.com/32586985/70844509-5e4fe780-1e85-11ea-82ed-b90f7c66fab9.PNG">


## 실습
- L1 정규화 검사
  - 작업1:L2 정규화/정규화율(람다):0.1
  <img src="https://user-images.githubusercontent.com/32586985/70844781-29de2a80-1e89-11ea-969d-962e56d52886.PNG">
  
  - 작업2:L2 정규화/정규화율(람다):0.3
  <img src="https://user-images.githubusercontent.com/32586985/70844786-411d1800-1e89-11ea-95ea-088dbf11a8ea.PNG">
  
  - 작업3:L1 정규화/정규화율(람다):0.1
  <img src="https://user-images.githubusercontent.com/32586985/70844789-55f9ab80-1e89-11ea-9c51-c1e63dc0a9c5.PNG">
  
  - 작업4:L1 정규화/정규화율(람다):0.3
  <img src="https://user-images.githubusercontent.com/32586985/70844797-732e7a00-1e89-11ea-8fd3-c9400977dfa5.PNG">
  
  - 작업5:L1 정규화/정규화율(람다):3(실험)
  <img src="https://user-images.githubusercontent.com/32586985/70844801-7fb2d280-1e89-11ea-83e4-d5741f2b5f49.PNG">
  
  - Q1.L2에서 L1으로 정규화를 전환하면 테스트 손실과 학습 손실 사이의 델타에 어떤 영향을 주는가?
    - L2에서 L1으로 정규화를 전환하면 테스트 손실과 학습 손실 사이의 델타가 대폭 줄어듬
  - Q2.L2에서 L1으로 정규화를 전환하면 학습된 가중치에 어떤 영향을 주는가?
    - L2에서 L1으로 정규화를 전환하면 학습된 모든 가중치를 완화함
  - Q3.L1정규화율(람다)을 높이면 학습된 가중치에 어떤 영향을 주는가?
    - L1정규화율을 높이면 일반적으로 학습된 가중치가 완화되지만, 정규화율이 지나치게 높아지면 모델이 수렴할 수 없고 손실도 굉장히 높아짐 


## 프로그래밍 실습
- 희소성과 L1 정규화
- 복잡도를 낮추는 방법 중 하나는 가중치를 정확히 0으로 유도하는 함수를 사용하는것
- 회귀와 같은 선형 모델에서 가중치 0은 해당 특성을 전혀 사용하지 않는 것과 동일함
- 과적합이 방지될 뿐 아니라 결과 모델의 효율성이 올라감
- 희소성을 높이는 좋은 방법임 
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
     
   # Choose the first 12000 (out of 17000) examples for training
   training_examples = preprocess_features(california_housing_dataframe.head(12000))
   training_targets = preprocess_targets(california_housing_dataframe.head(12000))
   
   # Choose the last 5000 (out of 17000) examples for validation
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
   
   def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
       """Trains a linear regression model.
       
       Args:
         features: pandas DataFrame of features
         targets: pandas DataFrame of targets
         batch_size: Size of batches to be passed to the model
         shuffle: True or False Whether to shuffle the data.
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
       
   def get_quantile_based_buckets(feature_values, num_buckets):
     quantiles = feature_values.quantile(
       [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
     return [quantiles[q] for q in quantiles.keys()]
     
   def construct_feature_columns():
     """Construct the TensorFlow Feature Columns.
     
     Returns:
       A set of feature columns
     """
     
     bucketized_households = tf.feature_column.bucketized_column(
       tf.feature_column.numeric_column("households"),
       boundaries=get_quantile_based_buckets(training_examples["households"], 10))
     bucketized_longitude = tf.feature_column.bucketized_column(
       tf.feature_column.numeric_column("longitude"),
       boundaries=get_quantile_based_buckets(training_examples["longitude"], 50))
     bucketized_latitude = tf.feature_column.bucketized_column(
       tf.feature_column.numeric_column("latitude"),
       boundaries=get_quantile_based_buckets(training_examples["latitude"], 50))
     bucketized_housing_median_age = tf.feature_column.bucketized_column(
       tf.feature_column.numeric_column("housing_median_age"),
       boundaries=get_quantile_based_buckets(training_examples["housing_median_age"], 10))
     bucketized_total_rooms = tf.feature_column.bucketized_column(
       tf.feature_column.numeric_column("total_rooms"),
       boundaries=get_quantile_based_buckets(training_examples["total_rooms"], 10))
     bucketized_total_bedrooms = tf.feature_column.bucketized_column(
       tf.feature_column.numeric_column("total_bedrooms"),
       boundaries=get_quantile_based_buckets(training_examples["total_bedrooms"], 10))
     bucketized_population = tf.feature_column.bucketized_column(
       tf.feature_column.numeric_column("population"),
       boundaries=get_quantile_based_buckets(training_examples["population"], 10))
     bucketized_median_income = tf.feature_column.bucketized_column(
       tf.feature_column.numeric_column("median_income"),
       boundaries=get_quantile_based_buckets(training_examples["median_income"], 10))
     bucketized_rooms_per_person = tf.feature_column.bucketized_column(
       tf.feature_column.numeric_column("rooms_per_person"),
       boundaries=get_quantile_based_buckets(training_examples["rooms_per_person"], 10))
       
     long_x_lat = tf.feature_column.crossed_column(
       set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000)
     
     feature_columns = set([
       long_x_lat,
       bucketized_longitude,
       bucketized_latitude,
       bucketized_housing_median_age,
       bucketized_total_rooms,
       bucketized_total_bedrooms,
       bucketized_population,
       bucketized_households,
       bucketized_median_income,
       bucketized_rooms_per_person])
       
     return feature_columns  
     
     def model_size(estimator):
       variables = estimator.get_variable_names()
       size = 0
       for variable in variables:
         if not any(x in variable
                    for x in ['global_step',
                              'centered_bias_weight',
                              'bias_weight',
                              'Ftrl']
                   ):
           size += np.count_nonzero(estimator.get_variable_value(variable))
       return size       
```

### 작업 1:효과적인 정규화 계수 구하기
- 모델 크기 600미만, 검증세트에 대한 로그 손실 0.36 미만이라는 두 조건을 모두 만족하는 L1 정규화 강도 매개변수를 구하여라
```python
   def train_linear_classifier_model(
       learning_rate,
       regularization_strength,
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
       learning_rage: A 'float', the learning rate.
       regularization_strength: A 'float' that indicates the strength of the L1
          regulaarization. A value of '0.0' means no regularization.
       steps: A non-zero 'int', the total number of training steps. A training step
         consists of a forward and backward pass using a single batch.
       feature_columns: A 'set' specifying the input feature columns to use.
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
     
     periods = 7
     steps_per_period = steps / periods
     
     # Create a linear classifier object.
     my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
     my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
     linear_classifier = tf.estimator.LinearClassifier(
         feature_columns=feature_columns,
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
     print("LogLoss (on validation data):")
     training_log_losses = []
     validaton_log_losses = []
     for period in range (0, periods):
       # Train the model, starting from the prior state.
       linear_classifier.train(
           input_fn=training_input_fn,
           steps=steps_per_period
       )
       # Take a break and compute predictions.
       training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
       training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
       
       validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
       validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
       
       # Compute training and validation loss.
       training_log_loss = metrics.log_loss(training_targets, training_probabilities)
       validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
       # Occasionally print the current loss.
       print("  period %02d : %0.2f" % (period, validation_log_loss))
       # Add the loss metrics from this period to our list.
       training_log_losses.append(training_log_loss)
       validation_log_losses.append(validation-log_loss)
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
     
     linear_classifier = train_linear_classifier_model(
         learning_rate=0.1,
         regularization_strength=0.1,
         steps=300,
         batch_size=100,
         feature_columns=construct_feature_columns(),
         training_examples=training_examples,
         training_targets=training_targets,
         validation_examples=validation_examples,
         validation_targets=validation_targets)
     print("Model size:", model_size(linear_classifier))    
```
<img src="https://user-images.githubusercontent.com/32586985/70846915-b399f200-1ea1-11ea-820b-f69549d09056.PNG">
