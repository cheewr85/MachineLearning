## Intro to Modeling

### Setup
```python
   import tensorflow as tf
   import numpy as np
   import pandas as pd
   import math
```

### Pandas, a helpful data analysis library for in-memory dataset
- 데이터를 읽는데 pandas를 사용할 것임 / 데이터를 탐색하고 기본적인 processing을 할 것임
- in memory의 데이터세트에 사용하는데 굉장히 도움이 됨 
```python
   # Set pandas output display to have one digit for decimal places and limit it to
   # printing 15 rows.
   pd.options.display.float_format = '{:.2f}'.format
   pd.options.display.max_rows = 15
```

### Load the dataset with pandas
```python
   # Provide the names for the columns since the CSV file with the data does
   # not have a header row.
   feature_names = [''symboling', 'normalized-losses', 'make', 'fuel-type',
        'aspiration', 'num-doors', 'body-style', 'drive-wheels',
        'engine-location', 'wheel-base', 'length', 'width', 'height', 'weight',
        'engine-type', 'num-cylinders', 'engine-size', 'fuel-system', 'bore',
        'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
        'highway-mpg', 'price']
   
   
   # Load in the data from a CSV file that is comma separated.
   car_data = pd.read_csv('https://storage.googleapis.com/mledu-datasets/cars_data.csv',
                        sep=',', names=feature_names, header=None, encoding='latin-1')
   
   
   # We'll then randomize the data, just to be sure not to get any pathological
   # ordering effects that might harm the performance of Stochastic Gradient
   # Descent.
   car_data = car_data.reindex(np.random.permutation(car_data.index))
   
   print("Data set loaded. Num examples: ", len(car_data))
```
- Data set loaded. Num examples:  205
- 예시가 205개 있는 작은 데이터 세트임 

### Task 0:Use pandas to explore and prepare the data
- pandas를 이용하여 데이터를 관찰하고 numeric_feature_names과 categorical_feature_names의 리스트를 수동적으로 curate 해보아라
```python
   car_data[4:7]
```
<img src="https://user-images.githubusercontent.com/32586985/74722571-7fb1c300-527c-11ea-9339-8811ed40537b.png">

```python
   LABEL = 'price'
   
   numeric_feature_names = car_data[['symboling','normalized-losses','wheel-base','engine-size','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']]
   categorical_feature_names = list(set(feature_names) - set(numeric_feature_names) - set([LABEL]))
   
   # The correct solution will pass these assert statements.
   assert len(numeric_feature_names) == 15
   assert len(categorical_feature_names) == 10
```
- 오류가 발생함 / 오류를 해결하기 위해서 데이터를 관찰하고 주어진 문제를 해결해보자

```python
   # Run to inspect numeric features.
   car_data[numeric_feature_names]
```
<img src="https://user-images.githubusercontent.com/32586985/74722988-2b5b1300-527d-11ea-837d-69e8f0298610.png">

```python
   # Run to inspect categorical features.
   car_data[categorical_feature_names]
```
<img src="https://user-images.githubusercontent.com/32586985/74723066-4f1e5900-527d-11ea-9a02-1b368d0a1125.png">

```python
   for feature_name in numeric_feature_names + [LABEL]:
     car_data[feature_name] = pd.to_numeric(car_data[feature_name], errors='coerce')
   
   car_data.fillna(0, inplace=True)
```
- Solution
```python
   numeric_features_names = ['symboling', 'normalized-losses', 'wheel-base',
        'length', 'width', 'height', 'weight', 'engine-size', 'horsepower',
        'peak-rpm', 'city-mpg', 'highway-mpg', 'bore', 'stroke',
         'compression-ratio']
   
   categorical_feature_names = list(set(feature_names) - set(numeric_feature_names) - set([LABEL]))
   
   assert len(numeric_feature_names) == 15
   assert len(categorical_feature_names) == 10
```

### Task 1:Make your best model with numeric features. No normalization allowed.
- the lowest eval loss를 얻기 위해 제공된 모델을 수정하라
- 다양한 hyperparameters를 바꾸어야 할 것임
   - learning rate
   - choice of optimizer
   - hidden layer dimensions - make sure your choice here makes sense given the number of training examples
   - batch size
   - num training steps
   - (anything else you can think of changing)
- 원본 (NaN loss발생, 에러가 
```python
   batch_size = 16
   
   print(numeric_feature_names)
   x_df = car_data[numeric_feature_names]
   y_series = car_data['price']
   
   # Create input_fn's so that the estimator knows how to read in your data.
   train_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       y=y_series,
       batch_size=batch_size,
       num_epochs=None,
       shuffle=True)
   
   eval_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       y=y_series,
       batch_size=batch_size,
       shuffle=False)
       
   predict_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       batch_size=batch_size,
       shuffle=False)
   
   # Feature columns allow the model to parse the data, perform common
   # preprocessing, and automatically generate an input layer for the tf.Estimator.
   model_feature_columns = [
       tf.feature_column.numeric_column(feature_name) for feature_name in numeric_feature_names
   ]
   print('model_feature_columns', model_feature_columns)
   
   est = tf.estimator.DNNRegressor(
       feature_columns=model_feature_columns,
       hidden_units=[64],
       optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
     )
     
   # TRAIN
   num_print_statements = 10
   num_training_steps = 10000
   for _ in range(num_print_statements):
     est.train(train_input_fn, steps=num_training_steps // num_print_statements)
     scores = est.evaluate(eval_input_fn)
     
     # The 'scores' dictionary has several metrics automatically generated by the
     # canned Estimator.
     # 'average_loss' is the average loss for an individual example.
     # 'loss' is the summed loss for the batch.
     # In addition to these scalar losses, you may find the visualization functions
     # In the next cell helpful for debugging model quality.
     print('scores', scores)
```

- 몇 번의 수정을 거침
```python
   # changing other parameters could improve model quality, but take it with a
   # grain of salt. The dataset is very small.
   
   batch_size = 16
   
   print(numeric_feature_names)
   x_df = car_data[numeric_feature_names]
   y_series = car_data['price']
   
   train_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       y=y_series,
       batch_size=batch_size,
       num_epochs=None,
       shuffle=True)
   
   eval_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       y=y_series,
       batch_size=batch_size,
       shuffle=False)
   
   predict_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       batch_size=batch_size,
       shuffle=False)
   
   # Feature columns allow the model to parse the data, perform common
   # preprocessing, and automatically generate an input layer for the tf.Estimator.
   model_feature_columns = [ 
       tf.feature_column.numeric_column(feature_name) for feature_name in numeric_feature_names
   ]
   print('model_feature_columns', model_feature_columns)
   
   est = tf.estimator.DNNRegressor(
       feature_columns=model_feature_columns,
       hidden_units=[64],
       optimizer=tf.train.AdagradOptimizer(learning_rate=0.01),
     )
   
   # TRAIN
   num_print_statements = 10
   num_training_steps = 10000
   for _ in range(num_print_statements):
     est.train(train_input_fn, steps=num_training_steps // num_print_statements)
     scores = est.evaluate(eval_input_fn)
     
     # The 'scores' dictionary has several metrics automatically generated by the
     # canned Estimator.
     # 'average_loss' is the average loss for an individual example.
     # 'loss' is the summed loss for the batch.
     # In addition to these scalar losses, you may find the visualization functions
     # In the next cell helpful for debugging model quality.
     print('scores', scores)
```
- 해당 코드를 실행시 밑의 이미지와 같이 지속적으로 실행 후 마무리됨
<img src="https://user-images.githubusercontent.com/32586985/74725675-91e23000-5281-11ea-84b8-032301fe1b0b.png">
<img src="https://user-images.githubusercontent.com/32586985/74725752-acb4a480-5281-11ea-851b-77b7a9b93f5a.png">
<img src="https://user-images.githubusercontent.com/32586985/74725764-b0e0c200-5281-11ea-9ceb-4f2641ea30cf.png">

### Visualize your model's predictions
- 모델이 학습한 후에, 모델의 inference가 실제 데이터와 어떻게 다른지 이해하는데 도움이 될 것임
- 실제 데이터는 회색이고, 예측한 모델은 오렌지색임
```python
   from matplotlib import pyplot as plt
   
   
   def scatter_plot_inference_grid(est, x_df, feature_names):
     """Plots the predictions of the model against each feature.
     
     Args:
       est: The trained tf.Estimator.
       x_df: The pandas dataframe with the input data (used to create
         predict_input_fn).
       feature_names: An iterable of string feature names to plot.
     """
     def scatter_plot_inference(axis,
                                x_axis_feature_name,
                                y_axis_feature,name,
                                predictions):
       """Generate one subplot."""
       # Plot the real data in grey.
       y_axis_feature_name = 'price'
       axis.set_ylabel(y_axis_feature_name)
       axis.set_xlabel(x_axis_feature_name)
       axis.scatter(car_data[x_axis_feature_name],
                    car_data[y_axis_feature_name],
                    c='grey')
       
       # Plot the predicted data in orange.
       axis.scatter(car_data[x_axis_feature_name], predictions, c='orange')
     
     predict_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       batch_size=batch_size,
       shuffle=False)
     
     predictions = [
       x['predictions'][0]
       for x in est.predict(predict_input_fn)
     ]
     
     num_cols = 3
     num_rows = int(math.cell(len(feature_names)/float(num_cols)))
     f, axarr = plt.subplots(num_rows, num_cols)
     size = 4.5
     f.set_size_inches(num_cols*size, num_rows*size)
     
     for i, feature_name in enumerate(numeric_feature_names):
       axis = axarr[int(i/num_cols), i%num_cols]
       scatter_plot_inference(axis, feature_name, 'price', predictions)
     plt.show()
   
   scatter_plot_inference_grid(est, x_df, numeric_feature_names)
```
<img src="https://user-images.githubusercontent.com/32586985/74727032-d969bb80-5283-11ea-9ca0-368f09a25e57.png">
<img src="https://user-images.githubusercontent.com/32586985/74727046-df5f9c80-5283-11ea-9619-8c7c79ced312.png">

### Task 2:Take your best numeric model from earlier. Add normalization.
- normalization을 추가하여 최고의 numeric model을 만들자
   - 어떠한 종류의 normalization을 추가할 지 어떤 features를 쓸 지 결정해야함
   - numeric_column에 normalizer_fn을 사용함
   - 몇몇의 pandas functions 사용
   - hyperparameters에 대해서 retune해야함
```python
   for feature_name in numeric_feature_names:
     car_data.hist(column=feature_name)
```
<img src="https://user-images.githubusercontent.com/32586985/74727522-a247da00-5284-11ea-9590-6f27199b9c14.png">
<img src="https://user-images.githubusercontent.com/32586985/74727545-ad9b0580-5284-11ea-97c5-47123c4cef0c.png">
<img src="https://user-images.githubusercontent.com/32586985/74727560-b7bd0400-5284-11ea-9f48-a2d70dac7a0f.png">
<img src="https://user-images.githubusercontent.com/32586985/74727581-c0add580-5284-11ea-9725-0586a4055185.png">
<img src="https://user-images.githubusercontent.com/32586985/74727611-c99ea700-5284-11ea-96e9-252447217d68.png">

- Train your model with numeric features + normalization
```python
  # This does Z-score normalization since the distributions for most features looked
  # roughly normally distributed.

  # Z-score normalization subtracts the mean and divides by the standard deviation,
  # to give a roughly standard normal distribution (mean = 0, std = 1) under a
  # normal distribution assumption. Epsilon prevents divide by zero.

  # With normalization, are you able to get the model working with 
  # GradientDescentOptimizer? Z-score normalization doesn't seem to be able to get
  # SGD working. Maybe a different type of normalization would?
  
  batch_size = 16
  
  print(numeric_feature_names)
  x_df = car_data[numeric_feature_names]
  y_series = car_data['price']
  
  train_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=x_df,
      y=y_series,
      batch_size=batch_size,
      num_epochs=None,
      shuffle=True)
      
  eval_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=x_df,
      y=y_series,
      batch_size=batch_size,
      shuffle=False)
  
  predict_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=x_df,
      batch_size=batch_size,
      shuffle=False)
  
  # Epsilon prevents divide by zero.
  epsilon = 0.000001
  model_feature_columns = [
      tf.feature_column.numeric_column(feature_name,
                                       normalizer_fn=lambda val: (val - x_df.mean()[feature_name]) / (epsilon + x_df.std()[feature_name]))
      for feature_name in numeric_feature_names
  ]
  print('model_feature_columns', model_feature_columns)
  
  est = tf.estimator.DNNRegressor(
      feature_columns=model_feature_columns,
      hidden_units=[64],
      optimizer=tf.train.AdgradOptimizer(learning_rate=0.01),
  )
  
  # TRAIN
  num_print_statements = 10
  num_training_steps = 10000
  for _ in range(num_print_statements):
    est.train(train_input_fn, steps=num_training_steps // num_print_statements)
    scores = est.evaluate(eval_input_fn)
    
    # The `scores` dictionary has several metrics automatically generated by the 
    # canned Estimator.
    # `average_loss` is the average loss for an individual example.
    # `loss` is the summed loss for the batch.
    # In addition to these scalar losses, you may find the visualization functions
    # in the next cell helpful for debugging model quality.
    print('scores', scores)
  
  scatter_plot_inference_grid(est, x_df, numeric_feature_names)
```
- 학습을 하고 마지막으로 시각화함
<img src="https://user-images.githubusercontent.com/32586985/74728823-c99fa680-5286-11ea-98e4-978c9a6e7921.png">
<img src="https://user-images.githubusercontent.com/32586985/74728862-dae8b300-5286-11ea-9512-4fac735220bb.png">
<img src="https://user-images.githubusercontent.com/32586985/74728894-e6d47500-5286-11ea-8af4-226262fa2d33.png">
<img src="https://user-images.githubusercontent.com/32586985/74728919-f05ddd00-5286-11ea-9773-23f22c92e9c8.png">
<img src="https://user-images.githubusercontent.com/32586985/74728939-f9e74500-5286-11ea-9e81-9a260267b396.png">

### Task 3:Make your best model using only categorical features
- categorical features를 위해 사용 가능한 feature columns를 보아라
```python
   # We have the full list of values that each feature takes on, and the list is
   # relatively small so we use categorical_column_with_vocabulary_list.
   
   batch_size = 16
   
   x_df = car_data[categorical_feature_names]
   y_series = car_data['price']
   
   train_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       y=y_series,
       batch_size=batch_size,
       num_epochs=None,
       shuffle=True)
   
   eval_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       y=y_series,
       batch_size=batch_size,
       shuffle=False)
   
   predict_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       batch_size=batch_size,
       shuffle=False)
   
   model_feature_columns = [
       tf.feature_column.indicator_column(
           tf.feature_column.categorical_column_with_vocabulary_list(
               feature_name, vocabulary_list=car_data[feature_name].unique()))
    for feature_name in categorical_feature_names
   ]
   print('model_feature_columns', model_feature_columns)
   
   est = tf.estimator.DNNRegressor(
       feature_columns=model_feature_columns,
       hidden_units[64],
       optimizer=tf.train.AdagradOptimizer(learning_rate=0.01),
  )
  
  # TRAIN
  num_print_statements = 10
  num_training_steps = 10000
  for _ in range(num_print_statements):
    est.train(train_input_fn, steps=num_training_steps // num_print_statements)
    scores = est.evaluate(eval_input_fn)
    
    # The `scores` dictionary has several metrics automatically generated by the
    # canned Estimator.
    # `average_loss` is the average loss for an individual example.
    # `loss` is the summed loss for the batch.
    # In addition to these scalar losses, you may find the visualization functions
    # in the next cell helpful for debugging model quality.
    print('scores', scores)
```
- 진행과정은 위의 예시와 동일하나 과정중에 발생하는 값이 다름 결론적으로 아래 사진과 같이 최종값은 다름
<img src="https://user-images.githubusercontent.com/32586985/74729705-3bc4bb00-5288-11ea-8f43-c18ca69db5b0.png">

### Task 4:Using all the features, make the best model that you can make
- 모든 features를 종합하여, numerical과 categorical model만 사용하는 모델보다 더 좋은 성능을 구사하는 모델을 만들어보자
```python
   # This is a first pass at a model that uses all the features.
   # Do you have any improvements?
   
   batch_size = 16
   
   x_df = car_data[numeric_feature_names + categorical_feature_names]
   y_series = car_data['price']
   
   train_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       y=y_series,
       batch_size=batch_size,
       num_epochs=None,
       shuffle=True)
       
   eval_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       y=y_series,
       batch_size=batch_size,
       shuffle=False)
   
   predict_input_fn = tf.estimator.inputs.pandas_input_fn(
       x=x_df,
       batch_size=batch_size,
       shuffle=False)
   
   epsilon = 0.000001
   model_feature_columns = [
       tf.feature_column.indicator_column(
           tf.feature_column.categorical_column_with_vocabulary_list(
               feature_name, vocabulary_list=car_data[feature_name].unique()))
       for feature_name in categorical_feature_names
   ] + [
       tf.feature_column.numeric_column(feature_name,
                                        normalizer_fn=lambda val: (val - x_df.mean()[feature_name]) / (epsilon + x_df.std()[feature_name]))
       for feature_name in numeric_feature_names
   ]
   
   
   print('model_feature_columns', model_feature_columns)
   
   est = tf.estimator.DNNRegressor(
       feature_columns=model_feature_columns,
       hidden_units=[64],
       optimizer=tf.train.AdagradOptimizer(learning_rate=0.01),
  )
  
  # TRAIN
  num_print_statements = 10
  num_training_steps = 10000
  for _ in range(num_print_statements):
    est.train(train_input_fn, steps=num_training_steps // num_print_statements)
    scores = est.evaluate(eval_input_fn)
  
    # The `scores` dictionary has several metrics automatically generated by the 
    # canned Estimator.
    # `average_loss` is the average loss for an individual example.
    # `loss` is the summed loss for the batch.
    # In addition to these scalar losses, you may find the visualization functions
    # in the next cell helpful for debugging model quality.
    print('scores', scores)
```
- 진행과정은 위의 예시와 동일하나 과정중에 발생하는 값이 다름 결론적으로 아래 사진과 같이 최종값은 다름
<img src="https://user-images.githubusercontent.com/32586985/74730676-eab5c680-5289-11ea-9115-403f5da940d0.png">
