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
