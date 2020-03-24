## Step 4:Build, Train, and Evaluate Your Model
- 이제 모델을 building, training, evaluating하는 것을 할 것이고 / Step 3에서 n-gram모델이나 sequence model중 하나를 S/W 비율을 사용해서 선택해 사용할 것임 / 이제 classification algorithm을 쓰고 학습할 시기임 / TensorFlow의 keras API를 사용할 것임 
- keras와 함께 ML모델을 만드는것은 layers를 한 곳에 합치는것이고, data-processing building blocks, Lego bricks를 모으는 것 같은 것임 
- 이 layers은 our input에 우리가 원하는 정도의 수행을 할 수 있는 transformation의 sequence를 특정화하는 걸 허락하게함
- learning algorithm을 single text input과 single classification의 outputs으로 가지게 함으로써 Sequential model API를 사용함으로써 layers의 선형적인 stack을 만들 수 있음
<img src="https://user-images.githubusercontent.com/32586985/77385642-0b62c600-6dcc-11ea-8760-7049888cb046.PNG">

- input layer과 intermediate layers이 서로 다르게 constructed될 것이고, n-gram과 sequence model을 만드는데 달려있음 / 하지만 model type의 irrespective가 last layer에 주어진 문제에 같게 나타날 것임 

- Constructing the Last Layer
  - 오직 2개의 classes(binary classification)만을 가지고 있을 때, 모델은 single probability score가 output이어야만함 
  - 예를들어 주어진 input sample에 outputtin이 0.2라면 20%만이 이 sample이 first class(class 1)에 확실히 있고 80%가 second class(class 0)이 있음을 의미함 
  - 이러한 probability score의 output을 가지는것과 last layer의 activation function은 sigmoid function이어야하고, 모델을 학습하는데 사용된 loss function은 binary cross-entropy여야만 함(Figure 10의 왼쪽)
  - 만일 2 classes이상 있다면 (multi-class classification), 모델은 각 class마다 one probability score가 output이어야 함 
  - 이러한 scores의 합은 반드시 1이어야함 / 예를들어 outputting {0: 0.2, 1: 0.7, 2: 0.1}은 sample의 20%만이 class 0에 있고, 70%는 class 1에 있고, 10%는 class 2에 있다는것을 의미함 
  - 이러한 scores의 output은 last layer의 activation function은 softmax여야만하고 모델을 학습하는데 사용되는 loss function은 categorical cross-entropy여야만 함(Figure 10의 오른쪽)
  <img src="https://user-images.githubusercontent.com/32586985/77481776-b5dbf700-6e67-11ea-916e-4027fff4bae5.PNG">
  
  - 아래의 코드는 input으로써 classes의 수를 가지는 function을 정의하고, layer units의 정확한 수를 outputs로 하며(1 unit for binary classification, otherwise 1 unit for each class)그리고 정확한 activation function을 outputs로 함 
  ```python
     def _get_last_layer_units_and_activation(num_classes):
     """Gets the # units and activation function for the last network layer.

     # Arguments
         num_classes: int, number of classes.

     # Returns
         units, activation values.
     """
     if num_classes == 2:
         activation = 'sigmoid'
         units = 1
     else:
         activation = 'softmax'
         units = num_classes
     return units, activation
  ```
  
  - 다음의 2개의 sections은 n-gram models과 sequence models의 model layers을 남긴 creation을 확인해볼 것임
  - S/W 비율이 작다면, n-gram models이 sequence models보다 더 잘 수행하는 것을 알 수 있고 sequence models은 small,dense vectors의 수가 클 때 더 잘 수행됨 / 이것은 embedding relationships이 dense space에서 학습하기 때문이고, 그리고 이것은 많은 samples에서의 나타남

- Build n-gram model(Option A)
  - n-gram models로써 tokens이 독립적으로 process하는 models을 언급한 것임(not taking into account word order) / Simple multi-layer perceptrons(including logistic regression), gradient boosting machines and support vector machines models모두 이 카테고리 안에 있고, 그것은 text ordering에 대해서 어떠한 정보도 leverage 할 수 없음 
  - 위에서 언급한 몇몇 n-gram models의 수행능력을 비교했고, multi-later perceptrons(MLPs)가 일반적으로 다른 options보다 더 잘 수행됨을 확인했음 / MLPs는 정의하고 이해하기 쉬우며, 좋은 accuracy를 제공하며, 상대적으로 적은 계산을 필요로함 
  - 아래의 코드는 Dropout layers for regularization의 couple을 추가함으로써 two-layer MLP models을 (tf.keras에) 정의해줌 (training samples을 overfitting하는것을 막기위해서)
  ```python
     from tensorflow.python.keras import models
     from tensorflow.python.keras.layers import Dense
     from tensorflow.python.keras.layers import Dropout
     
     def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
         """Creates an instance of a multi-layer perceptron model.

     # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

     # Returns
         An MLP model instance.
     """
     op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
     model = models.Sequential()
     model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

     for _ in range(layers-1):
         model.add(Dense(units=units, activation='relu'))
         model.add(Dropout(rate=dropout_rate))

     model.add(Dense(units=op_units, activation=op_activation))
     return model     
  ```
