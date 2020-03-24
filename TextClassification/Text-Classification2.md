## Step 4:Build, Train, and Evaluate Your Model
- 이제 모델을 building, training, evaluating하는 것을 할 것이고 / Step 3에서 n-gram모델이나 sequence model중 하나를 S/W 비율을 사용해서 선택해 사용할 것임 / 이제 classification algorithm을 쓰고 학습할 시기임 / TensorFlow의 keras API를 사용할 것임 
- keras와 함께 ML모델을 만드는것은 layers를 한 곳에 합치는것이고, data-processing building blocks, Lego bricks를 모으는 것 같은 것임 
- 이 layers은 our input에 우리가 원하는 정도의 수행을 할 수 있는 transformation의 sequence를 특정화하는 걸 허락하게함
- learning algorithm을 single text input과 single classification의 outputs으로 가지게 함으로써 Sequential model API를 사용함으로써 layers의 선형적인 stack을 만들 수 있음
<img src="https://user-images.githubusercontent.com/32586985/77385642-0b62c600-6dcc-11ea-8760-7049888cb046.PNG">

- input layer과 intermediate layers이 서로 다르게 constructed될 것이고, n-gram과 sequence model을 만드는데 달려있음 / 하지만 model type의 irrespective가 last layer에 주어진 문제에 같게 나타날 것임 
