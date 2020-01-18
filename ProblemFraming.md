## 일반적인 ML 문제
- ML은 데이터 세트를 학습시키고 유용한 예측을 하는 과정을 보이고 모델이라고도 부름
- 이 예측으로 이전에 보지 못한 데이터에 대해서도 예측을 할 수 있음
- ML에 대해서 말할 때 supervised와 unsupervised 두 가지 관점으로 볼 수 있는데 이 강좌로 두 개를 아울러서 개념을 익힐 것임

### Supervised Learning이란 무엇인가?
- 이 학습은 라벨이 있는 학습 데이터를 제공하는 학습임을 말함
- 예를 들어 식물학자라고 가정할 경우 구분을 해야할 두 가지의 식물종이 있다고 가정해보자
  - 두 개의 종은 서로 비슷하고 이 식물종을 관찰을 통해 발견한 특정 데이터 세트로 구분한다고 보자
  - 다음은 해당하는 데이터 세트임
  <img src="https://user-images.githubusercontent.com/32586985/71637392-cb38ef00-2c85-11ea-8ae2-87b864a994ab.PNG">
  
  - Leaf width와 leaf length는 features에 해당하고 종은 라벨이 되어있다
  - 실제 식물의 데이터는 더 많은 features를 가지고 있겠지만 지금은 해당 데이터만 다루기로 함
  - features은 측정이나 묘사이며 라벨은 필연적으로 정답이 있다
  - 예를 들어 데이터 세트의 목표가 이 식물은 어떤 종인가에 대한 다른 식물학자의 정답일 수도 있다.
  - 해당 데이터 세트는 오직 4개의 예제만을 가지고 있다(실제로는 더 많을 수 있음)
  
  - leaf width와 leaf length에 대해 그래프를 표현해보면
  <img src="https://user-images.githubusercontent.com/32586985/71637412-371b5780-2c86-11ea-9d39-8bfee328c904.PNG">
  
  
  - supervised learning에서는 학습이라 불리는 알고리즘 과정에서 features와 그와 맞는 라벨을 적용시킴
  - 해당 학습중 알고리즘은 점진적으로 features와 그에 해당하는 라벨에 대한 관계를 결정할 것이다
  - 이러한 과정을 model이라고 부름
  - 종종 머신러닝에서는 모델이 복잡하지만 이 모델은 간단한 직선형태로 표현할 수 있음
  <img src="https://user-images.githubusercontent.com/32586985/72659032-d49fb680-39fc-11ea-941e-8f92682f96b2.PNG">
  
  
  - 모델이 존재한다면 새로 발견한 식물에 대해 구분지을 수 있다
  <img src="https://user-images.githubusercontent.com/32586985/72659051-116bad80-39fd-11ea-99f0-6f7f45175e79.PNG">
  
  - 이것을 종합해보면 supervised learning을 통해서 수학적으로 함수로 표현되는 데이터와 라벨 사이에 패턴을 찾을 수 있음
  - input features가 주어진다면 시스템은 예상되는 output label을 줄 것이고 이것이 학습되는 것을 관찰할 수 있을 것이다
  - ML 시스템은 라벨이 있는 데이터를 통해 패턴을 배울 것이고 이 패턴이 앞으로의 데이터들을 예측하는데 사용될 것임
  
  - Supervised Learning을 실생활에 적용하여 암 예측 모델을 만들고 이 예측 모델이 추후에 새로운 데이터들에 대해서도 예측을 하는 등 사용되고 있음

### Unsupervised Learning
- Unsupervised Learning에서의 목표는 데이터에서 의미있는 패턴을 찾아내는 것임
- 목표를 달성하기 위해 라벨이 되지 않은 데이터에 대해서 학습을 해야함
- 모델은 각각의 데이터에 대해서 어떻게 목록화 해야하는지에 대한 힌트가 없고 오로지 특정 룰에 의해 추론해야만 함
- 아래의 예시와 같이 라벨도 없고 각각의 차이가 무엇인지 모르는 예시가 주어졌을 경우
<img src="https://user-images.githubusercontent.com/32586985/72659141-8095d180-39fe-11ea-8b10-94dd7fdd3949.PNG">

- 단순히 선을 긋는것은 도움이 되지 않음/같은 모양에 대해서 두 부분으로 나눌만한 것이 없음/새로운 접근방식이 필요함
<img src="https://user-images.githubusercontent.com/32586985/72659153-b5a22400-39fe-11ea-8fab-f197d6d6cc3a.PNG">

- 아래와 같이 2개의 clusters가 있는데 이것은 무엇을 나타내는지 알기 매우 어렵다
- 가끔 모델은 원하지 않는 학습을 하고 해당 패턴을 데이터에 찾는데 이를 stereotypes 혹은 bias라고 함
<img src="https://user-images.githubusercontent.com/32586985/72659188-25b0aa00-39ff-11ea-8581-9f0000f0f9cc.PNG">

- 하지만 새로운 데이터를 받을 경우 이와 같이 알려진 cluster를 통해 쉽게 목록화 할 수 있음
- 이것은 매우 일반적인 사례이고 clustering이 유일한 unsupervised learning은 아님
<img src="https://user-images.githubusercontent.com/32586985/72659195-5a246600-39ff-11ea-9e40-104bb8917d5d.PNG">


#### Reinforcement Learning
- ML의 다른 방식으로 reinforcement learning(강화학습)이 있는데 이 학습은 다른 ML의 타입과는 확연히 다름
- 이 방식은 라벨이 있는 예시를 모을 필요도 없음
  - 예를 들어 단순한 비디오 게임이 있고 절대 지지 않게 학습을 시킨다면 단순히 모델을 만들고 (RL에선 agent라고 부름)
  - 게임에서 모델에게 게임오버라는 단어가 뜨지 않게 하고 해당 학습동안 agent는 학습을 수행하는데 reward 함수라고 불리는 것을 통해 보상을 받음
  - RL동안 agent는 매우 빠르게 어떻게 인간을 뛰어넘는지 배울 수 있음
- 데이터가 부족하다면 RL은 새로운 접근법을 만듬/하지만 좋은 reward함수를 만드는 것이 매우 어려움/RL 모델은 supervised보다 덜 안정적이고 예측하기 힘듬
- 그리고 실시간으로 상호작용하는 agent를 만들어 데이터를 생산해내야 함

### Types of ML Problems
<img src="https://user-images.githubusercontent.com/32586985/72659284-afad4280-3a00-11ea-89e7-1692e27e02f4.PNG">

