## 데이터 변환
  - Feature engineering은 모델이 학습하는데 있어서 feature를 유용하게 만드는 과정임
  - 이러한 features를 만드는데는 log files이나 다른 sources로부터 찾은 raw data를 변환함으로써 만듬

- Reasons for Data Transformation
  - 1.Mandatory transformations for data compatibility
    - non-numeric features를 numeric하게 변환함 / string을 matrix multiplication을 할 수 없기 때문에, string을 numeric representation으로 변환해야함  
    - input을 fixed size로 resizing함 / Linear models과 feed-forward neural networks는 input nodes들이 fixed number을 가지고 있으므로 input data가 반드시 same size여야함
    - 예를들어, image model은 데이터세트에 맞는 size로 image를 reshape해야함
  - 2.Optional quality transformations that may help the model perform better
    - Tokenization 혹은 lower-casting of text features
    - Normalized numeric features (대부분의 모델은 더 잘 구동함)
    - linear model을 feature space에서 non-linearities가 도입되도록 하게 함
  - quality transformations은 반드시 필요한 것도 아니고 model이 해당사항이 없어도 구동은 가능함
  - 하지만 이러한 techniques를 쓴다면 model이 좀 더 나은 결과를 나올 수 있게함

- Where to Transform?
  - transformations은 disk에 data를 generating하거나 모델 안에서 적용할 수 있음
  - Transforming prior to training
    - 학습하기전에 transformation을 적용하는 방법임 / ML 모델로부터 별도로 존재함(코드가)
    - Pros
      - Computation이 오직 한 번만 수행함
      - Computation으로 전체 데이터 세트를 transformation할 지 볼 수 있음
    - Cons
      - Transformations은 예측하는 시간에 reproduced되야함 / skew를 조심하라!
      - 어떠한 transformation 변화도 data generation을 재구동하는 것이 필요하고, 이것은 반복의 저하를 일으킴
    - skew는 online serving할 때 매우 위험함 / offline serving에서는 training data를 만드는데 코드를 재사용할 수 있음
    - online serving에서는 코드가 데이터세트를 만들고, 코드가 live traffic을 다루는데 어렵고 이러한것으로 쉽게 skew가 나타날 수 있음
  - Transforming within the model
    - 이 방법은 transformation이 model 코드의 일부분이 됨 / 모델은 untransformed data를 input으로 받을 것이고 모델안에서 transform이 일어날 것임
    - Pros
      - 반복이 쉬워짐 / transformation을 바꿔도 같은 데이터 파일을 쓸 수 있음
      - 학습과 예측시간에 대해서 같은 transformation을 보장받음
    - Cons
      - Expensive transform이 model의 latency를 일으킴
      - transformations are per batch.
    - transforming per batch에 대해서 여러가지 고려사항이 존재함
    - feature를 평균적인 값으로 normalize하길 원하는 즉, 만약 feature 값을 mean 0나 standard deviation 1으로 가지는 feature value로 변화하길 원한다고 하면
    - 모델을 transforming할 때, 이러한 normalization은 오직 하나의 데이터 batch에만 적용되고 모든 데이터세트에는 그러지 못할 것임
    - batch안에 평균값을 normalize하거나 평균을 미리 계산하거나 모델에서의 constant를 수정할 수 있음
  - Explore, Clean, and Visualize Your Data
    - transformation하기 전 데이터에 대해서 탐색하고 확인하기 위해서 데이터 세트를 구축하고 모으는데 있어서 몇가지 따라야할 사항이 있음
      - Examine several rows of data
      - Check basic statistics
      - Fix missing numerical entries 
    - data를 자주 visualize하여라
    - 데이터가 가장 기본적인 통계치로 보일 수 있고 graph화 할 수 있다
    - analysis를 제대로 하기 전에 data를 graphically하게 보고 scatter plots이나 histograms를 통해서 확인하라
    - graphs를 보는 것은 pipeline의 시작일 뿐아니라 transformation까지도 통함
    - 시각화는 내가 만든 가정을 주기적으로 확인하는데 도움이 되고 큰 변화에 대해서는 눈에 보일 것임
    
### 숫자 데이터 변환 
- 숫자 데이터 변환
  - numeric data를 transformation하는 두 가지 방식을 적용할 수 있음
  - Normalizing
    - numeric data를 다른 numeric data와 같은 scale로 transforming함
  - Bucketing 
    - numeric data를 categorical data로 transforming하는 것임

- Why Normalize Numeric Fatures?
  - Normalization은 같은 feature안에 서로 다른 value를 가지고 있을 경우 매우 필요함 
  - Normalization이 없다면, 학습은 NaN으로 날라가 버릴 것이고 gradient update는 매우 커질 것임
  - gradient descent를 bounce를 일으키거나 convergence를 느리게 만드는 것을 일으키는 광범위하게 서로 다른 범위에서의 다른 두 가지 features를 가지고 있다면 
  - Adagrad나 Adam같은 Optimizers은 이러한 문제를 일으키는 것을 각 feature를 각각의 효율적인 learning rate를 만듬으로써 막을 것임
  - 하지만 Optimizers은 single feature에서의 광범위한 범위의 값을 저장해줄 순 없으므로 이러한 경우에 반드시 normalize해야함

- 정규화
  - normalization의 목표는 features를 비슷한 scale로 transform하는데에 있음 / 이것은 모델에서 수행능력과 training의 안정성을 향상시킴

- Normalization Techniques at a Glance
  - 자주 사용되는 4가지의 techniques이 있음 / scaling to a range, clipping, log scaling, z-score
  - 아래의 차트는 각각의 normalization technique이 raw한 feature의 영향을 미친 것에 대해서 보여줌
  <img src="https://user-images.githubusercontent.com/32586985/74599290-c065dc80-50c2-11ea-9fce-5708c67fc74b.png">
  
  - Scaling to a range
    - scaling은 자연수의 범위(예.100-900)에서의 floating-point feature value를 standard한 범위(보통 0,1 / 가끔씩 -1 - +1)로 변환하는 것을 의미함
    - 아래의 간단한 공식을 따를수도 있음
    <img src="https://user-images.githubusercontent.com/32586985/74599335-61ed2e00-50c3-11ea-93dd-748069bee128.png">
    
    - 아래의 상황에 놓여진다면 scaling a range를 사용하는 것은 좋은 선택임
      - outlier가 조금이거나 거의 없는 데이터의 대략적인 upper이나 lower bound를 알 경우 
      - 데이터가 범위를 넘어서서 균일하게 distributed 될 때
    - 좋은 예시는 나이로 들 수 있다 대부분의 나이는 0~90사이의 값을 형성하기 때문이고 모든 부분이 사람들의 수에 걸맞게 형성하기 때문에
    - 수입원에 대해서 scaling하는 것은 좋지 않다, 왜냐하면 몇몇 사람들의 수입은 너무 높기 때문임 / income에 대한 linear sclae의 upper bound가 매우 높아서 대부분의 사람들이 scale의 가장 작은 부분으로 압축되어 버릴 것임
  
  - Feature Clipping
    - 만일 데이터가 매우 많은 outlier을 가지고 있다면, 모든 높은(낮은) feature 값의 특정 값을 교정을 해주는 feature clipping을 해야함
    - 예를 들어 temperature values가 40이 넘는 값을 정확히 40으로 clip할 수 있음
    - feature clipping을 normalization 전이던 후던 모두 적용할 수 있음
    - Formula:Set min/max values to avoid outliers
    <img src="https://user-images.githubusercontent.com/32586985/74599458-7b8f7500-50c5-11ea-81ea-676fba789916.png">
    
    - 또 다른 clipping 전략은 z-score을 +-Nσ로 clip하는 것임 (예를들어, limit to +-3σ) / σ은 standard deviation임
    
  - Log Scaling 
    - Log Scaling은 value들을 넓은 범주에서 좁은 범주로 압축하기 위해서 log 계산을 하는것임
    - x'=log(x)
    - Log Scaling은 values들이 handful하게 많은 포인트를 가지고 있을때, 다른 values들은 적은 포인트를 가지고 있을 때 유용함
    - 이러한 데이터의 분배는 power law distribution으로 알려져 있음 
    - 영화평점이 좋은 예시임 / 아래의 예시에서는 대부분의 영화가 매운 낮은 평점을 가지고 있지만, 몇몇은 많은 rating을 가지고 있음 / Log Scaling은 이러한 distribution을 바꾸어 linear model로써의 수행능력을 향상시킴
    <img src="https://user-images.githubusercontent.com/32586985/74599512-8b5b8900-50c6-11ea-962d-ca1acf0ae8eb.png">
    
  - Z-Score
    - Z-score은 mean으로부터 standard deviations의 수를 나타내는 scaling에서의 variation임
    - z-score을 사용함으로써 feature distributions이 mean = 0이나 std = 1로 가질 수 있게 됨을 확실시함
    - 조금의 outliers가 있을때 유용하고 clipping을 할 정도 수준에서는 쓰이지 못함
    - 공식은 아래와 같음
    <img src="https://user-images.githubusercontent.com/32586985/74599545-1fc5eb80-50c7-11ea-9e6e-5afe3dad9496.png">
    <img src="https://user-images.githubusercontent.com/32586985/74599549-38ce9c80-50c7-11ea-862c-a570c4c505be.png">
    
    - z-score는 ~40000정도의 raw values를 -1~+4정도의 범위로 압축시킴
    - 만일 outliers가 extreme한 지 모를 경우, model이 학습하지 않을 feature value를 통해서 z-score를 해보아라 / error가 나오거나 quirk한 결과의 value를 가지고 하는등
  
  - Summary 
    <img src="https://user-images.githubusercontent.com/32586985/74599632-797ae580-50c8-11ea-9780-81f7c62b1db2.png">
    
- 버케팅
  - latitude 예시를 이용하여보면, latitude를 각각의 bucket에서의 housing value에 대해 서로 다른 학습을 할 수 있는 bucket으로 latitude를 나눌 수 있음
  - 이러한 numeric features를 set의 임계값을 이용하여 categorical features로 변환하는 것을 bucketing이라고 부름(binning이라고함)
  - bucketing example에서 boundaries는 동일한 space를 차지함
  <img src="https://user-images.githubusercontent.com/32586985/74599822-bac0c480-50cb-11ea-8489-e2adfb686cfa.png">
  
- Quantile Bucketing
  - 아래의 예시처럼 bucket이 추가된 것에 대해서 해당 bucket은 불필요하게 쓰이는 것으로 보임
  <img src="https://user-images.githubusercontent.com/32586985/74605211-54f32d80-5109-11ea-85aa-ea7994c26fec.png">
  
  - 문제는 매우 동일하게 나뉘어진 buckets이 이러한 distribution을 잘 나누지 못함
  - 해결책은 bucket을 만들때 같은 수의 points를 만드는 것임
  - 이러한 것을 quantile bucketing이라고 불리는데, 해당 예시에서처럼 같은 수를 얻기 위해서 몇몇의 buckets은 좁아지고 특정 buckets은 넓어지기도 함
  <img src="https://user-images.githubusercontent.com/32586985/74605250-c29f5980-5109-11ea-8aab-50da16ae00b3.png">
  
- Bucketing Summary
  - numerical features를 bucketize를 할 때 어떻게 boundaries를 setting 하는지 명확히 해야하고 어떠한 type의 bucketing을 적용할 것인지 명확히 해야함
  - Buckets with equally spaced boundaries 
    - boundaries를 같은 수준의 범위로써 압축하거나 수정함
    - 몇몇 buckets은 많은 points를 포함하거나 오히려 거의 없거나 아예 없을수도 있음
  - Buckets with quantile boundaries
    - 각각의 bucket은 같은 수의 포인트를 가지고 있음
    - boundaries는 fixed되지 않거나 값의 정도가 wide혹은 narrow하게 될 수 있음

- 범주형 데이터 변환
  - 몇몇 features는 ordered된 관계를 가지지 않은 discrete values일 수 있음
  - 이러한 features는 categorical이라고 알려져 있고 이러한 값들은 category라고 부름
  - 이러한 값들을 strings이나 even number로 나타낼 수 있지만 이러한 숫자들을 비교하거나 특정 부분에서 뺼 순 없음 
  - numerical data를 대신해서 categorical data로써 integer value를 포함하는 것이 나타나는 features 볼 수 있음
  - 예를 들면 postal code가 integers로써 값을 표현하는 것
  - 이러한 코드를 만약 numerical한 feature로 나타낸다면 서로 다른 postal code사이에 numeric한 관계를 찾으려고 모델은 작동할 것임
  - postal codes가 categorical data로써 나타난다면 모델이 각각의 postal code로부터 분리된 signal을 찾을 수 있음
  - 만약 데이터 필드의 categories가 작다면, the day of the week나 palette의 색깔과 같이 작다면, 각각의 category의 unique한 feature를 만들 수 있음
  <img src="https://user-images.githubusercontent.com/32586985/74605461-af8d8900-510b-11ea-86d6-e84c48057836.png">
  
  - 모델은 각각의 색에 서로 다른 가중치를 학습시킬 수 있음 / 예를 들면 모델은 red cars를 green cars보다 가중치가 더 부여되게끔 할 수 있음
  - features은 indexed 될 수 있음 / 이러한 mapping을 vocabulary라고 함
  <img src="https://user-images.githubusercontent.com/32586985/74605513-114df300-510c-11ea-9e29-c0a1d1770150.png">
  
- Vocabulary
  - vocabulary에서는 각각의 value가 unique한 feature를 나타냄
  <img src="https://user-images.githubusercontent.com/32586985/74605536-422e2800-510c-11ea-87a3-7a6415bf4829.png">
  
  - 모델은 string에 해당하는 index를 보고 그에 맞는 slot에 1.0이라는 feature vector를 할당하고 다른 slot에는 0.0이라는 featuer vector를 할당함
  <img src="https://user-images.githubusercontent.com/32586985/74605552-8de0d180-510c-11ea-81e9-7c37fce248ca.png">
  
  - sparse representation이라는 것을 인지해야함
    - days of the week를 예를 들어 Friday의 feature vector를 [0,0,0,0,1,0,0]이라고 나타내면 대부분의 ML system은 이 vector가 memory에서 sparse representation으로 나타낸다고 실행할 것임
    - 일반적인 representation에서는 value가 비어있지 않는 list를 만드는데 예를 들면 1.0 value가 [4]의 index에 있는것과 같음
    - 이것은 상대적으로 많은 0들을 저장하는데 메모리를 사용하는것에 조금은 줄일 수 있고 효율적인 matrix multiplication을 하게 함

- Out of Vocab(OOV)
  - categorical data도 outliers를 포함함 / 예를들면 차들의 descriptions을 포함하는 데이터세트가 있다고 할 때 이 데이터 세트의 features 중 하나는 차의 색깔일 수 있음 
  - 차의 흔한 색깔이 이 데이터 세트의 잘 represented되어 있다고 가정하고 그 중 하나를 이러한 서로 다른 색깔이 어떻게 value의 영향을 끼치는지 학습하기 위해서 category했다고 해보자
  - 하지만 이 데이터세트는 차량의 eccentric한 color(mauve,puce,avocado)를 포함한다고 하면 
  - 이러한 색깔들을 separate category로 분류하기보단 Out Of Vocab(OOV)이라고 불리는 category의 한 번에 묶어버릴 수 있음
  - OOV를 사용함으로써 시스템은 rare한 colors에 대해서 학습하는데 시간을 낭비하지 않음

- Hashing
  - 다른 옵션은 사용가능한 index space에 모든 string을 hash하는 것임
  - hashing은 collisions을 종종 일으키지만, 몇몇 주어진 문제에 잘 작동하는 같은 인덱스에 있는 공유된 representation으로 된 categories를 모델이 학습하는데 있어서 사용할 것임
  - hashing은 중요한 term인데, hashing은 collisions때문에 vocabulary를 선택하는 것보다 더 안 좋은 결과를 초래할 수 있음
  - 다른한편으론 hashing은 vocabuly를 모으는것을 필요로하지 않는데 이것은 feature distribution이 바뀌는데 많은 시간이 걸리는 상황에서는 유용할 수 있음
  <img src="https://user-images.githubusercontent.com/32586985/74605899-6808fc00-510f-11ea-9921-79fc44edb53b.png">
  
- Hybrid of Hashing and Vocabulary
  - vocabulary를 hashing하는 것과 같이 혼합된 접근법을 사용할 수 있음 
  - 데이터에서 중요한 categories를 vocabulary로 사용하고 OOV bucket을 다양한 OOV buckets으로 바꾸고 hashing을 사용하여 categories를 buckets으로 바꿈
  - hash buckets에 있는 categories는 index로 나뉘어져야함 / 모델은 좋은 예측은 하지 않을 것임
  - 하지만 몇몇의 memory는 Outside of our vocabulary에 대해서 학습하려는 시도를 할 것임
  <img src="https://user-images.githubusercontent.com/32586985/74605970-efef0600-510f-11ea-8682-c1c4db066a7a.png">
  
- Note about Embeddings
  - embedding이 continuous-valued feature에 대해서 categorical feature를 나타내게 함을 생각해보아라
  - Deep models은 index를 embedding으로부터 변환하게 됨 / 아래 예시 참조
  <img src="https://user-images.githubusercontent.com/32586985/74606012-42c8bd80-5110-11ea-8c0c-1055880dc45e.png">
  
  - 위에서 언급한 다양한 transformations은 disk의 저장되어 구동되지만 embeddings은 다름
  - embeddings을 학습하면 일반적인 data transformation은 아님 (model의 일부임)
  - 다른 model의 가중치를 학습하고 마치 layer of weights와 동등하게 역할을 함
