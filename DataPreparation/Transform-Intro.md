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
    - 
