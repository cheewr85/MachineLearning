## 데이터세트 구축
- 데이터 세트를 구축하기 위해 아래와 같이 따라야함
  - 1.raw data를 모음
  - 2.feature과 label의 source를 명시하라
  - 3.샘플링 전략을 선택하라
  - 4.데이터를 구분하라
- 해당 과정은 ML 문제에 대해서 어떻게 framed을 했는지에 달려 있음  

### 데이터 수집
- 데이터 세트의 크기 및 품질
  - 데이터 세트의 품질을 측정하고 향상시키며, 유용한 결과를 나오게 하기 위해서는 문제를 해결하는 것에 달려있음

- 데이터 세트의 크기
  - 가장 직관적인 방식으로는 모델에 대해서 최소한 학습할 수 있는 정도보다 그 이상의 예시를 측정하여 학습하는 것임
  - 간단한 모델의 많은 데이터 세트가 일반적임 
  - 많은의 기준은 프로젝트가 무엇이냐에 따라 달려 있음 
  - 이러한 데이터 세트의 대략적인 크기를 비교하라
  <img src="https://user-images.githubusercontent.com/32586985/74428091-b1511580-4e9b-11ea-99e1-9af5a3dddcad.PNG">
  
- 데이터 세트의 품질
  - 많은 데이터를 쓰지 않는 것은 좋은 데이터가 아닌것과 같이 품질 역시 같음
  - 품질을 측정하는 것은 어렵기 때문에 경험적인 접근법이난 최고의 결과를 나타내는 옵션을 선택하는 것을 고려해봐야 함
  - 이러한 마인드셋으로 품질이 좋은 데이터가 문제를 해결하는데 있어서 성공적일 것임
  - 다른 한편으론 데이터가 좋기 위해서는 몇 가지 선행되는 과제가 필요할 것임
  - 데이터를 모으는데 있어서 품질이라는 정의에 좀 더 집중하는 것이 도움이 될 것임
  - better-performing model로 여겨지는 품질의 측면이 있음
    - reliability
    - feature representation
    - minimizing skew
    
- Reliability    
  - Reliability는 당신의 데이터를 얼마나 신뢰하는지에 대한 척도를 말함
  - reliable한 데이터 세트를 통해 학습한 모델은 unreliable한 데이터 세트로 학습한 모델보다 더 유용한 예측을 생산해냄
  - reliability를 측정하는데 있어서 반드시 결정해야 할 것이 있음
    - 얼마나 label의 에러가 있는가? / 만일 데이터를 사람이 labeled 했으면 가끔씩 실수가 있을 수 있음
    - features들이 noisy한가? / GPS 측정시 데이터 세트에 대한 모든 noise를 확인할 수 없음
    - 데이터가 정확하게 문제를 filter하는가? / 시스템을 구축하였지만 발전시키는 과정에서 혼선이 생겨 원하는 결과와 정반대가 될 수 있음
  - unreliable하게 데이터를 만드는 것
    - values를 빠뜨리는 것
    - 예시를 똑같이 쓰는 것
    - 좋지 않은 label
    - 좋지 않은 feature values
    
- Feature Representation
  - Representation은 유용한 features에 대한 데이터를 mapping함 / 해당 사항에 대해서 생각해보아야함
    - 모델에 대해 데이터가 어떻게 보이는가?
    - 수 적인 values에 대해서 일반화 할 수 있는가?
    - 이상점에 대해서 조절할 수 있는가?

- Training versus Prediction
  - 오프라인으로 좋은 결과치를 얻었다면 이것은 live 경험에는 도움이 되지 않음
  - 이러한 문제는 training/serving skew를 나타냄
  - 즉, metrics에서의 training time과 serving time에 계산의 차이로 나타남
  - 이러한 차이는 커보이지 않더라도 결과에 대해서는 큰 차이가 보임 
  - 항상 모델에 예측 시간을 어떠한 데이터가 사용가능한지 고려해야함
  - 학습하는 동안 serving을 하는데 이용가능한 특징을 사용하고 serving traffic이 나타나는 training set에 대해서 명확히하라
  - 예측하려는만큼 학습시켜라! / 즉 training task가 prediction task에 잘 맞을수록 더욱 더 ML 시스템이 잘 구현될 것임

- 로그 결합
  - training set를 모을 때 가끔씩 데이터의 다양한 소스를 모음

- Types of Logs
  - 다음에 해당하는 input data를 사용할 것임
    - transactional logs
    - attribute data
    - aggregate statistics
  
  - Transactional logs 
    - 특정한 event를 기록함 / 쿼리를 만드는 ip 주소를 기록하던지, 쿼리가 만들어질때 생성된 데이터와 시간을 만듬
    - Transactional events는 specific event와 같음
  - Attribute data
    - 정보의 특정 상황을 포함함 / 예를 들면 user demographics, search history at time of query
    - Attribute data는 정확한 시간에 event나 사건을 집중하지 않지만, 예측을 하는데 있어서 여전히 유용함
    - attribute data는 데이터 타입의 일종임
    - Attribute data와 transactional logs는 서로 연관되어 있고, 몇몇의 transactional logs를 조합함으로써 attribute data의 종류를 만들 수 있음
    - 이러한 상황에서 유저를 위한 single attribute를 만들기 위해 많은 transactional logs를 볼 수 있음
  - Aggregate statistics
    - attribute를 다양한 transactional logs를 통해서 만들 수 있음 / 예를 들어 유저 쿼리의 빈도, 특정 광고의 평균 클릭률

- Joining Log Sources
  - 각각의 로그 타입은 각기 다른 location에 있음
  - 머신러닝 모델을 위해서 데이터를 모을 때 데이터 세트를 만들기 위한 서로 다른 소스를 한꺼번에 모아야함
  - 예시 / transaction timestamp를 사용해서 쿼리의 어느 순간의 기록 이력을 선택하는것
  - attribute data를 보는데 있어서 event timestamp를 이용하는 것은 매우 중요함

- Prediction Data Sources - Online vs. Offline
  - online과 offline을 고르는 것은 시스템이 data를 모으는데 영향을 줌
  - Online
    - Latency가 주요 쟁점이고, 시스템이 input을 빠르게 생산해내야함
  - Offline
    - 계산하는데는 제한이 없을 것이지만, training data를 생산하는데 복잡하게 구동됨 

- 라벨 소스

- Direct vs. Derived Labels
  - 머신러닝은 label이 잘 정의되어 있을 수록 쉬워짐 / best label은 내가 예측하기 원하는 것에 대한 direct한 label임
  - 예를 들어 Taylor Swift팬인 유저를 예측하기 위해서 direct label은 User is a Taylor Swift fan
  - 가장 간단한 방법은 유튜브에서 Taylor Swift를 보는 유저를 찾는 것임
  - 유튜브로 Taylor Swift를 시청한 유저는 derived label이라고 할 수 있음 / 왜냐하면 예측하고자 하는 것을 directly하게 측정하지 않았기 때문임
  - derived label이 과연 믿을만한 지표일 것인가? / 모델은 derived label과 추구하고자 하는 예측에 사이에서 좋은 연결고리를 가지고 있을 것임 

- Label Sources
  - 모델의 결과물은 Event이거나 Attribute 둘 중 하나임
  - 이러한 결과는 두 개의 type의 label을 나타낼 수 있음
    - Direct label for Events / 예시:유저가 가장 많이 클릭한 연구 결과를 보았는가?
    - Direct label for Attributes / 예시:광고주가 다음주에 X달러 이상 소비할 것인가?

- Direct Labels for Events
  - events에서 direct labels은 대체로 직관적임 / event동안 라벨을 사용할 때 유저의 행동에 대해서 log를 할 수 있기 때문임
  - events를 labeling할 때 다음과 같은 질문을 생각하라
    - logs를 어떻게 설계하였는가?
    - logs에서 고려해야할 event는 무엇인가?
  - 예를들어 시스템 로그는 유저가 search result에 클릭한 것인가 유저가 search를 만든것인가?
  - 클릭 로그가 있다면 클릭이 없다면 이상적인 결과에 대해서 볼 수 없다는 것을 깨달을 것임
  - event를 원하게끔 하기위해선 log가 필요함 / 그래서 유저가 본 top search result에 대한 모든 사례를 봐야함

- Direct Labels for Attributes
  - 라벨이 만약 광고주가 다음주에 X달러 이상 소비할 것인가?라면
  - 일반적으로 이전에 데이터를 사용하여 그 다음날 무슨일이 일어날지 예측할 것임
  - 아래의 예시를 참고하자면
  <img src="https://user-images.githubusercontent.com/32586985/74444175-ce93dd00-4eb7-11ea-9cfa-596cd27b6234.png">
  
  - seasonality하고 cyclical한 효과를 기억하라 / 예를 들어 광고주는 일주 이상의 시간을 쓸 것임
  - 이러한 이유로 14일정도의 예시 자료를 쓰거나 모델이 yearly effect하게 배울 수 있는 특성으로써 데이터를 사용할 것임
  - event data를 고르는데 있어서 cyclical이나 seasonal effects를 피하거나 이러한 효과들을 염두해 두어라

- Direct Labels Need Logs of Past Behavior 
  - 이전의 케이스에서 true result에 대한 데이터가 필요함을 인지함
  - 광고주가 얼마나 돈을 쓰는지, 어떠한 유저가 Taylor Swift 비디오를 시청하는지에 대해서 필요함
  - supervised machine learning을 이용하기 위한 historical data가 필요함
  - ML은 과거의 일어난 사례들을 통해서 예측을 만듬 / 만일 과거의 log가 없다면 그것을 얻어야함

- What if You Don't Have Data to Log?
  - 만약 아직 product가 존재하지 않는다면, 어떠한 log를 위한 데이터도 필요없음
  - 이러한 경우, 다음의 절차를 생각해야함
    - 첫 시작시 heuristic을 사용하고, logged data를 바탕으로 시스템을 학습시킴
    - 유사한 문제에 대한 log를 사용해서 시스템을 bootstrap하여라
    - 문제를 마무리하며 사람이 데이터를 생산함

- Why Use Human Labeled Data?
  - Human-labeled data에 장점과 단점이 있음
  - Pros
    - 사람이 하는 것은 좀 더 광범위한 부분의 문제를 수행할 수 있음
    - 데이터는 문제의 정의하는데 있어서 명확히 할 수 있게끔 함
  - Cons
    - 특정 domain에서 데이터가 expensive함
    - 좋은 데이터는 대체로 여러번의 반복을 요구함

- Improving Quality
  - 항상 사람이 시행한 일에 대해서 확인하라
  - 데이터를 직접 보는것은 어떠한 데이터든지 간에 좋은 훈련이 됨 


### 샘플링 및 분할
- 샘플링
  - 데이터를 충분히 모으기도 어렵지만 가끔은 데이터가 너무 많아서 학습 예제를 선별해야할 때가 있음
  - 선별하는 작업에 있어서는 순전히 문제에 달려있음 / 우리가 무엇을 예측하고 싶은지?, 어떠한 features를 원하는지?
  - 이전 쿼리의 feature를 사용하기 위해, session level에서 sample이 필요로함 / sessions은 쿼리의 결과를 포함하기 때문에
  - 이전의 user behavior에 해당하는 feature를 사용하기 위해서 user level의 샘플이 필요함

- Filtering for PII(Personally Identifiable Information)
  - 만약 데이터가 PII를 포함한다면, 데이터를 filter하는게 필요함 / 이 방식은 infrequent features를 제거하는 걸 필요로함
  - 이러한 filtering은 어느정도의 손실이 일어날 수 있고 차이가 있을 수 있음
  - 하지만 이런 filtering은 유용함, infrequent features를 학습하는 것은 어렵기 때문에 
  - 데이터 세트가 head 쿼리로부터 편향될 수 있다는 점도 인지하는 것이 중요함
  - serving time으로 인해 예시를 좋지 않게 serving할 수 있기 때문에 이로 인해 training data로부터 예시가 많이 filter 될 수 있음
  - 비록 이러한 skew는 피할 수 없지만, 이러한 분석을 할 때 알아두고 있어야함

- 불균형 데이터
  - class의 부분이 차이가 나는 classification 데이터세트를 불균형이라고 함
  - class는 majority classes라고 불리는 데이터 세트의 큰 부분이 있고 이것은 minority class라고 하는 작은 부분을 만듬

- imbalanced는 무엇으로 측정하는가? / mild ~ extreme의 정도의 범위가 있음
<img src="https://user-images.githubusercontent.com/32586985/74534662-90f68900-4f77-11ea-9d2e-089bf390d43e.PNG">

  
  - 불균형 데이터 세트를 classificartion 문제로 다룰 때 특별한 샘플링 기법을 적용해야함
  <img src="https://user-images.githubusercontent.com/32586985/74534761-cbf8bc80-4f77-11ea-8686-673f98eb82a9.PNG">
  
  - true distribution에서는 0.5%정도는 positive가 나와야함
  - 이 그래프가 문제가 되는 이유는 몇몇의 positive는 negative와 연관되어 있고 학습모델은 대부분을 negative 예시에 사용하고 충분한 positive 예시를 학습하지 않음
  - 만약 불균형 데이터 세트를 다룬다면 맨 먼저, true distribution을 학습하여라
  - 모델이 잘 구동되고 학습된다면 제대로 된 것이지만 그렇지 않다면 downsampling이나 upweighting technique를 사용해야함

- Downsampling and Upweighting
  - 불균형 데이터를 효과적으로 다루는 방법은 downsample과 upweight가 있음
  - Downsampling
    - 불균형한 majority class 예시에서의 low subset을 학습하는 것을 의미함
  - Upweighting
    - downsample한 요소와 같은 downsampled class를 가중치를 부여하여 예시에 추가하는 것을 의미함

  - Step1:Downsample the majority class
    - fraud한 데이터 세트의 예시를 다시 고려해봄, negatives를 1/10로 가져감으로써 20정도의 factor로 downsample할 수 있음
    - 이렇게 된다면 모델을 학습하기 더 수월하게 데이터의 10%정도가 positive하게 됨
    <img src="https://user-images.githubusercontent.com/32586985/74535513-4fff7400-4f79-11ea-88fa-2b7ceac76da4.PNG">
  
  - Step2:Upweight the downsampled class
    - 마지막으로 downsampled class를 가중치를 부여하여 예시에 추가함
    <img src="https://user-images.githubusercontent.com/32586985/74535591-76bdaa80-4f79-11ea-95f5-d16c4f286d2e.PNG">
    
  - weight라는 개념은 학습하는동안 개별의 예시를 더욱 의미있게 측정할 수 있음을 의미함
  - 다음 공식과 같음 / {example weight} = {original example weight} x {downsampling factor}
  
- Why Downsample and Upweight?
  - downsampling한 후에 가중치를 예시에 부여하고 추가하는 것이 생소할 수 있지만, 모델의 minority class를 통해서 모델을 향상시킬 수 있음
  - 왜 majority를 upweight하는가? / 다음과 같은 결론이 나옴
  - Faster convergence 
    - 학습하는동안 minority class를 더 접한다면 모델이 converge를 빠르게 하는데 도움이 됨 
  - Disk space
    - majority class를 큰 weight으로 몇 몇의 예시에 적용함으로써 disk space를 덜 사용함 
    - 이러한 절약은 minority class를 위해 더 많은 disk space를 사용할 수 있음 
    - 이러한 class를 통해서 우리는 더 넓은 범위의 예시와 더 큰 수를 구할 수 있음
  - Calibration 
    - Upweighting은 확실하게 모델을 교정할 수 있음 / 결과값은 여전히 probabilities하게 해석될 수 있음 

- 데이터 분할 예시
  - 필요로한 데이터와 샘플링을 모은 후에는 데이터를 training sets, validation sets, testing sets로 분할함

- When Random Splitting isn't the Best Approach
  - random splitting이 ML 문제에 대해서 좋은 접근법이지만, 항상 좋은 해결책은 아님
  - 만일 비슷한 예시로 cluster된 예시가 있는 데이터 세트가 있다고 하자 / 여기서 뉴스의 주제의 text에 따라 토픽을 classify하는 모델을 원한다고 하자 / 왜 random split이 문제가 될까?
  <img src="https://user-images.githubusercontent.com/32586985/74596707-415aaf00-5096-11ea-9fc7-64bd631d4356.PNG">
  
  - News stories가 cluster에 나타날 것임 / 같은 topic에 다양한 stroies가 동시에 생길 것임
  - 만약 데이터를 randomly하게 나눈다면 test set와 training set에 same stories 포함될 것임
  - 실제로는 이러한 방식으로 작동되지 않을 것임 / 모든 stories가 동시에 나오기 때문에 이러한 방식으로 split하는 것은 skew를 유발함
  
  <img src="https://user-images.githubusercontent.com/32586985/74596808-92b76e00-5097-11ea-9bc9-160a64940ed3.PNG">
  
  - 이러한 문제를 해결하기위한 접근법은 데이터를 story가 발행된 시점을 기반으로 나누는 것임 / story가 발행된 날일것임
  - 같은 날 발행된 것으로 stories를 나누는 결과는 same split으로 나뉘게 될 것임
  
  <img src="https://user-images.githubusercontent.com/32586985/74596843-2852fd80-5098-11ea-8515-126379407817.PNG">
  
  - 많은 stories를 다루게 된다면, day를 넘어서서 비율이 나뉘어 질 것임
  - 하지만 실제로는 이러한 stories는 news cycle에서의 two days를 넘어서 split될 것임
  - 대안적으로, 데이터를 어떠한 overlap이 발생하지 않도록 하기 위해서 특정한 범위에서 cutoff를 할 것임
  - 예를들면, April의 해당하는 달을 stories를 train시킨다고 할 때, week gap으로 overlap을 방지하기 위해서 May에서 2번째 주를 test set으로 사용할 것임 
  
- 데이터 분할
  - news story example에서 보였듯이, random split 자체는 좋은 접근법이 아님
  - online system에서의 데이터를 split하는 자주 쓰이는 technique는 다음과 같음
    - Collect 30 days of data
    - Train on data from Days 1-29
    - Evaluate on data from Day 30
  - Online systems에서는 training data가 serving data보다 older하고, 이러한 technique은 validation set이 training과 serving사이에 lag가 되는 것을 보여줌
  - 하지만 time-based splits은 많은 데이터세트가 있을 경우 잘 작동함
  - 만일 프로젝트의 데이터가 많지 않다면, 이러한 distributions은 다른 training, validation, testing을 나타내게끔 할 것임
  - 데이터는 3개의 authors 중 하나로 나뉠 것이고, 데이터는 3개의 main groups으로 들어갈 것임
  - random split에서 보여줬듯이, 데이터의 각각의 그룹은 training, evaluation, testing set으로 나뉨
  - 그러므로 모델은 정보를 학습하는데 있어서 prediction time이 필요하지 않게됨
  - 이러한 문제들은 데이터를 series data든, 특정 기준으로 clustered한 것이든 그룹화하는데 있어서 언제든지 생길 수 있음 
  - Domain knowledge가 어떻게 데이터를 split하는지 알려주게 됨 
  - representative한 데이터의 split을 설계할 때 데이터가 무엇을 represent하는지 고려하라
  - 데이터를 split하는데 golden rule이 존재함 / testing task가 production task와 가능한 밀접하게 match되어야 함 

- 임의 선택

- Practical Considerations 
  - 
