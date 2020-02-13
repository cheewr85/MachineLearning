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
  - 
