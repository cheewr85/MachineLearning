## ML 결정
- 데이터를 자세히 들여다보기 전에 몇 가지 부분에 대해서 깊이 생각을 해야함
  - 1.명확하고 간단하게 시작하라
    - ML 모델이 무엇을 했으면 좋은가?
    - 이 시점에서는 충분히 가치 있는, 확실하게 실질적인 목표를 잡아야함(부정확한 목표가 아닌)
    <img src="https://user-images.githubusercontent.com/32586985/72897921-62193880-3d66-11ea-9257-bc58cbd5f52a.PNG">
    
  - 2.가장 이상적인 결과는 무엇인가?
    - ML 모델이 어느정도 이상적인 결과를 만들어 내야함 / 이 결과는 어떻게 모델을 설정하는지에 따라 달라짐
    - 이미 optimizing한 product의 metrics하는것을 제한하지말고 대신에 product나 service의 큰 목표에 집중하라
    <img src="https://user-images.githubusercontent.com/32586985/72897960-7826f900-3d66-11ea-8eee-b8c058039773.PNG">
    
  - 3.성공과 실패 metrics
    - 수량화를 하여라 / 성공과 실패 metrics는 precision, recall, AUC 등 같은 metrics를 평가하는 것에서 독립적으로 표현될 것
    - 대신 예측할만한 결과에 집중하라 / 시작하기전에 성공 metrics를 설정하고, 모델을 시현하기 전 비용을 들게 하는 것에 대해 방지하라
    <img src="https://user-images.githubusercontent.com/32586985/72898259-16b35a00-3d67-11ea-976b-d4f0d186ce30.PNG">
    
    - metrics가 측정할만한가? / 측정가능한 metrics는 실제 평가를 성공적으로 하기 위한 충분한 정보를 제공함
    - 만일 측정이 안된다면 그것은 유용한 metrics가 아닐 것임
    - 해당 질문을 생각해보아라
      - metrics를 어떻게 측정할 수 있나?
      - metrics를 언제 측정할 수 있나
      - 새로운 ML 시스템이 성공하거나 실패하는 걸 아는데 얼마나 걸리는가?
    - 이상적으로 실패를 빨리하는 것이 나을 수 있다 / 데이터에 대한 신호가 적거나, 데이터가 예측이 안되던가하면 이론이 틀렸다고 생각하게 될 것임
    - 실패를 빨리한다면 이론을 모델이 진행하기 전 보다 일찍 수정할 수 있을 것이고 시간 절약을 할 수 있을 것임
    <img src="https://user-images.githubusercontent.com/32586985/72898877-42830f80-3d68-11ea-94e4-3cbe379c1e5a.PNG">
    
    - 다른 실패 시나리오 / 성공 metrics와 연관되지 않는 것을 보아라
    
  - 4.ML 모델이 어떠한 결과를 생산하기를 바라는가?
    - 아래의 테이블을 보아 어떤 타입의 결과인지 생각해 보아라
    <img src="https://user-images.githubusercontent.com/32586985/72899056-9857b780-3d68-11ea-9467-ac96cb403321.PNG">
    
    - 좋은 output의 요소
      - output은 머신이 생산할 수 있는 확실한 개념과 함께 수량화 될 수 있어야함
      <img src="https://user-images.githubusercontent.com/32586985/72899225-e371ca80-3d68-11ea-80fb-961b8162c560.PNG">
      
      - 만일 직접적인 질문을 통해 정확하게 볼 수 없으면 proxy label을 사용하라 이 라벨은 훌륭한 대체제이긴 함
      - 하지만 이러한 proxy label도 부족한 부분이 있을 수 있지만 충분히 측정가능하고, 확인 가능하면 충분한 예측 신호를 보내므로 가치가 있음
      
      - output은 반드시 이상적인 결과와 연결되어야 함
      - model은 output을 optimize할 것이고 그러므로 output을 확실히 다룰줄 알아야함
      - proxy label은 항상 이상적인 결과를 직접적으로 측정할 수 없기 때문에 필요할 수 밖에 없음
      - 하지만 라벨과 실제 결과 사이에 강한 연관성이 있다면 optimizing이 옳은 방향으로 가고 있다고 확실할 수 있음
      <img src="https://user-images.githubusercontent.com/32586985/72899605-b4a82400-3d69-11ea-8191-173a8ce13b28.PNG">
      
      
      - Training data에 사용하기 위해서 예시 Output을 포함할 수 있는가?
      - 어떻게 어떤 source로 왔는가? / Supervised machine learning은 라벨이 있는 데이터에 의존함
      - 만일 training을 위한 예시 결과를 포함하는것이 어렵다면, 과거에 예제의 응답을 재확인하거나 문제와 목표를 재구성하여 데이터를 모델에 학습시킬 것임 
      
      
      - Output을 사용
      - 모델은 두 가지의 관점으로 예측을 할 수 있음
        - 실제 시간에 맞춰 유저간의 상호작용 즉시 응답함(online)
        - batch로 cached함(offline)
      - ML 모델을 예측이 아닌 결정을 하는데 사용하는 것을 기억하라
      <img src="https://user-images.githubusercontent.com/32586985/72900194-c9d18280-3d6a-11ea-918c-e6adba47098a.PNG">
      
      - 어떻게 적용되는지에 대해서 의사코드를 보며 생각해보아라
      - 이러한 과정을 통해서 모델이 결정을 위한 예측을 할 수 있을 것임
      - 이후 어디에 이런 구조가 코드를 통해서 이용되는지 찾아보아라 이를 위해 다음과 같은 질문을 생각하라
        - 코드가 접근하기 위해 어떠한 데이터가 필요하면서 언제 모델을 불러와야 하는가?
        - latency requirements는 무엇인가? / UI를 lagging 하는 것을 피하기 위해서 빠르게 실행하는게 필요한가? / 혹은 유저와 상관없이 실행하는 과정을 유저가 기다리는가?
        - 이러한 requirement는 모델이 사용하는 features에 영향을 줄 수 있는 이전의 질문들에 대해 답을 찾을 수 있음
        - 이러한 features를 학습 시키는데 언제든지 모델을 불러오는 것과 언제 학습을 시켜야하는지 모를때도 있음
        - 각각의 feature에 대해서 latency requirement안에 있다는 것을 확실히 하여라
        - latency standpoint로부터 비용이 드는 특정 feature는 원격서비스로부터 데이터를 사용하라
      
      - 마지막으로 오래된 데이터를 사용하는 것에 대해서는 유의하라
      - 몇몇 학습 데이터는 종종 오래된 것일 수 있음을 생각하여라                                                                                                                  
      - 실시간 교통상황의 경우 어떤 것이 이용가능한 것인지 모를 수 있다 / 아마 데이터베이스가 오직 30분마다 히스토리를 업데이트 하기 때문에 최신에 데이터를 사용 못하기 때문임
  
  - 5.나쁜 목표들
    - 설정을 매우 적절하게 한다면 ML 시스템은 매우 좋은 목표를 추구하게 될 것임
    - 이와 반대로 그렇게 하지 않는다면 의도치 않은 결과를 제공하게 됨
    - 그러므로 시스템의 목표가 어떻게 문제를 해결하는데 도움이 줄 지에 대해서 유심히 고려해 보아야함
    
  - 6.휴리스틱
    - ML없이 문제를 해결한다고 하면 어떨까?
    - 당장 내일 마감을 위해서 그저 무식하게 코드만을 짤 시간 밖에 없다면 휴리스틱(non-ML solution)을 사용할 수 있음
    <img src="https://user-images.githubusercontent.com/32586985/72901863-dc00f000-3d6d-11ea-95f9-5f06d3495908.PNG">
    
    - 휴리스틱을 연습하는 것이 ML 모델을 확실시 하는데 좋은 신호이고 도움을 줌
    - 가끔은 Non-ML 해답이 ML 해답을 유지하는 것보다 더 쉬울 수 있음 
    
## 문제점 고안
- ML 문제의 framing을 하는 접근법을 제안함
  - 1.Articulate Your Problem
    - classification과 regression의 몇몇의 서브타입이 있는데 이에 상응하는 플로우차트를 사용하여 어떤 서브타입을 쓸 지 결정하라
    - 플로우차트는 ML 문제에 대해서 올바른 언어를 사용하는지 등을 모으는데 도움이 될 것임
    - 문제에 관하여 Classification이나 regression을 통해 플로우차트를 사용하라
    <img src="https://user-images.githubusercontent.com/32586985/72989281-4c247a00-3e31-11ea-8a66-f06aa4220888.PNG">
    
    - 이러한 문제들 중 가장 최선의 frame은
      - Binary classification
      - Unidimensional regression
      - Multi-class single-label classification
      - Multi-class multi-label classification
      - Multidimensional regression
      - Clustering (unsupervised)
      - Other (translation, parsing, bounding box id, etc.)
    - 문제를 framing한 후 모델이 어떤 것을 예측하는지 알 수 있을 것임
    - 이러한 요소들을 종합하여 결과를 내어 문제의 상태를 판단하여라 
    <img src="https://user-images.githubusercontent.com/32586985/72989523-c1904a80-3e31-11ea-8396-75f96d917e1d.PNG">
    
  - 2.Start Simple
    - 먼저 모델링의 일을 단순화하여라 / binary classification이나 unidimensional regression 문제로 접근해보아라
    - 그러면 모델이 가능한 단순하게 사용할 수 있고 단순한 모델일수록 실행하기 쉽고 이해하기 쉽다
    - 제대로 된 full ML 파이프라인이 있다면 간단한 모델로써 이용할 수 있을 것임
    <img src="https://user-images.githubusercontent.com/32586985/72989807-54c98000-3e32-11ea-817d-d4cac3805175.PNG">
    
    - 단순한 모델은 좋은 기반을 제공함 / 단순한 모델은 아무리 복잡한 문제라도 해결하는데 도움을 줌 / 모델이 복잡할수록 훈련시키기 어렵고 느리며 이해하기도 더욱 어렵다 그러므로 단순함을 유지하는 것이 좋음
    <img src="https://user-images.githubusercontent.com/32586985/72989963-a5d97400-3e32-11ea-9de5-f95b7ae3c712.PNG">
    
    - ML은 처음 시작때 많은 결과치를 얻고 처음 데이터가 큰 역할을 함 / 가장 큰 수확은 시작에서 나오고 잘 테스트된 방법을 고르는 것이 과정을 쉽게하게끔 할 수 있는 좋은 사안임
    
  - 3.Identify Your Data Sources
    - 라벨에 대해서 다음의 질문을 고려해봐야함
      - How much labeled data do you have?
      - What is the source of your label?
      - Is your label closely connected to the decision you will be making?
    <img src="https://user-images.githubusercontent.com/32586985/72990241-2a2bf700-3e33-11ea-82e2-2b620626db3d.PNG">
  
  - 4.Design your Data for the Model
    - ML 시스템을 예측(input->output) 하기 위한 데이터를 규정하라
    <img src="https://user-images.githubusercontent.com/32586985/72990458-88f17080-3e33-11ea-9059-5649ce6b23d2.PNG">
    
    - 각 열은 하나의 예측을 위한 데이터로 구성되어 있음 / 예측을 할 수 있는 범위에서 사용가능한 정보만을 내포하고 있음
    - 각각의 input은 스칼라 혹은 1차원 정수,소수,바이트로 되어있음
    - 만일 input이 스칼라나 1차원 리스트가 아니면 데이터의 잘 표현된 것이 무엇인지 고려해야함
    - 예시
    <img src="https://user-images.githubusercontent.com/32586985/72990695-f56c6f80-3e33-11ea-8b92-0943362f5a2b.PNG">
    
  - 5.Determine Where Data Comes From
    - 각각의 열을 구성하기 위해 데이터 파이프라인 만들어지는데 얼마나 많은 일을 하였는지 접근하여라
    - 학습 목적을 위해 예시 결과가 언제 사용되었는지 / 만약 예시 결과를 얻기 힘들다면 결과를 다시 확인하여보고 모델을 위해 다른 결과를 사용했는지 확인해 보아라
    - 사용한 형태에 대해서 예측하는 시점에 input이 사용가능함을 알고 예측 시점에 맞춰 특정 특성을 포함하거나 제외하는 것은 어려울 것임
    <img src="https://user-images.githubusercontent.com/32586985/72991282-11244580-3e35-11ea-97b6-604269ba0c52.PNG">
    
  - 6.Determine Easily Obtained Inputs
    - 1~3개의 input을 쉽게 골라 포함하면 유의미한 결과가 나올것이라 믿을것임
    - 어떠한 input이 휴리스틱 방식을 이용하는 것보다 유용한가?
    - input을 준비하기 전 데이터 파이프라인을 개발하는 cost와 input을 통해 모델에 예상되는 이점은 무엇인지 고려해보아라
    - 간단한 파이프라인에 단일 시스템을 내포한 input에 집중하고 시작은 만들 수 있는 최소한의 인프라로 시작하라
  
  - 7.Ability to Learn
    - ML모델이 학습할 수 있는가? / 학습에 어려움을 유발하는 문제에 관한 측면으로 예시를 보아라
    <img src="https://user-images.githubusercontent.com/32586985/72991706-d078fc00-3e35-11ea-8ec3-753afa99ef4e.PNG">
    
  - 8.Think About Potential Bias
    - 많은 데이터세트는 어떠한 측면으로든 biased함 / 이러한 biases는 학습과 예측을 만드는 것에 영향을 줄 수 있음
    <img src="https://user-images.githubusercontent.com/32586985/72991805-00c09a80-3e36-11ea-9018-82598ed5995c.PNG">
    
