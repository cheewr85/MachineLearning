## Recommendations
- What and Why?
  - What are Recommendations?
  - ML-based recommendation model은 비디오와 앱이 너가 좋아하는 것과 얼마나 similar한지 결정해줌 그리고 recommendation을 함
  - 두 가지 종류가 일반적임
    - Home page recommendations
    - related item recommendations
  - Homepage Recommendations
    - 그들이 잘 알려진 interests에 기반한 유저의 특성화된 기술임
    - 모든 유저가 서로 다른 recommendations을 봄
    - Google Play Apps 홈페이지를 본다면 당신은 아래와 같은 그림을 볼 것임
    <img src="https://user-images.githubusercontent.com/32586985/75679928-dd5dfa80-5cd3-11ea-867a-001bb7acd734.png">
    
  - Related Item Recommendations
    - 이름에서 볼 수 있듯이 related items은 특별한 item에 대해서 비슷한 recommendations을 하는 것임
    - Google Play Apps의 예시에서처럼 유저가 math app을 페이지에서 보게 된다면 다른 math app이나 science apps과 같이 그와 관련된 패널의 앱을 보게 될 것임  
  
  - Why Recommendations?
    - 추천시스템은 유저들에게 large corpora에서 설득력있는 컨텐츠를 찾는데 도움을 줌
    - 예를들어 구글 플레이스토어는 수백만개의 앱을 추천하는 반면에 유튜브는 수십억개의 영상을 제공함 / 매일 새로운 앱과 비디오가 추가됨
    - 어떻게 유저가 새로운 설득력있는 컨텐츠를 찾게끔하는가? / 하나는 컨텐츠의 접근해서 search를 사용하는 것이다
    - recommendation engine은 items을 보여줄 수 있고 그들 스스로가 서치를 통해서 보진 않음

- Terminology
- 몇 개의 단어에 대해서 알아볼 필요가 있음
  - Items (also known as documents)
    - system recommends라고 제목을 붙일 수 있음 / 구글 플레이스토어에서는 인스톨할 앱이 item이고 유튜브에서는 비디오가 item임
  - Query (also known as context)
    - 시스템의 정보는 recommendations을 만드는데 사용됨 / Queries는 아래의 결합이라고 할 수 있음
      - user information
        - the id of the user
        - items that users previously interacted with
      - additional context
        - time of day
        - the user's device
  - Embedding
    - discrete set으로부터 vector space로 mapping한 것을 embedding space라고 부름
    - 많은 recommendation system이 queries나 items로 나타나는 정확한 embedding의 학습으로 의존함 

- Recommendation Systems Overview
  - 아래의 해당하는 요소가 추천시스템의 일반적인 구조 중 하나로 구성되어 있음
    - candidate generation
    - scoring 
    - re-ranking
    <img src="https://user-images.githubusercontent.com/32586985/75680781-9cff7c00-5cd5-11ea-8d29-0913e2c05208.png">
    
    - Candidate Generation
      - 처음 시작에는 system은 huge corpus로부터 시작하여 그리고 좀 더 작은 candidates의 subset을 생산함
      - 예를들어, YouTube에서의 candidate generator은 수 백만개의 videos를 수백 수천개로 줄임
      - 모델은 주어진 큰 사이즈의 corpus를 가지고 queries를 평가함
      - 주어진 모델은 다양한 candidate generators를 제공할 것이며 각각의 다른 candidate의 subset이 nominating될 것임
    - Scoring
      - 그 다음은, 다른 모델은 유저에게 보여주기 위해서 item의 set을 선택하기 위해서 candidates를 scores하거나 ranks함
      - 이 모델은 items의 상대적으로 작은 subset을 평가하기 때문에, 시스템은 additional queries에 의존하여 좀 더 정교한 모델을 사용함
    - Re-ranking
      - 최종적으로, 시스템은 마지막 ranking을 위해서 추가적인 제약을 고려해야만함
      - 예를들어, 시스템은 유저들이 확실하게 싫어하는 items을 제거하고 새로운 컨텐츠의 범위로 boosts함
      - Re-ranking은 다양성, 신선성과 공평성을 확실히 하는데 도움을 줄 수 있음
      - 이 class의 course를 넘어서 이 각각의 stages에서 논의를 할 것이고 유튜브같이 다른 recommendation system으로부터 예시를 줄 것임
      
