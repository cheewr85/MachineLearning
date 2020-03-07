## Retrieval
- embedding model이 있다고 가정해보자, 유저에게 제공한다고 했을 때 어떤 items을 추천할 지 결정할 것인가?
- serve time일 때 query를 주어, 이 중 하나를 시작하면됨
  - matrix factorization model의 경우, query(user) embeddings은 statically하게 알려져 있으나, 시스템은 유저의 embedding matrix로부터 간단하게 보일 수 있음
  - DNN model 같은 경우, 시스템은 네트워크에서 feature vector x를 실행시킴으로써 serve time때 query embedding을 계산함
- query embedding q를 가지고 있으면, embedding space안에 있는 q와 가깝게 item embeddings Vj를 탐색함
- 이것은 nearest neighbor problem임 / 예를들어, similarity score s(q,Vj)에 맞게 top k를 return 할 수 있음
<img src="https://user-images.githubusercontent.com/32586985/76135862-64adc400-606e-11ea-9526-61f342e02e25.png">

- related-item recommendations에서 비슷한 접근법을 사용할 수 있음 
- 예를들어, user가 유튜브 영상을 볼 때, 시스템은 item의 embeddings을 처음으로 볼 수 있고, embedding space에 가까운 다른 item Vj의 embeddings을 볼 수 있음

- Large-scale Retrieval
  - embedding space에서의 nearest neighbors를 계산하기 위해서, 시스템은 모든 가능성 있는 후보군에 대해서 모두 score할 수 있음
  - Exhaustive scoring은 큰 corpora에 대해선 expensive할 수 있지만, 아래의 전략을 통해서 더 효율적으로 만들 수 있음
    - 만약 query embedding이 statically로 알려져있다면, 시스템은 scoring offline을 철저하게 수행할 것이고, 각각의 query의 top 후보군의 list를 미리 계산하거나 storing할 것임 / 이것은 related-item recommendation을 위한 일반적인 practice임
    - approximate nearest neighbors를 사용하라

## Scoring
- candidate genreation이후에 다른 모델은 items의 set을 선택하여 보여주기 위해 생성된 후보를 scores하고 ranks함
- recommendation system은 아래와 같이 다른 sources를 사용해서 다양한 candidate generators를 가지고 있음
<img src="https://user-images.githubusercontent.com/32586985/76136184-21edeb00-6072-11ea-9e7b-afffda650c88.png">
  
- 시스템은 single model에 의해 scored되고 score에 의해 ranked된 candidates의 일반적인 pool로 서로 다른 sources를 결합시킴
- 예를들어, 시스템은 아래 주어진것과 같이 YouTube비디오를 본 것에대한 유저의 가능성을 예측하는 모델을 학습함
  - query features(for example, user watch history, language, country, time)
  - video features(for example, title, tags, video embedding)
- 시스템은 모델의 예측과 일치하는 candidates의 pool에서 videos를 rank할 수 있음

- Why Not Let the Candidate Generator Score?
  - candidate generators이 score을 계산하기 때문에(embedding space에서 similarity measure같이) 이것을 아마도 ranking하는데 또한 사용할 경향도 있음 
  - 하지만,그런 practice는 아래의 이유에 따라 피해야함
    - 몇 개의 systems이 다양한 candidate generators에 의존함 / 이렇게 다른 generators의 scores는 비교할 수가 없음
    - candidates의 작은 pool로 인해서 시스템은 더 많은 features를 사용할 여유가 있고 더 복잡한 모델이 context를 잡는데 더 나을것임

- Choosing an Objective Function for Scoring
  - Introduction to ML Problem Framing에서 배운 것을 생각해보면 ML은 장난스러운 지니와 같이 행동함
  - 제공하는 objective를 배우는 것은 좋지만 원하는바를 위해서 굉장히 조심스러워야 함
  - mischievous quality는 또한 recommendation systems에도 적용됨
  - scoring function을 선택하는 것은 items의 ranking과 recommendations의 quality에 극적인 효과를 줄 수 있음
  - 예를들어
    - Maximize Click Rate
      - 만약 scoring function이 clicks의 optimize하면, 시스템은 click-bait videos를 추천할 것임
      - 이 scoring function은 clicks를 생성하지만 좋은 user experience를 만들진 않음 / Users' interest가 매우 빠르게 사라질 것임
    - Maximize Watch Time
      - 만약 scoring function이 watch time의 optimize하면, 시스템은 매우 긴 videos를 추천할 것임, 그것은 user experience가 좋지 못할 것임
      - 다양한 short watches는 하나의 긴 watch보다 좋음을 알고 있어라
    - Increase Diversity and Maximize Session Watch Time
      - shorter videos를 추천한다면, user engaged를 keep할 것임

- Positional Bias in Scoring
<img src="https://user-images.githubusercontent.com/32586985/76136703-f241e180-6077-11ea-915a-b2a72caf7ff7.png">

- screen의 아래의 위치한 items이 screen의 위에 위치한 items보다 덜 클릭되는 경향이 있음
- 하지만 scoring videos는 시스템이 비디오가 screen의 어디에 위치하여 나타난지 알 수 없음
- 모든 가능한 positions에서 model을 querying하는 것은 expensive함
- 심지어 만약 multiple positions을 querying하는것이 feasible하면 시스템은 다양한 ranking scores사이에서 연속되는 ranking을 찾을 수 없음
- Solution
  - Create position-independent rankings
  - Rank all the candidates as if they are in the top position on the screen

## Re-ranking
- recommendation system의 마지막 단계는, 시스템이 추가적인 criteria나 constraints를 고려하여 candidates를 re-rank할 수 있음
- re-ranking의 접근법 중 하나는 몇개의 candidates를 제거하는 filters를 사용하는 것임
- 예시 / 아래를 적용하여 video recommender를 re-ranking으로 적용하여라
  - 1.Training a separate model that detects whether a video is click-bait
  - 2.Running this model on the candidate list
  - 3.Removing the videos that the model classifiers as click-bait
- 다른 re-ranking 접근법은 score를 ranker로 returned해서 수동적으로 transform하는 것임
  - 예시 / 시스템이 videos를 score를 함수로써 modifying함으로써 re-rank함
    - video age(perhaps to promote fresher content)
    - video length
- Freshness
  - 대부분의 recommendation systems는 최신의 사용된 정보를 통합하는데 목표를 가지고 있음, 현재 user history나 최신의 item같이
  - model을 fresh하게 유지하는 것은 model을 좋은 recommendations으로 만드는 것으로 도움을 줌
  - Solutions
    - training을 가능한만큼 다시 실행시켜 latest training data에 대해서 학습하게 하여라
    - training을 warm-starting 하기를 추천하는데 모델이 scratch로부터 re-learn하지 않게 하기 위해서임
    - warm-starting은 training time을 확실하게 감소시킬 수 있음 
    - 예를들어, matrix factorization에서 items들에 embeddings에 대해서 warm-start하면 모델의 이전 예시에서 나타날 것임
    - matrix factorization models에서 new users를 나타내기 위한 average user를 생성하라
    - 각각의 유저마다 같은 embedding일 필요는 없음 / user features를 기반으로 한 users clusters를 만들 수 있음
    - softmax model이나 two-tower model같은 DNN을 사용하여라 / model이 input에서 feature vector를 가지기 떄문에, 학습하는 동안 보이지 않는 item이나 query를 구동시킬 수 있음
    - feature로써 age를 추가하여라 / 예를들어 YouTube는 video's age를 추가할 수 있거나 그것의 time을 feature로써 보면서 지속할 수 있음

- Diversity
<img src="https://user-images.githubusercontent.com/32586985/76137058-fb34b200-607b-11ea-86ef-a38014945a95.png">

  - 만약 시스템이 항상 query embedding과 가까운 items를 추천한다면, candidates는 각각에 매우 similar한 경향을 보임
  - 이것은 diversity가 부족한 것으로 user experiecne를 bad나 boring을 유발할 수 있음
  - 예를들어 만약 YouTube가 user가 본 영상과 비슷한 영상만 추천해준다면 유저는 빠르게 흥미를 잃을 것임
  - Solutions
    - Train multiple candidate generators using different sources
    - Train multiple rankers using different objective functions
    - Re-rank items based on genre or other metadata to ensure diversity

- Fairness
  - 모델은 모든 유저에게 공평하게 다루어야 함 / 그러므로 모델이 학습데이터로부터 예상치 못한 편향을 학습하게끔 만들어선 안됨
  - Solutions
    - Include diverse perspectives in design and development
    - Train ML models on comprehensive data sets. Add auxiliary data when your data is too sparse (for example, when certain categories are under-represented)
    - Track metrics (for example, accuracy and absolute error) on each demographic to watch for biases
    - Make separate models for underserved groups

## 프로그래밍 실습
- 너무 방대하므로 추후에 필요시 직접 실습할 
