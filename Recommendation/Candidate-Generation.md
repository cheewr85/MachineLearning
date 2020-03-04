## Candidate Generation Overview
- Candidate generation은 recommendation의 첫번째 관문임
- Query를 줌으로써 시스템은 상대적인 candidates의 set을 생산함
- 아래의 테이블은 접근방식에 대해서 보여줌
<img src="https://user-images.githubusercontent.com/32586985/75682565-f6b57580-5cd8-11ea-96a5-56aa4d1cc0f0.png">

- Embedding Space
  - content-based와 collaborative filtering map둘 다 각각의 item과 query가 embedding space E=R^d에 embedding vector를 함
  - embedding space는 낮은 차원이며, item이나 query set의 숨어있는 구조를 찾아냄
  - 비슷한 items으로 유튜브 비디오같이 같은 유저가 본 것에 대해서 embedding space에서 결국 가깝게 귀결되는것이 있음
  - closeness의 개념은 similarity measure로 정의됨

- Similarity Measures
  - similarity measure은 embeddings의 쌍을 가지는 함수 s:E x E -> R, 그리고 그들의 similarity를 측정하여 scalar로 returns함
  - embeddings는 candidate generation을 다음을 바탕으로 사용함 / query embedding을 줌으로써 (q가 E에 속함)
  - 시스템은 (embeddings x가 E에 속함)을 q에 근접한 item으로 봄, 즉 embeddings이 s(q,x)와 높은 similarity를 가지는 것임
  - similarity의 정도를 결정하기 위해서 대부분의 recommendation systems은 하나 혹은 그 이상의 의존함 / cosine, dot product, Euclidean distance
  - Cosine
    - 이것은 두 벡터 사이의 단순한 cosine각임 / s(q,x) = cos(q,x)
  - Dot Product
    - 두 벡터의 내적은 <img src="https://user-images.githubusercontent.com/32586985/75683586-c5d64000-5cda-11ea-9c74-b31a6de21651.png">
    - 그것은 또한 <img src="https://user-images.githubusercontent.com/32586985/75683653-e4d4d200-5cda-11ea-8951-efd292e42518.png">
      - 코사인 각을 product of norms에 곱함
    - 따라서 만약 embeddings이 normalized되면, dot-product와 cosine은 일치할 것임
  - Euclidean distance
    - 이것은 Euclidean space에서의 일반적인 distance임 <img src="https://user-images.githubusercontent.com/32586985/75684163-983dc680-5cdb-11ea-8134-bb35d62227f7.png">
    - 작은 distance는 높은 similarity를 의미함
    - embeddings이 normalized될 때, squared Euclidean distance는 상수에서의 dot-product와 일치함 
    - 해당 케이스로 인해 <img src="https://user-images.githubusercontent.com/32586985/75684310-e5219d00-5cdb-11ea-9825-29c833bbd0df.png">

- Comparing Similarity Measures
<img src="https://user-images.githubusercontent.com/32586985/75684398-1306e180-5cdc-11ea-8c9e-f6da05e22f1b.png">
  
  - 위의 예시를 생각해보자, black vector는 query embedding을 표현하고 다른 세 개의 embedding vectors는 candidate items을 나타냄
  - similarity measure를 사용하는 것에 따라, items의 랭킹이 다를 수 있음
  - 이미지를 사용하여 세 개의 similarity measure(cosine, dot product, Euclidean distance)를 사용하며 item ranking을 결정해보아라
    - Item A가 큰 norm을 가지고 있고, dot-product에 따르면 높게 랭크되어 있음 
    - Item C는 query와 함께 가장 작은 각을 가지고 있고 cosine similarity에 따르면 가장 첫번째로 ranked되어 있음
    - Item B는 query와 물리적으로 가까우므로 Euclidean distance가 높음
    <img src="https://user-images.githubusercontent.com/32586985/75684821-e3a4a480-5cdc-11ea-8e45-8eb91ab8d8bf.png">
    
- Which Similarity Measure to Choose?
  - cosine과 비교했을 때, dot product의 similarity는 embedding의 norm에 더 민감함
  - 즉 embedding에서의 norm이 크면 similarity가 높고 item이 더욱 더 recommended될 것임 
  - 이것은 recommendations에 영향을 끼칠것임
    - 학습 세트에서 매우 빈번하게 나타는 아이템은 large norm을 embedding하는 경향이 있음
    - popularity information을 capturing하는 것을 원한다면 dot product를 선호해야만 할 것임
    - 하지만 신경쓰지 않는다면, popular items는 recommendations을 압도할 것임
    - 실제로는 item의 norm을 덜 강조하는 similarity measures의 다른 variants를 사용할 것임
    - 예를 들어 <img src="https://user-images.githubusercontent.com/32586985/75685325-c02e2980-5cdd-11ea-9028-579ef8aff13d.png"> 
    - some <img src="https://user-images.githubusercontent.com/32586985/75685404-d9cf7100-5cdd-11ea-9802-0967be28ce98.png">
    
    - 거의 나타나지 않은 Items는 학습하는동안 거의 updated되지 않을 것임 
    - 결과적으로 만약 그것이 큰 norm으로 initialized 된다면, 시스템은 더 상대적인 items를 넘어서 rare items를 추천해줄 것임 
    - 이 문제를 피하기 위해서 embedding initialization을 조심하고 정확한 regularization을 사용하여라
    
## Content-based Filtering
- Content-based filtering은 유저의 이전의 actions과 했던 feedback을 기반으로 유저가 좋아하는 것과 유사한 다른 items를 추천하는 item features를 사용함
- content-based filtering을 증명하기 위해서, 구글 플레이스토어의 some features를 hand-engineer를 보자
- 아래의 figure은 각 열은 앱을 나타내고 각 행은 feature를 나타내는 feature matrix를 보여줌 
- Features는 앱 제작자, 혹은 그 다른 것들의 categories를 포함할 수 있음 
- 단순화하기 위해서, 이 feature matrix가 binary하다고 가정해보자 non-zero value는 app이 feature를 가지고 있음을 의미함

- user가 같은 feature space에 나타낼수도 있음 / 몇개의 user-related features는 user에 의해서 확연히 제공될 수 있음
- 예를들어 user가 profile에서 Entertainment apps를 선택했다고 해보자 / 다른 features는 그들이 이전의 설치한 앱을 기반으로 implicit할 수 있음
- 예를들어 유저는 Science R US가 배포한 다른 앱을 설치할 수 있음

- 모델은 유저와 연관된 items를 추천할 수 있음 / 이렇게 하기 위해서, similarity metric을 먼저 골라야함
- 그때 system을 이러한 similarity metric에 따라 각각의 candidate item을 score하는 것으로 set up할 수 있어야만 함
- recommendations는 이 유저에 대해 특정화되었고 모델은 다른 유저에 대한 어떠한 정보도 사용하지 않는다는 것을 알아야함
<img src="https://user-images.githubusercontent.com/32586985/75686667-fa002f80-5cdf-11ea-92bc-578f57199d4f.png">

- Using Dot Product as a Similarity Measure
  - 유저가 embedding x인 그리고 앱이 embedding y인 binary vectors의 경우가 있다고 고려해보자
  - <img src="https://user-images.githubusercontent.com/32586985/75686882-48153300-5ce0-11ea-81af-be0787be2647.png"> 
    - 로 인해 feature가 x,y가 1부터 합까지로 contributes함 
  - 다른 한편으로는 <x,y>가 두 vectors에 동시에 active한 features의 수라고 할 수 있음
  - high dot product는 더 common features를 가르키고 따라서 higher similarity임을 알 수 있음
  
- Advantages
  - 유저에 대해서 recommendations이 특정화하기 때문에, 다른유저에 대한 어떠한 데이터도 필요로 하지 않음
  - 이것은 많은 수의 유저에 대해서 더 쉽게 scale하게 만듬
  - 모델은 user의 특정 interests를 잡을 수 있음 / 그리고 아주 적은 다른 유저들이 흥미있어 하는 niche items를 추천할 수 있음
- Disadvantages
  - items의 feature를 나태내기 때문에 몇몇 extentsms hand-engineered되고 이 기술은 많은 domain 지식을 필요로함
  - 그러므로 모델은 hadn-engineered features에서만 좋음
  - 모델은 오직 유저의 현재의 intersets에 기반한 recommendations을 만듬 / 다른 한편으로는 모델은 유저의 현재의 interests의 확장하는데 능력에 한계가 있음    
    
## Collaborative Filtering
- content-based filtering의 한계를 보완하기 위해서 collaborative filtering은 추천을 제공하기 위해서 users와 items사에어서 similarities를 동시에 사용함
- 이것은 serendipitous recommendations을 제공함 / 즉, collaborative filtering models은 user A에게 item을 similar user B의 interests에 기반하여 추천해줌
- 더욱이 embeddings은 features의 hand-engineering에 의존하지 않고 자동으로 학습함

- A Movie Recommendation Example
  - 아래의 feedback matrix로 구성된 training data가 된 movie recommendation system을 생각해보자
    - Each row represents a user
    - Each column represents an item (a movie)
  - movie에 대한 feedback은 하나 혹은 두개의 카테고리로 빠짐
    - Explicit:users specify how much they liked a particular movie by providing a numerical rating
    - Implicit:If a user watches a movie, the system infers that the user is interested.
  - 단순화하기 위해서, feedback matrix를 binary로 가정함 / 즉, 1은 영화의 흥미를 나타냄
  - user가 홈페이지를 방문할 때, 시스템은 아래 두 가지를 바탕으로 추천함
    - similarity to movies the user has liked in the past
    - movies that similar users liked
  - 묘사를 위해서 아래의 예시를 보자
  <img src="https://user-images.githubusercontent.com/32586985/75843119-9e37c280-5e15-11ea-9c61-b198924304c6.PNG">
  
- 1D Embedding
  - 각각의 영화를 아이들을 위한 영화일 경우(negative values) 혹은 어른들을 위한 영화일 경우(positive values)로 표현한 [-1,1] scalar로 할당한다고 해보자
  - 또한 각각의 유저를 유저의 interest가 아이들의 영화의 있다면(-1의 근접) 혹은 어른들의 영화의 있다면(+1의 근접) 그것을 묘사한 각각의 유저를 [-1,1]의 scalar로 할당한다고 해보자
  - movie embedding과 user embedding의 product는 유저가 좋아하는 것을 기대하는 영화를 위해 높아야 함
  <img src="https://user-images.githubusercontent.com/32586985/75843378-5b2a1f00-5e16-11ea-9d92-d781c4237446.PNG">
  
  - 아래의 다이어그램을 보면, 각각의 체크마크는 특정 유저가 보는 영화를 나타냄 
  - 3,4번째 유저는 이 feature에 설명된 것과 맞게 선호함 / 3번째 유저는 아이들을 위한 영화를 선호하고 4번째 유저는 어른들을 위한 영화를 선호함 / 하지만 1,2번째 유저의 선호도는 single feature에 의해 잘 설명되지 않음
  <img src="https://user-images.githubusercontent.com/32586985/75843513-ad6b4000-5e16-11ea-8a1e-508b780f9fd8.PNG">
  
- 2D Embedding
  - 모든 유저의 선호도를 설명하기에는 하나의 feature로는 부족함 / 이 문제를 해결하기 위해서 second feature를 추가해보자 
  - 각각의 영화를 blockbuster 혹은 arthouse 영화로 정도를 정해보자
  - 두 번째 feature에서는 각가의 영화를 아래의 2차원 embedding으로 나타낼 수 있음
  <img src="https://user-images.githubusercontent.com/32586985/75843667-0509ab80-5e17-11ea-9adb-4deab31bf51e.PNG">
  
  - feedback matrix에서 잘 설명하기 위해서 같은 embedding space에 유저들을 다시 배치하였음
  - 각각의 (user,item) pair는, 0이 아니라면 유저가 영화를 봤을 때 user embedding과 item embedding의 내적이 1에 근접할 것임
  <img src="https://user-images.githubusercontent.com/32586985/75843754-4e59fb00-5e17-11ea-8e94-1a258a4a0b29.PNG">
  
- 해당 예제에서는 embeddings을 hand-engineered하였음 / 실제로는 embeddings은 자동으로 학습되며 그것이 collaborative filtering models의 힘임
- 이 접근법의 collaborative nature은 모델이 embeddings를 학습할 때 정확함 
- movies를 위한 embeddings vectors가 교정된다고 가정해보자 / 모델은 유저들을 위해 그들의 선호도를 잘 설명해줄 embeddings vector를 학습할 것임 / 결과적으로 유사한 선호도와 함께한 유저의 embeddings은 같이 close 될 것임 
- 이와 비슷하게 유저를 위한 embeddings가 교정된다면, feedback matrix를 잘 설명해줄 movie embeddings를 학습해줄 것임
- 결과적으로 similar users가 좋아한 embeddings movies는 embedding space에서 가까워질 것임

- Matrix Factorization
  - Matrix factorization은 간단한 embedding model임 
  - feedback matrix <img src="https://user-images.githubusercontent.com/32586985/75857863-55decb80-5e3a-11ea-9f6f-31076530982c.PNG">, m은 유저(혹은 queries)의 수이고, n은 items의 수라고 하자
    - A user embedding matrix <img src="https://user-images.githubusercontent.com/32586985/75857940-86266a00-5e3a-11ea-92a6-ee86a93771e0.PNG">, row i가 user i를 위한 embedding일 때    
    - An item embedding matrix <img src="https://user-images.githubusercontent.com/32586985/75858020-af46fa80-5e3a-11ea-8bca-b633bffb0e86.PNG">, row j가 item j를 위한 embedding일 때     
  <img src="https://user-images.githubusercontent.com/32586985/75858092-d00f5000-5e3a-11ea-9c56-ef7a498ba332.PNG">
  
    - embeddings은 UxV^T의 곱 같은 것이 feedback matrix A의 가장 좋은 접근방식임
    - UxV^T의 전체의 (i,j)를 보는 것은 <Ui, Vj>, user i, item j의 embeddings의 내적 값으로 간단히 표현가능함 (A(i,j))와 근접하길 바라는)

- Choosing the Objective Function
  - 하나의 직관적인 objective function은 squared distance임 / 이렇게 하기 위해서 확인되는 entries에서의 모든 pairs를 넘은 squared errors의 합을 최소화해야함
  <img src="https://user-images.githubusercontent.com/32586985/75858528-c2a69580-5e3b-11ea-8722-3eb5499101b5.PNG">
  
  - 이 objective function에서는 오직 확인된 pairs(i,j)만을 합을 만들 수 밖에 없음 / 즉 feedback matrix에 non-zero values를 넘어선 값만 가능
  - 하지만 오직 값 하나만을 가지고 합을 구하는 것은 좋은 idea가 아님 / all ones의 matrix는 최소한의 손실을 가질 것이고 모델은 효과적인 recommendations을 만들어 생산할 수 없을 것이고 좋지 않게 구동될 것임
  <img src="https://user-images.githubusercontent.com/32586985/75858916-7019a900-5e3c-11ea-8c1b-c65123304b8b.PNG">
  
  - 아마도 unobeserved된 값을 zero로 할 것이고 matrix안에서 모든 entries로 합을 구할 것임
  - 이것은 A와 근사값인 UxV^T사이에 frobenius distance의 squared를 최소화한 것과 일치함
  <img src="https://user-images.githubusercontent.com/32586985/75859106-c4bd2400-5e3c-11ea-871a-8242595dac05.PNG">
  
  - 이 이차식을 matrix의 Singular Value Decomposition (SVD)를 통해서 해결할 수 있을 것임
  - 하지만 SVD는 좋은 풀이법은 아님/ 실제 적용을 할 때 matrix A는 매우 희소할 수 있기 떄문임
  - 예를들어, Youtube의 모든 비디오를 특정 유저가 본 모든 비디오와 비교한다고 했을 때 / UxV^T는 0에 근접할 것이고, poor generalization performance를 보일 것임
  - 이와 반대로 Weighted Matrix Factorization은 objective를 아래의 두 합으로 decompose할 것임
    - A sum over observed entries
    - A sum over unobserved entries (treated as zeroes)
    <img src="https://user-images.githubusercontent.com/32586985/75859409-4c0a9780-5e3d-11ea-8eff-f90b5be6887f.PNG">
    
    - w0는 two terms의 가중하는 hyperparameter고 하나나 다른 것들로 objective가 dominated되지 않음
    - 이러한 hyperparameter를 설정하는 것은 매우 중요함
    <img src="https://user-images.githubusercontent.com/32586985/75865318-b5db6f00-5e46-11ea-8f52-7b532a81cecf.PNG">
    
      - w(i,j)는 query i와 item j의 frequency function임

- Minimizing the Objective Function
  - 일반적인 알고리즘은 objective function을 minimize하는 것을 포함함
    - Stochastic gradient descent (SGD)는 loss functions을 minimize하는 일반적인 방법임
    - Weighted Alternating Least Squares (WALS)는 특별한 objective의 specialized되어있음
  - objective는 두 matrices U와 V 각각의 quadratic임 / WALS는 embeddings을 무작위로 initializing하지만, 두 개로 대체할 수 있음
    - Fixing U and solving for V
    - Fixing V and solving for U
  - 각각의 stage는 (linear system의 해답을 통해) 정확히 풀릴 수 있고 distributed되게 할 수 있음 
  - 이 기술은 converge를 보장함 왜냐하면 각각의 과정이 loss를 감소시키는걸 보장하기 때문임

- SGD vs. WALS
  - SGD와 WALS는 장단점이 있음 / 아래를 참고하여 비교하여라
  - SGD
    - advantages
      - Very flexible / can use other loss functions
      - Can be parallelized
    - disadvantages
      - Slower / does not converge as quickly
      - Harder to handle the unobserved entries (need to use negative sampling or gravity)
  - WALS
    - advantages
      - Can be parallelized
      - Converges faster than SGD
      - Easier to handle unobserved entries
    - disadvantage
      - Reliant on Loss Squares only
      
- Collaborative Filtering Advantages & Disadvantages
  - Advantages
    - No domain knowledge necessary
      - 도메인 지식을 알 필요가 없음 / embeddings가 자동으로 학습되기 때문에
    - Serendipity
      - 모델이 유저의 새로운 interests를 발견하는걸 도와줌
      - ML system이 user에게 주어진 item에 대해서 흥미가 있는지 알 수없으나 모델은 similar users가 그 item으로 흥미를 가졌기 때문에 여전히 추천할 것임
    - Great starting point
      - 몇 번의 확장으로, system은 matrix factorization model을 학습하는데 feedback matrix만을 필요로 할 것임 
      - 특별히 시스템은 contextual한 features를 필요로 하진 않아함 / 실제로는 이것은 다양한 후보 생성 중에 하나로 쓰일 것임
  - Disadvantages
    - Cannot handle fresh items
      - 주어진 pair (user,item)을 위한 모델의 예측은 해당하는 embeddings에서의 내적임
      - 그래서 만약 item이 학습하는동안 보내지지 않는다면, 시스템은 embedding을 위해서 생성할 수 없을 것이며 이 item을 가지고 모델을 query할 수 없을 것임
      - 이 주제는 cold-start problem이라고 종종 불려왔음 / 하지만 아래의 기술은 cold-start problem의 확장된 생각을 가져오게 할 수 있음
        - Projection in WALS
          - 학습에서 보여지지 않은 새로운 item i0가 주어지면 시스템은 유저에 대한 몇 번의 상호작용후, 시스템은 embedding v(i0)를 이 item을 위해 전체 모델을 재학습하지 않고 쉽게 계산할 수 있음
          - 시스템은 아래의 방정식이나 가중치버전을 풀 수 있음 
          <img src="https://user-images.githubusercontent.com/32586985/75867438-f8eb1180-5e49-11ea-8c4c-b3e278197b25.PNG">
          
          - 이 방정식은 WALS에서의 하나의 iteration과 같음 / user embeddings은 계속해서 교정될 것이고 시스템은 item i0의 embedding을 위해 해결될 것임 / 새로운 유저에게도 똑같이 적용될 것임
        - Heuristics to generate embeddings of fresh items
          - 만약 시스템이 interactions을 가지고 있지 않다면, 시스템은 같은 범주로부터의 items의 embeddings를 평균을 냄으로써 embeddings를 예상할 수 있음 (같은 업로더(Youtube)같은)
    - Hard to include side features for query/item
      - query나 item ID를 넘어선 어떠한 features를 side feauters라고 함 
      - movie recommendations에선 side features는 country나 age를 포함할 수 있음 
      - 이용가능한 side features를 포함하는 것은 모델의 quality를 증가시킴
      - 비록 WALS에서의 side features를 포함시키는 것이 쉽지 않지만, WALS를 generalization을 하는 것은 이것을 가능하게 함
      - WALS를 generalize하기 위해서 block matrix <img src="https://user-images.githubusercontent.com/32586985/75868075-e3c2b280-5e4a-11ea-8f3c-8504fcbd21e6.PNG">를 정의함으로써, features가 함께 있는 input matrix를 
        - Block (0,0)은 feedback matrix A의 원본임
        - Block (0,1)은 user features에 multi-hot encoding임 
        - Block (1,0)은 item features에 multi-hot encoding임 
        
- 프로그래밍 실습



## Deep Neural Network Models
- matrix factorization의 한계가 있음
  - side features를 사용하는 것에 어려움(즉, query ID/item ID를 넘어선 어떠한 features) / 결과적으로 모델은 학습세트에서 나타내는 오직 user나 item만을 queried할 수 있음
  - recommendation과의 관련성 popular items을 모두를 위해서 추천하는 경향이 있음, 특히 similarity measure로써 내적을 사용할 때
  - 특정 유저의 interests를 확인하는 것이 더 나음
- DNN 모델은 matrix factorization의 이러한 한계를 다룰 수 있음 
- DNN은 query features와 item features를 쉽게 통합시킬 수 있음(network input layer의 유용성때문에) / user의 특정 interests를 잡아내는데 도움을 줌 / 그리고 recommendation과의 연관성을 향상시킴

- Softmax DNN for Recommendation
  - DNN모델로 가능한 것은 softmax이며 이것은 문제를 multiclass prediction problem으로써 다룰 수 있음
    - input은 user query임
    - output은 corpus에서의 items의 수와 같은 사이즈의 가능한 벡터이며, 각각의 item을 interact하는 가능성을 나타내며, 가능성은 Youtube 비디오를 보거나 클릭하는 가능성임
  
  - Input 
    - DNN의 input은 아래를 포함함
      - dense features (for example, watch time and time since last watch)
      - sparse features (for example, watch history and country)
    - matrix factorization 접근법과는 다르게, age나 country같은 side features를 추가할 수 있음 / input vector를 x로 표기함
    <img src="https://user-images.githubusercontent.com/32586985/75870157-eecb1200-5e4d-11ea-90e0-7c3f457bb0d8.PNG">
    
  - Model Architecture
    - model architecture은 complexity를 결정하거나 모델의 expressivity를 결정함
    - hidden layers와 non-linear activation functions을 추가함으로써, 모델은 데이터에서 더 복잡한 관계를 파악할 수 있음
    - 하지만 parametres의 수가 증가함에따라 model을 학습하기 어렵게 만들거나 serve를 더 힘들게 만듬 
    - 마지막 hidden layer의 output을 <img src="https://user-images.githubusercontent.com/32586985/75870439-51241280-5e4e-11ea-8550-07fb20a533ca.PNG">로 표기함     
    <img src="https://user-images.githubusercontent.com/32586985/75870515-64cf7900-5e4e-11ea-9874-3d618a67b411.PNG">
  
  - Softmax Output:Predicted Probability Distribution
    - 모델은 last layer의 ouput <img src="https://user-images.githubusercontent.com/32586985/75870619-93e5ea80-5e4e-11ea-9429-61a83ae3e48c.PNG">,probability distribution가 <img src="https://user-images.githubusercontent.com/32586985/75870728-c0016b80-5e4e-11ea-8502-90c45452e20c.PNG">인 softmax layer를 통해서 maps함
    <img src="https://user-images.githubusercontent.com/32586985/75870848-f0e1a080-5e4e-11ea-8f78-2664794867ab.PNG">
    
    - softmax layer은 scores <img src="https://user-images.githubusercontent.com/32586985/75870922-166eaa00-5e4f-11ea-8f9c-ee635646b965.PNG"> 의 vector를 probability distribution으로 maps함
    <img src="https://user-images.githubusercontent.com/32586985/75871015-4322c180-5e4f-11ea-8bc1-5df6d380df7a.PNG">
    
