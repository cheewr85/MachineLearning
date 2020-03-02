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
    
