## Clustering

### What is Clustering?
- 예를들어 음악을 배운다고 할 때 의미있는 그룹과 collections을 볼 것이고 뮤직의 장르를 볼 것임
- 그리고 더 나아가서 장르 속에서 무엇인가가 있는지에 대해서 그리고 년도 순으로 더 깊게 들어가는지 찾아볼 것임
- 머신러닝 또한 같은 개념인데, 머신러닝 시스템에서 가장 중요한 첫 번째 step은 주제(데이터세트)를 이해하는 것임
- clustering이라고 불리는 unlabeled한 예제를 grouping하는 것임
- unlabeled한 예제로써 clustering은 unsupervised machine learning에 의존함
- 만약 예시가 labeled되었다면, clustering은 classification 될 것임 

- Figure 1:Unlabeled examples grouped into three clusters.
<img src="https://user-images.githubusercontent.com/32586985/75082741-7a909480-5558-11ea-8193-ea1fe04cc0d0.PNG">

- 비슷한 예제로 group하기 전, 먼저 similar한 예시를 찾는 것이 필요함
- similarity measure로 불리는 방법을 통해서 examples'의 feature data를 묶은 예제들 사를 metric으로 부터 similarity를 측정할 수 있음
- 예제가 하나 또는 두개의 features로 정의된다면, similarity를 측정하기 쉬울 것임 
- features의 수가 증가한다면, similarity measure은 더욱 복잡해질 것임

### What are the Uses of Clustering?
- clustering이후, 각각의 cluster은 cluster ID라고 불리우는 number가 할당됨
- cluster ID를 통해서 예시를 위해 전체적인 feature set을 condense할 수 있음
- 간단한 cluster ID를 통해서 복잡한 예시를 나타내는 것이 clustering을 강하게 만듬 
- 데이터를 clustering 하는 것은 큰 dataset을 단순화할 수 있음
- 아래의 예시를 통해서 서로다른 features에 대해 group할 수 있음을 암
- Examples 
  - Group stars by brightness
  - Group organisms by genetic information into a taxonomy
  - Group documents by topic
- 머신러닝 시스템은 cluster ID를 사용하여 큰 데이터를 단순화하는 과정을 함
- clustering의 output은 ML 시스템의 downstream을 위해 feature data로 전달됨
- clustering을 generalization, data compression, privacy preservation에 이용할 수 있음

- Generalization
  - cluster의 몇개의 예시에서 missing feature data를 가지고 있다면, missing data를 cluster의 다른 예시로부터 infer 할 수 있음
  - Example
    - 덜 유명한 videos를 video recommendations을 향상시키기 위해서 더 유명한 videos로 clustered할 수 있음

- Data Compression
  - 앞에 말한 것과 같이 cluster에 있는 모든 예시에 대한 feature data는 relevant한 cluster ID로 대체될 수 있음
  - 이러나 대체는 feature data를 단순화하고 storage를 저장함
  - 이러한 benefits은 큰 데이터를 scaled할 때 명확함
  - ML 시스템은 cluster ID를 모든 feature dataset을 대신해서 input으로 사용할 수 있음
  - input data의 복잡성을 재생산하는것은 ML 모델을 단순하게 만들고 학습을 더 빠르게 만듬
  - Examples
    - single Youtube video를 위한 feature data는 아래의 사항을 포함할 수 있음
      - viewer data on location, time, and demographics
      - comment data with timestamps, text, and user IDs
      - video tags
    - Youtube video를 clustering하는 것은 이러한 feature set을 single cluster ID로 바꿔주고, 그렇게 해서 데이터를 compressing함

- Privacy Preservation
  - 특정 유저대신 cluster IDs를 user data에 연관시키고, users를 clustering함으로써 privacy를 preserve할 수 있음
  - 특정 유저와 함께 user data를 관련시키지 않는 것을 확실시 하기 위해서 cluster는 충분한 수의 users를 반드시 group해야함
  - Examples 
    - 모델이 Youtube users의 video history를 추가하기 위한 것을 보면 User ID대신 users를 cluster하고 cluster ID로 대신 의존할 수 있음
    - 그렇게 되면 모델은 특정 유저의 video history로 연관시키지 않고 오로지 많은 유저 그룹으로 나타나는 cluster ID를 사용할 것임

### Clustering Algorithms
- clustering algorithm을 고를 때 algorithm이 데이터세트에 scales하는지 고려해야함
- 머신러닝의 데이터 세트는 수많은 예시가 있고 모든 clustering algorithm이 효율적으로 scale 하는 것은 아님
- 많은 clustering algorithm은 모든 예시의 pair사이에서 similarity를 계산하는데 사용됨
- 이것은 n개의 예시의 수의 제곱만큼 runtime이 증가한다는 것을 의미함 / 복잡도의 개념인 O(n^2)
- O(n^2)알고리즘은 예시가 무수히 많을 때 효율적이진 않음
- O(n)에 대한 복잡도는 k-means algorithm에서 다룰 것임 (algorithm이 n의 선형적인 것을 scale 함)

### Types of Clustering
- Centroid-based Clustering
  - Centroid-based Clustering은 non-hierarchical cluster로 데이터를 구성함
  - k-means는 centroid-based clustering algorithm에서 널리 쓰이는 방법임
  - centroid-based algorithms은 효율적이지만 초기 값과 outliers에 민감함
  <img src="https://user-images.githubusercontent.com/32586985/75084554-212e6280-5564-11ea-93d6-b8446d020616.PNG">
  
- Density-based Clustering
  - Density-based clustering은 cluster안에 높은 예시 밀도 지역을 연결하는 것임
  - 이것은 arbitraty-shaped distributions을 밀집구역으로 연결시킬 수 있게끔 하는것을 사용함
  - 이 알고리즘은 데이터의 densities가 다양하고 dimensions이 높으면 사용하기 어려움
  - 이 알고리즘은 cluster의 outlier을 할당하지 않음
  <img src="https://user-images.githubusercontent.com/32586985/75084610-86825380-5564-11ea-9373-5ecacf0822c7.PNG">
  
- Distribution-based Clustering
  - 이 clustering은 data가 Gaussian distributions같은 distribution으로 구성되어있는 것을 모은다
  - 아래의 그림과 같이 cluster는 3개의 Gaussian distributions로 데이터가 있음
  - distribution's의 중앙으로부터 멀어지면 해당 point가 distribution에 속할 가능성은 감소함
  - bands는 가능성의 감소를 보여줌
  - 만약 데이터에 distribution의 타입을 알지 못한다면, 다른 알고리즘을 사용해야함
  <img src="https://user-images.githubusercontent.com/32586985/75084683-0e685d80-5565-11ea-95cf-a2e8d49764fa.PNG">
  
- Hierarchical Clustering
  - Hierarchical Clustering은 clusters의 트리를 만듬 
  - Hierarchical Clustering은 hierarchical한 데이터와 잘 맞음
  - 아래의 예시와 같이 이 방식은 어떠한 clusters들도 적당한 층에 tree로 cutting함으로써 선택되어지는 장점이 있음
  <img src="https://user-images.githubusercontent.com/32586985/75084720-5d15f780-5565-11ea-975e-0d58b9d1d358.PNG">
  
