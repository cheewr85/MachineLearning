## Run the Clustering Algorithm
- 머신러닝에서는 데이터세트의 예시가 수백만개인 경우도 있는데 이러한 경우 ML 알고리즘은 효율적으로 scale을 함
- 하지만 많은 clustering 알고리즘은 모든 포인트에 대해서 similarity를 계산할 수 없기 때문에 scale을 사용하지 않음
- 이것은 그들의 런타임이 포인트의 제곱수로 늘어날 수 있다는 걸 의미함
- 따라서 이 코스에서는 k-means에 집중함 왜냐하면 scales는 O(nk)로 되기때문임(k는 cluster의 수임)
- k-means는 점과 그들 cluster의 centroid 사이에 거리를 최소화함으로써 포인트를 k clusters로 그룹화함
- cluster의 centroid는 cluster의 모든 점을 의미함
- k-means는 circular clusters로 발견됨 / 의미적으로 이것은 k-means가 circular distribution의 수로 구성된 데이터를 효과적으로 다루는것을 의미 / 이 distribution에 상응하는 clusters를 찾으려고 시도함 
- 실제로는 data에 outlier를 포함하고 이러한 모델에 맞지 않을수도 있음
- k-means를 시현하기 전에 clusters의 수 k를 선택해야하만 함 / 처음에 k를 가늠해서 시작하고 나중에는 이 숫자를 어떻게 정의할지 생각해야함

- K-means Clustering Algorithm
- k clusters로 데이터를 cluster하기 위해서, k-means는 아래의 과정을 따름
  - Step One
  - 알고리즘이 각각의 cluster를 위해 centroid를 랜덤으로 선택함 / 예시에서는 3개를 고름
  <img src="https://user-images.githubusercontent.com/32586985/75254493-c62a9300-5823-11ea-998c-612777de6fc0.png">
  
  - Step Two
  - 알고리즘이 각각의 포인트에서 가까운 centroid를 k의 inital clusters로 둠
  <img src="https://user-images.githubusercontent.com/32586985/75254635-fd00a900-5823-11ea-9831-82f883277578.png">
  
  - Step Three
  - 모든 cluster에 대해서, 알고리즘은 cluster안에 모든 점에 평균을 구함으로써 centroid를 다시 설정함
  - centroids의 변화는 아래의 그림과 같음 
  - centroids가 변홤에 따라 알고리즘은 가까운 centroid로 점을 재할당함 / Step Four의 그림은 재할당한 새로운 clusters를 보여줌
  <img src="https://user-images.githubusercontent.com/32586985/75254819-49e47f80-5824-11ea-9f48-256bc946d043.png">
  
  - Step Four
  - 알고리즘은 포인트가 clusters를 변화시키는 것을 그만둘때까지 포인트의 할당과 centroids의 계산을 반복함
  - 큰 데이터세트를 clustering할 때, convergence에 도달하기 전까지 다른 기준을 사용하면서 알고리즘을 멈춰야함 
  <img src="https://user-images.githubusercontent.com/32586985/75255013-9fb92780-5824-11ea-8219-bd565d8993fb.png">
  
  - centroid 위치가 초기에 랜덤으로 선택되기 때문에, k-means는 연속적인 실행에서의 확연히 다른 결과를 줄 수 있음
  - 이 문제를 해결하기 위해서, k-means를 여러번 시현하고, best quality metrics를 선택하라 
  - better initial centroid position을 선택했다면 더 상위버전의 k-means가 필요할 것임

## Interpret Results and Adjust Clustering 
- clustering은 unsupervised이기 때문에, 결과를 입증할 수 있는 truth가 없음 
- truth의 부재로 quality를 접근하는데 어려움을 겪음 
- 실제 데이터 세트는 예시에 대한 clusters가 명확하게 떨어지지 않음 (아래의 그림과 같이)
<img src="https://user-images.githubusercontent.com/32586985/75255475-5c12ed80-5825-11ea-8f64-cba5c03b9639.png">

- 실제 데이터 세트는 오히려 아래의 그림과 더욱 같음 / clustering quality를 시각적으로 보기 힘들정도로
<img src="https://user-images.githubusercontent.com/32586985/75255611-8bc1f580-5825-11ea-9a0d-866d825630f8.png">

- 아래의 플로우 차트는 clustering의 quality를 어떻게 체크하는지 요약한 것임 
<img src="https://user-images.githubusercontent.com/32586985/75255694-b14eff00-5825-11ea-829b-74d43b36f05c.png">

- Step One:Quality of Clustering
  - clustering의 quality를 측정하는 것은 clustering이 truth가 없기 때문에 정밀한 과정이 아님
  - 해당 과정은 clustering의 quality를 올리기 위한 과정임 
  - 먼저 clusters가 예상한 것과 같이 보이는지 시각적으로 확인하고 같은 cluster에서 예시에서 similar하게 나타나는지 확인하라
  - 해당 commonly-used metrics를 확인해보아라
    - Cluster cardinality
    - Cluster magnitude
    - Performance of downstream system
  - Cluster cardinality
    - cluster cardinality는 각 cluster마다의 예시의 수임 / 모든 clusters의 cluster cardinality를 보고 주요 outliers에 대해서 clusters를 확인하라
    <img src="https://user-images.githubusercontent.com/32586985/75256317-a3e64480-5826-11ea-8660-7ccecde01f24.png">
    
  - Cluster Magnitude
    - cluster magnitude는 모든 예시로부터 cluster에서의 centroid의 거리의 합임
    - cardinality와 유사하게 clusters를 넘어서, anomalies를 보는데 magnitude가 어떻게 다양한지 확인하라
    <img src="https://user-images.githubusercontent.com/32586985/75256548-ee67c100-5826-11ea-85b2-c3dc23ca4f95.png">
    
  - Magnitude vs.Cardinality
    - higher cluster cardinality가 higher cluster magnitude인 경향이 있다는 것을 알아라
    - clusters는 다른 clusters에 비해서 cardinality가 magnitude가 연관이 없을때 변칙이 됨 
    - 변칙 clusters를 magnitude와 cardinality의 관계를 보면서 확인하라 
    - 아래의 예시를 보면 cluster 0이 변칙임을 알 수 있음
    <img src="https://user-images.githubusercontent.com/32586985/75256826-7057ea00-5827-11ea-89ae-1e2e1c482979.png">

  - Performance of Downstream System
    - clustering output은 종종 downstream ML systems에서 사용되기 때문에 clustering process를 바꿀 때 downstream systems의 성능이 향상되는지 확인하라 
    - downstream 성능의 효과는 clustering의 quality를 위한 실질적인 테스트임
    - 단점은 이러한 확인이 복잡하다는 것임
  
  - Questions to Investigate If Problems are Found
    - 문제를 찾았다면 data preparation과 similarity measure을 확인하고 아래의 질문을 생각하라
      - Is your data scaled?
      - Is your similarity measure correct?
      - Is your algorithm performing semantically meaningful operations on the data?
      - Do your algorithm's assumptions match the data?
      
- Step Two:Performance of the Similarity Measure
  - clustering 알고리즘은 similarity measure에서만 잘 구동 될 것임 / similarity measure가 sensible한 results를 return하게 하여라
  - 간단한 확인 방법은 다른 pair보다 more similar하거나 less similar로 알려진 예시의 pairs를 확인하는 것임
  - 그러면 각각의 에시의 pair에 대해서 similarity measure을 계산할 것임 
  - 더 많은 similar examples에 대한 similarity measure은 더 적은 similar examples에 대한 similarity measure보다 더 확실함
  - similarity measure를 확인하는 spot의 예시는 데이터 세트를 대표해야함
  - similarity measure이 모든 예시를 다루는 것을 확실히해야함
  - 정밀한 검증은 similarity measure이 manual이던 supervised던 데이터세트를 넘어서 일치함을 확인할 수 있음
  - 만약 similarity measure이 몇몇의 예제에 대해서 불일치하면 그 예제는 similar examples로 clustered 된 것이 아님
  - 부정확한 similarities로 예제를 찾는다면, similarity measure은 이러한 예시를 구분하는 feature data를 확인하지 못할 것임
  - similarity measure를 실험함으로써 더 정확한 similarities를 얻을 것음 

- Step Three:Optimum Number of Clusters
  - k-means는 clusters k의 수를 이전의 정하는 것을 필요로 함
  - k의 수를 어떻게 결정해야 할 것인가 / k를 증가시키면서 알고리즘을 실행하라, cluster magnitudes의 합을 확인하라
  - k가 증가할수록, clusters는 작아질 것이고, 총 distance는 감소할 것임 / 이 distance는 clusters의 수와 반대됨 
  - 아래의 그림에서 보듯이 특정 k는 k가 증가함에따라 loss의 마진이 감소함
  - 수학적으로 k의 slope이 거의 -1 (각이 135도 이상)일 경우임
  - 이 가이드라인은 optimum k에 대한 정확한 값은 아니지만 근접한 값임 
  - 아래의 그래프에서 보듯이 optimum k는 거의 11에 근접함 / 만약 좀 더 세분화된 clusters를 바란다면 아래의 가이드에서 높은 k를 선택하라
  <img src="https://user-images.githubusercontent.com/32586985/75258586-0b51c380-582a-11ea-8e13-235de552c327.png">

