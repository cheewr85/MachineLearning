## K-Means Adavantages and Disadvantages
- Advantages of k-means
  - Relatively simple to implement
  - Scales to large data sets
  - Guarantees convergence
  - Can warm-start the positions of centroids
  - Easily adapts to new examples
  - Generalizes to clusters of different shapes and sizes, such as elliptical clusters

- k-means Generalization
  - clusters가 densities와 sizes가 다르면 무슨일이 일어날까? / Figure1을 보면서 왼쪽의 클러스터들과 k-means가 있는 오른쪽에 cluster를 비교해봐라
  - 해당 비교를 통해서 k-means가 특정 데이터세트에서는 제대로 되지 않는 것을 볼 수 있음
  - Figure1에서 imbalanced한 clusters를 자연스럽게 cluster하기 위해서는 k-means를 적용시킬 수 있음
  <"img src="https://user-images.githubusercontent.com/32586985/75602064-0714f700-5b05-11ea-93ab-5ef2bf1824ab.png">
  
  - Figure2에서 lines은 k-means를 generalizing한 후 cluster boundaries임을 보여줌
  <img src="https://user-images.githubusercontent.com/32586985/75602180-455ee600-5b06-11ea-944f-1720556b99df.png">
  
    - Left plot:No generalization, resulting in a non-intuitive cluster boundary
    - Center plot:Allow different cluster widths, resulting in more intuitive clusters of different sizes
    - Right plot:Besides different cluster widths, allow different widths per dimensions, resulting in elliptical instead of spherical clusters, improving the result

- Disadvantages of k-means
  - Choosing k manually
    - Loss vs. Clusters plot을 optimal(k)를 찾는데 사용하여라
  - Being dependent on initial values
    - low k로 인해 k-means를 서로 다른 initial values를 가지고 몇 번 구동하고 좋은 결과를 고름으로써 이러한 dependence를 완화할 것임
    - k가 증가한다면 initial centroids의 더 나은 값을 고르는 k-means의 상위 버전이 필요함
  - Clustering data of varying sizes and density
    - k-means는 clusters에서 sizes나 density가 다양할 때 데이터를 clustering하는데 어려움을 겪음
    - 이러한 데이터를 cluster하는데 있어서 Advantage section에서 나타낸 것과 같이 k-means를 generalize해야할 필요가 있음
  - Clustering outliers
    - Centroids는 outliers에 의해 dragged 될 수 있고, outliers는 ignored되는 대신 그 자체의 cluster를 가질 수 있음
    - clustering하기 전에 outliers를 제거하거나 clipping을 하는 걸 고려해야함
  - Scaling with number of dimensions 
    - dimensions의 수가 증가한다면, distance를 기반한 similarity measure은 주어진 어떠한 예시 사이에서의 상수값을 합침
    - feature data에 PCA를 사용하거나 clustering algorithm을 수정하는 spectral clustering을 사용함으로써 dimensionality를 제거하라
    
- Curse of Dimensionality and Spectral Clustering 
  - 이 plot에서는 예시가 감소하고 dimensions의 수가 증가하는 것 사이에 distance의 mean의 일반적인 편차 비율을 나타내는 것임
  - 이 convergence에서는 k-means는 예시를 분류하는데 덜 효과적이라고 보여짐 
  - 높은 차원의 데이터에서의 이 부정적인 결과는 dimensionality의 cures라고 불림
  <img src="https://user-images.githubusercontent.com/32586985/75602644-95d84280-5b0a-11ea-936f-eb5f51a14319.png">
  
  - Spectral clustering은 dimensionality의 curse를 pre-clustering을 추가함으로써 피할 수 있음
    - 1.PCA를 사용하여 feature data의 dimensionality를 제거하라
    - 2.모든 데이터 포인트를 lower-dimensional subspace에 project하라
    - 3.선택한 알고리즘을 사용함으로써 subspace에서의 데이터를 cluster하여라
  - spectral clustering은 clustering algorithm에 분리된 것이 아니라 어떠한 clustering algorithm에서도 사용할 수 있은 pre-clustering step임
  

## 프로그래밍 실습
- Clustering with Manual Similarity Measure

    
