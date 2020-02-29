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
  - 1.Load and clean data
  <img src="https://user-images.githubusercontent.com/32586985/75608234-19148b00-5b41-11ea-8b89-8d2401ac0e40.PNG">
  
  - 2.Preprocess Data
    - review data를 만들어보자 해당 데이터가 큰 변화가 없다면 review data에 상관관계도 없고 선택의 의미가 없음
    - 몇 개의 함수를 이용해서 data를 시각화해서 본다면 특이점이 보일 것임 / 이와 같이 데이터에 대해서 의문을 한 번쯤은 품어야함
    <img src="https://user-images.githubusercontent.com/32586985/75608313-d43d2400-5b41-11ea-96f9-2c6f337b47b6.PNG">
    
    - rating에 해당하는 distribution을 보아라 / distribution이 어떻게 되는지 보아라
    <img src="https://user-images.githubusercontent.com/32586985/75608346-fd5db480-5b41-11ea-968f-2a01b95136ba.PNG">
    
    - rating에 대한 distribution은 대개 Gaussian distribution임 / 그렇다면 data를 normalize해보자
    ```python
       # its a Gaussian! So, use z-score to normalize the data
       choc_data['rating_norm'] = (choc_data['rating'] - choc_data['rating'].mean()) / choc_data['rating'].std()
    ```
    - cocoa_percent가 어떻게 distribution되는지 어떻게 생산되는지 보아라
    ```python
       sns.distplot(choc_data['cocoa_percent'])
    ```
    <img src="https://user-images.githubusercontent.com/32586985/75608409-7d841a00-5b42-11ea-9372-feebc51423c8.PNG">
    
    - cocoa_percent가 Gaussian distribution에 충분히 근접하므로 데이터를 Normalize해보아라
    ```python
       choc_data['cocoa_percent_norm'] = (
           choc_data['cocoa_percent'] -
           choc_data['cocoa_percent'].mean()) / choc_data['cocoa_percent'].std()
    ```
    - 다음 코드를 통해서 normalization을 확인해보라
    ```python
       choc_data.head()
    ```
    <img src="https://user-images.githubusercontent.com/32586985/75608457-ea97af80-5b42-11ea-9c7d-1ba2368936df.PNG">
    
    - board_originrhk maker_location를 가지고 있어서 similarity 계산시 longitude와 latitude가 필요로함
    - DSPL로 해당 latitude와 longitude를 확인해보자
    <img src="https://user-images.githubusercontent.com/32586985/75608496-42ceb180-5b43-11ea-8f55-11fab79e48fe.PNG">
    
    - 어떻게 distribution을 생성하는지 확인해보아라
    <img src="https://user-images.githubusercontent.com/32586985/75608504-60038000-5b43-11ea-8299-acae9906d636.PNG">
    
    - latitude와 longitude가 특정한 distribution을 따르지 않기 때문에, latitude와 longitude 정보를 quantiles로 변환하라
    - 아래의 코드를 실행시켜 확인해 보아라
    ```python
       numQuantiles = 20
       colsQuantiles = ['maker_lat', 'maker_long', 'origin_lat', 'origin_long']
       
       def createQuantiles(dfColumn, numQuantiles):
         return pd.qcut(dfColumn, numQuantiles, label=False, duplicates='drop')
       
       
       for string in colsQuantiles:
         choc_data[string] = createQuantiles(choc_data[string], numQuantiles)
         
       choc_data.tail()  
    ```
    <img src="https://user-images.githubusercontent.com/32586985/75608568-eddf6b00-5b43-11ea-9a3f-5bfbe4914013.PNG">
    
    - Quantile values를 20까지 / quantile values를 [0,1]로 scaling함으로써 다른 feature data를 같은 scale로 가지고 와라
    ```python
       def minMaxScaler(numArr):
         minx = np.min(numArr)
         maxx = np.max(numArr)
         numArr = (numArr - minx) / (maxx - minx)
         return numArr
         
       for string in colsQuantiles:
         choc_data[string] = minMaxScaler(choc_data[string])
    ```
    - maker와 bean_type의 features은 categorical features임 / categorical features를 one-hor encoding으로 변환하라
    ```python
       # duplicate the "maker" feature since it's removed by one-hot encoding function
       choc_data['maker2'] = choc_data['maker']
       choc_data = pd.get_dumies(choc_data, columns=['maker2'], prefix=['maker'])
       # similarly, duplicate the "bean_type" feature
       choc_data['bean_type2'] = choc_data['bean_type']
       choc_data = pd.get_dummies(choc_data, columns=['bean_type2'], prefix=['bean'])
    ```
    - clustering이후, 결과를 해석할 때, processed feature data를 해석하기 어려울 수 있음
    - original feature data를 새로운 데이터프레임에 두고 나중에 reference를 하면 됨 
    - choc_data만을 processed data로 유지하여라
    ```python
       # Split dataframe into two frames: Original data and data for clustering 
       choc_data_backup = choc_data.loc[:, original_cols].copy(deep=True)
       choc_data.drop(columns=original_cols, inplace=True)
       
       # get_dummies returned ints for one-hot encoding but we want floats so divide by
       # 1.0
       # Note: In the latest version of "get_dummies", you can set "dtype" to float 
       choc_data = choc_data / 1.0
    ```
    - last few records를 본 결과 이전의 chocloate data는 꽤 괜찮은 것 같음 / choc_data는 오직 processed data에서의 columns에만 보이는 것을 생각하라 / columns가 original data를 가지고 있음 (choc_data_backup으로부터 온)
    ```python
       choc_data.tail()
    ```
    <img src="https://user-images.githubusercontent.com/32586985/75608758-dacd9a80-5b45-11ea-893c-31f6867189f5.PNG">
    <img src="https://user-images.githubusercontent.com/32586985/75608759-e3be6c00-5b45-11ea-8dfa-0a2cc2d6b13f.PNG">
    <img src="https://user-images.githubusercontent.com/32586985/75608768-ecaf3d80-5b45-11ea-8ac9-82aab6182fc9.PNG">
    <img src="https://user-images.githubusercontent.com/32586985/75608771-f3d64b80-5b45-11ea-9e47-6ae644983cbf.PNG">
    <img src="https://user-images.githubusercontent.com/32586985/75608778-fb95f000-5b45-11ea-963f-52fe6612c695.PNG">
    <img src="https://user-images.githubusercontent.com/32586985/75608782-03ee2b00-5b46-11ea-8aeb-ff8c4db74351.PNG">
    
