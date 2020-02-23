## Clustering Workflow 
- 데이터를 cluster를 하기 위해 아래의 방식을 따라야함
  - 1.Prepare data
  - 2.Create similarity metric
  - 3.Run clustering algorithm
  - 4.Interpret results and adjust your clustering 
  <img src="https://user-images.githubusercontent.com/32586985/75084757-ad8d5500-5565-11ea-8177-8c67d4d20548.PNG">
  
- Prepare Data
  - ML 문제에서는 반드시 normalize, scale, feature data를 transform해야함
  - 하지만 clustering동안, prepared data가 예시들 사이에서 정확하게 similarity를 계산하게 둬야함
- Create Similarity Metric
  - clustering algorithm이 데이터를 group하기 전에 예시가 얼마나 similar한 pairs를 가지고 있는지 알아야함
  - similarity metric을 만듬으로써 예시들 사이에 similarity를 수량화해야함
  - similarity metric을 만드는 것은 데이터에 대한 이해와 어떻게 feature로부터 similarity가 derive됐는지 알아야함
- Run Clustering Algorithm
  - clustering algorithm은 similarity metric을 data를 cluster하는 사용함
- Interpret Results and Adjust 
  - clustering output의 quality를 확인하는 것은 반복적이여하고 exploratory해야함 / clustering은 output을 증명하는데 truth가 부족하기 때문에            
  - cluster-level과 example-level의 기대를 하지 않고 결과를 확인해야함
  - 결과를 향상시키는 것은 이전의 과정이 어떻게 clustering에 영향을 미쳤는지 보면서 반복적으로 실험하는 것을 필요로함
  
## Prepare Data
- clustering에서는 두 개의 예시에서 similarity를 이러한 예시들의 모든 feature data를 numeric value로 합침으로써 계산할 수 있음
- 합쳐진 feature data는 같은 scale을 가지고 있는 data를 필요로 함 

- Normalizing Data
  - data를 normalizing함으로써 같은 scale으 다양한 features로 데이터를 transform할 수 있음
  - normalization은 가장 흔한 data distribution인 Gaussian distribution 생산하는 것에 가장 적합함 
  - quantiles와 비교했을 때, normalization은 계산하는데 데이터가 덜 필요함
  - 데이터를 normalize하는 것은 z-score를 이용하여 다음과 같음
  <img src="https://user-images.githubusercontent.com/32586985/75104396-ea337c00-564b-11ea-9683-b4d662b8aec0.png">
  
  - Normalization이 있는 것과 없는 것의 similarity 예시를 보자
  - 아래의 예시에서 보듯이, red가 yellow보다 blue에 더 similar하게 보임
  - 하지만 x축과 y축의 feature가 scale이 같지 않음 / 그러므로 측정한 similarity는 artifact거나 unscaled data일 수 있음
  - z-score를 사용한 normalization이후 모든 features는 같은 scale을 가지고 있음 / 여기서는 red가 yellow의 similar하게 보임
  - 따라서 데이터를 normalizing한 이후에는 similarity를 좀 더 정확하게 계산할 수 있음
  <img src="https://user-images.githubusercontent.com/32586985/75104462-85c4ec80-564c-11ea-945d-5f944ba72602.png">
  
  - 요약 / normalization을 적용하려면
    - 데이터가 Gaussian distribution이어야 함
    - 데이터세트가 quantiles를 만들기에 데이터가 충분치 않을 때

- Using the Log Transform
  - 가끔씩 데이터 세트가 데이터의 끝부분에서 상승하는 power law distribution 보이기도함 / 아래의 예시에선 red가 yellow에 더 similar함
  <img src="https://user-images.githubusercontent.com/32586985/75104493-eeac6480-564c-11ea-9fbd-7e804bc834af.png">
  
  - log transform을 사용하여 power-law distribution을 생성할 수 있음 
  - 아래의 예시에서 처럼 log transform은 smoother distribution을 만들고, red가 blue에 더 similar함
  <img src="https://user-images.githubusercontent.com/32586985/75104508-27e4d480-564d-11ea-8a03-1c2085e33abf.png">
  
- Using Quantiles
  - Normalization과 log transform은 특별한 data distribution을 다룸 / 만일 Gaussian과 power-law distribution의 데이터가 아니라면? / 어떠한 data distribution에도 적용할 일반적인 접근방식이 있을까?
  <img src="https://user-images.githubusercontent.com/32586985/75104538-ab9ec100-564d-11ea-8976-a843bb7c7a70.png">
  
  - 만일 두 개의 예시가 서로에 대해 적은 예시만 같고 있다면 두 예시의 그들의 값에 대한 무관하게 similar하고 
  - 만일 두 개의 예시가 서로에 대해 많은 예시를 같고 있다면 두 예시의 similar은 줄어듬
  - 두 개의 예시에 대한 similarity는 예시의 수가 줄어들수록 높아짐
  
  - 데이터를 Normalizing하는것은 normalization이 linear한 transform이기때문에 data distribution을 재생산함
  - log transform을 적용하는 것은 어떻게 similarity가 작용하는지 직접적으로 반영되지 않음, 아래의 예시와 같이
  <img src="https://user-images.githubusercontent.com/32586985/75104575-5fa04c00-564e-11ea-9382-1b90caa175b7.png">
  
  - 대신에 같은 수의 예시를 포함한 각각의 interval로 data를 나눈다면, 이러한 interval boundaries를 quantiles이라고 함 
  - 데이터를 quantiles로 바꾸는 것은 다음과 같은 방식을 따름
    - 1.Decide the number of intervals
    - 2.Define intervals such that each interval has an equal number of examples.
    - 3.Replace each example by the index of the interval it falls in
    - 4.Bring the indexes to same range as other feature data by scaling the index values to [0,1]
    <img src="https://user-images.githubusercontent.com/32586985/75104623-cc1b4b00-564e-11ea-997e-435e4297f81e.png">
    
  - 데이터를 quantiles로 변환한 후 두 예시의 similarity는 두 예시 사이에서의 예시들의 수의 반대적인 비율이 나옴
  - 수학적으로는
  <img src="https://user-images.githubusercontent.com/32586985/75104643-0258ca80-564f-11ea-9183-ebb149ae2544.png">
  
  - quantiles는 data를 transform하는데 최고의 default한 선택지다
  - quantile를 만드는데 있어서 data distribution이 강조하는 믿을만한 indicators가 필요함 / 많은 양의 데이터 필요
  - thumb룰에 의해서 n의 quantile을 만드는데 있어서 최소 10n의 예가 필요함 만일 충분한 데이터가 없다면, normalization을 고수해야함

- Missing Data
  - dataset가 특정 feature에서 missing values가 있는 예시가 있다면(이러한 경우는 드뭄) 이 예시를 삭제하라 
  - 만약 이러한 예시가 자주 일어난다면, 이 feature를 모두 없애버리는 선택과 ML모델을 사용하여서 다른 예시로부터 missing values를 예측하는 선택이 있음
  - 예를들어 존재하는 feature data를 학습시킨 regression model를 사용함으로써 missing numerical data를 추론할 수 있음

## Create a Manual Similarity Measure
- 두 예시 사이에서의 similarity를 계산할때, 모든 feature data를 이 두 예시를 하나의 numeric value로써 합쳐야 함
- shoe size만 있는 하나의 shoe data set을 생가해보면 서로 다른 사이즈를 계산함으로써 두 개의 유사한 신발을 수량화할 수 있음
- numerical difference의 사이즈가 작을 수록 두 신발의 similarity는 증가함 
- 이러한 handcrafted similarity measure을 manual similarity measure라고 함

- 만약 shoes 사이의 similarities를 발견하기 위해서 size와 color을 사용하면 어떨까?
- color은 categorical data고 numerical size data와 합치기 어려움
- data가 더욱 복잡해질수록 manual similarity measure을 만드는 것은 더 어려움 
- data가 충분히 복잡하다면, manual measure을 할 필요가 없음 / 그 땐 supervised similarity measure로 바꾸어라 (supervised machine learning model이 similarity를 계산할 것임)

- manual similarity measure의 작동법을 이해하기 위해서 shoes의 예시를 보자
- 모델이 두 개의 features(shoe size and shoe price data)가 있다고 하자
- 두 개의 features는 numeric하고 이 두개를 single number로 묶어서 similarity로 나타낼 수 있음 / 아래를 따르라
<img src="https://user-images.githubusercontent.com/32586985/75104896-bc513600-5651-11ea-9b08-13ca4f7f3aab.png">

- 예시를 단순히 하기 위해서, US size인 8~11, prices 120~150에서의 similarity를 계산해보자
- distribution을 이해할 충분할 데이터가 없기 때문에, normalizing이 아닌 quantiles를 사용하여 데이터를 scale하여 단순화함
<img src="https://user-images.githubusercontent.com/32586985/75104917-09cda300-5652-11ea-8bda-670f07d94f74.png">

- 측정한 similarity는 feature data가 similar할수록 증가해야함 / 대신에 measured similarity는 실제로는 감소해야함
- measure similarity를 intuition에서 1을 빼서 확인해라 / Similarity = 1 - 0.17 = 0.83
- 일반적으로, numerical data를 Prepare data에서 준비할 수 있고 Euclidean distance를 이용하여 데이터를 합칠 수 있음
- Categorical data 같은 경우에는 아래와 같이 조치할 수 있음
  - Single valued(univalent), such as a car's color("white" or "blue" but never both)
  - Multi-valued(multivalent), such as a movie's genre(can be "action" and "comedy" simultaneously, or just "action")

- 만약 univalent data가 맞다면 similarity는 1일 것임 아닐 경우는 0일것
- multivalent data는 더 다루기 힘듬 / 예를들어 영화장르는 다루기 더 힘듬
- 이 문제를 다룰때는 어느정도 정립된 장르로부터 영화 장르를 할당해야함
- 일반적인 values에서의 비율을 사용함으로써 similarity를 계산하는 것을 Jaccard similarity로 불림
- 예시
<img src="https://user-images.githubusercontent.com/32586985/75104984-e6efbe80-5652-11ea-8aeb-08beafd5715c.png">

- 아래의 tabel은 categorical data를 어떻게 다루는지에 대한 몇 개의 예시를 보여줌
<img src="https://user-images.githubusercontent.com/32586985/75104990-0850aa80-5653-11ea-8d12-44d27afbb2cf.png">

- 일반적으로 similarity measure은 반드시 실제 similarity와 일치해야하고 metric이 그렇지 않다면, 필요한 정보를 encoding 한 것이 아님
- similarity measure을 만들기 전, 데이터를 주의깊게 process하라
- 비록 이 페이지에서의 예시는 작고 간단한 데이터세트에 의존하지만 실제 데이터 세트는 이것보다 더 크고 복잡함
- quantiles이 numeric data를 processing 하는데 좋은 default choice란 것을 기억하라

### 실습
<img src="https://user-images.githubusercontent.com/32586985/75105033-a3498480-5653-11ea-887b-adb1881ed3df.png">

- Preprocessing
  - 가장 먼저 numerical features를 preprocessing 함(price,size,number of bedrooms, and postal code)
  - 이 features들은 서로 다른 구동을 할 것임 / 이런 경우 pricing data는 bimodal distribution으로 여겨짐 
  - 그 이후는?
  <img src="https://user-images.githubusercontent.com/32586985/75105089-51552e80-5654-11ea-814a-56e877c601a3.png">
  
  - try explaining how you would process size data
    - Poisson: Create quantiles and scale to [0,1]
  
  - try explaining what how you would process data on the number of bedrooms 
    - clipping outliers and scaling to [0,1] / if find a power-law distribution then a log-transform might be necessary
  
- Calculating Similarity per Faeture
  - numeric features의 경우 간단히 차이를 발견할 수 있음
  - binary features의 경우 0과 1을 얻어서 차이를 발견할 수 있음
  - categorical features는?
  <img src="https://user-images.githubusercontent.com/32586985/75105351-45b63780-5655-11ea-8813-23776e03dff8.png">
  
- Calculating Overall Similarity
  - 모든 feature에 대해서 similarity를 numerically하게 계산할 수 있음 
  - 하지만 clustering algorithm은 cluster houses에 대해서 전반적인 similarity를 필요로함 
  - RMSE(root mean squared error)를 사용하여 per-feature similarity를 합치고 a pair of houses에 전체적인 similarity를 계산해보자
  <img src="https://user-images.githubusercontent.com/32586985/75105378-a5acde00-5655-11ea-8b2f-610cea7491fa.png">
  
- Limitations of Manual Similarity Measure
  - 실습에서 보였던 것과 같이 데이터가 복잡해진다면 데이터를 process하고 combine하는데 어렵고 정확히 similarity를 의미있는 방식으로 측정하는데 어려움을 겪음
  - similarity measure를 만들때 예시들 사이의 similarity를 완전히 반영하진 않고 derived clusters는 의미가 없을 것임
  - 이것은 categorical data의 많이 일어나며 supervised measure로 방식을 바꾸게끔 함

### 프로그래밍 실습
- 해당 데이터세트는 cocoa percentage, bean type, bean origin, maker name, maker country가 포함된 초콜릿바의 비율에 관한 것임

- 1.Load and clean data
<img src="https://user-images.githubusercontent.com/32586985/75105422-6c28a280-5656-11ea-9ffd-c27c7b4bc337.png">

- 2.Preprocess Data
  - review_data를 통해서 10년간의 데이터를 확인할 것임
  - 하지만 함수를 통해서 데이터가 좋은 데이터인지 확인해 볼 것임
  <img src="https://user-images.githubusercontent.com/32586985/75105451-c590d180-5656-11ea-9ef3-1aed910a930c.png">
  
  - distribution도 봄
  <img src="https://user-images.githubusercontent.com/32586985/75105464-e0634600-5656-11ea-8af6-bdb699869e68.png">
  
  - distribution이 Gaussian distribution의 형태를 띄므로 Normalize를 할 것임
  ```python
     # its a Gaussian! So, use z-score to normalize the data
     choc_data['rating_norm'] = (choc_data['rating'] - choc_data['rating'].mean()) / choc_data['rating'].std()
  ```
  - cocoa_percent의 distribution을 보고 어떻게 형성되었는지 보자
  <img src="https://user-images.githubusercontent.com/32586985/75105493-3801b180-5657-11ea-9730-e6c08baf97e4.png">
  
  - 역시 Gaussian distribution이므로 Normalize할 것임
  ```python
     choc_data['cocoa_percent_norm'] = (
         choc_data['cocoa_percent'] -
         choc_data['cocoa_percent'].mean()) / choc_data['cocoa_percent'].std()
  ```
  - normalization한 rating과 cocoa_percent를 확인해보자
  <img src="https://user-images.githubusercontent.com/32586985/75105516-9f1f6600-5657-11ea-9b69-ccf0c4ef9722.png">
  
  - broad_origin과 maker_location에서 similarity를 계산히기 위해서 longitude와 latitude가 필요함
  - DSPL을 이용하여 해당 필드를 추가할 것임
  <img src="https://user-images.githubusercontent.com/32586985/75105542-ee659680-5657-11ea-8826-a03894ecc52e.png">
  
  - latitudes와 longitudes의 distribution을 확인하여 어떻게 생성됐는지 확인함
  <img src="https://user-images.githubusercontent.com/32586985/75105551-0fc68280-5658-11ea-8e97-7e99ba8e1a77.png">
  
  - latitudes와 longitudes는 특별한 distribution이 아니므로 quantiles를 이용할 것임
  ```python
     numQuantiles = 20
     colsQuantiles = ['maker_lat', 'maker_long', 'origin_lat', 'origin_long']
     
     def createQuantiles(dfColumn, numQuantiles):
       return pd.qcut(dfColumn, numQuantiles, labels=False, duplications='drop')
     
     for string in colsQuantiles:
       choc_data[string] = createQuantiles(choc_data[string], numQuantiles)
     
     choc_data.tail()
  ```
  <img src="https://user-images.githubusercontent.com/32586985/75105575-7b105480-5658-11ea-844f-2ff30919c2b2.png">
  
  - Quantile values를 20까지로 했으므로 quantile values를 다른 feature data와 똑같은 scale로 함 [0,1]로 scaling하면서
  ```python
     def minMaxScaler(numArr):
       minx = np.min(numArr)
       maxx = np.max(numArr)
       numArr = (numArr - minx) / (maxx- minx)
       return numArr
     
     for string in colsQuantiles:
       choc_data[string] = minMaxScaler(choc_data[string])
  ```
  - maker과 bean_type은 categorical features이므로 one-hot encoding을 함
  ```python
     # duplicate the "maker" feature since it's removed by one-hot encoding function
     choc_data['maker2'] = choc_data['maker']
     choc_data = pd.get_dummies(choc_data, columns=['maker2'], prefix=['maker'])
     # similarly, duplicate the "bean_type" feature
     choc_data['bean_type2'] = choc_data['bean_type']
     choc_data = pd.get_dummies(choc_data, columns=['bean_type2'], prefix=['bean'])
  ```
  - clustering이후 결과를 해석하기 어려우므로 원래의 feature data를 새로운 dataframe으로 저장하라 
  ```python
     # Split dataframe into two frames: Original data and data for clustering 
     choc_data_backup = choc_data.loc[:, original_cols].copy(deep=True)
     choc_data.drop(columns=original_cols, inplace=True)
     
     # get_dummies returned ints for one-hot encoding but we want floats so divide by
     # 1.0
     # Note: In the latest version of "get_dummies", you can set "dtype" to float 
     choc_data = choc_data / 1.0
  ```
  - 완성된 데이터를 관찰해보라!
  ```python
     choc_data.tail()
  ```
  <img src="https://user-images.githubusercontent.com/32586985/75105575-7b105480-5658-11ea-844f-2ff30919c2b2.png">
  <img src="https://user-images.githubusercontent.com/32586985/75105699-f32b4a00-5659-11ea-9d3a-26296607245c.png">
  <img src="https://user-images.githubusercontent.com/32586985/75105701-f6bed100-5659-11ea-8442-b74afccb12b4.png">
  <img src="https://user-images.githubusercontent.com/32586985/75105703-faeaee80-5659-11ea-996e-eaa72bc00175.png">
  <img src="https://user-images.githubusercontent.com/32586985/75105708-fde5df00-5659-11ea-9514-8464dc4745c1.png">
  
