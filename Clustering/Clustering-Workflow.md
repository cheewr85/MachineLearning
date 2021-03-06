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
  
- 3.Calculate Manual Similarity
  - similarity function에 해당하는 코드를 먼저 실행함
  ```python
     def getSimilarity(obj1, obj2):
       len1 = len(obj1.index)
       len2 = len(obj2.index)
       if not (len1 == len2):
         print("Error: Compared objects must have same number of features.")
         sys.exit()
         return 0
       else:
         similarity = obj1 - obj2
         similarity = np.sum((similarity**2.0) / 10.0)
         similarity = 1 - math.sqrt(similarity)
         return similarity
  ```
  - first chocolate와 그 다음에 4 chocolates사이의 similarity를 계산하라 
  - 다음셀에서는 나오는 실제 feature data에 대한 similarity와 비교해보아라
  - choc1: 0 / chocsToCompare: [1,4]
  <img src="https://user-images.githubusercontent.com/32586985/75106567-80be6800-5661-11ea-9409-9830ab78495e.png">

- 4.Cluster chocolate Dataset
  - k-means clustering functions을 사용하여 설정을 하여라
  - k가 30 클러스터의 수가 30인 데이터세트로 설정하여 실행하라
  <img src="https://user-images.githubusercontent.com/32586985/75106773-3938db80-5663-11ea-8e6c-a56b122e43fa.png">
  <img src="https://user-images.githubusercontent.com/32586985/75106796-8026d100-5663-11ea-93ce-d6ddd4a0f94e.png">
  <img src="https://user-images.githubusercontent.com/32586985/75106802-89b03900-5663-11ea-8560-b3fd850f37f9.png">
  <img src="https://user-images.githubusercontent.com/32586985/75106808-916fdd80-5663-11ea-9ef4-702a18215677.png">
  <img src="https://user-images.githubusercontent.com/32586985/75106810-9af94580-5663-11ea-95d0-77dfdcbea753.png">
  <img src="https://user-images.githubusercontent.com/32586985/75106813-a3518080-5663-11ea-8ad7-71ac42379384.png">
  
  - Inspect Clustering Result
  - clusterNumber: 7
  <img src="https://user-images.githubusercontent.com/32586985/75107066-d6940f80-5663-11ea-8717-a87290e19804.png">
  
  - clustering result가 의도치않게 특정 features에 가중치를 많이 부여함
  - 몇 몇의 겹치는 항목으로 인해서 생긴것으로 보임 / 해결책은 supervised similarity measure을 사용하는 것임 / DNN이 해당 정보를 제거해 줄것임
- 5.Quality Metrics for Clusters
  - 오류로 인해 추후 결과해석에서 다룰예정

## Supervised Similarity Measure
- 수동으로 결합된 feature data를 비교하는 대신, 임베딩을 통해서 feature data를 줄일 수 있으며, 임베딩을 통해서 비교할 수있음
- 임베딩은 supervised deep neural network를 통해서 feature data를 스스로 학습함으로써 생산해냄
- 임베딩맵은 임베딩 공간에 feature data를 벡터화한 것임
- 임베딩 공간은 feature data set에 있는 숨어있는 몇 개의 구조를 feature data안에 잡아내는 방식보다는 더 적은 차원을 가지고 있음
- 임베딩 벡터는 예를 들면 유튜브 영상을 같은 유저가 시청했을 경우, 결국 같은 임베딩 공간으로 close 하게 옴
- similarity measure이 이 closeness를 예제들의 pairs를 위해 similarity를 수량화하는데 어떻게 사용하는지 알 수 있음

- Comparison of Manual and Supervised Measures
<img src="https://user-images.githubusercontent.com/32586985/75212878-964fa100-57cb-11ea-88f9-4539d9a03abc.png">

- Process for Supervised Similarity Measure
<img src="https://user-images.githubusercontent.com/32586985/75212945-e29ae100-57cb-11ea-97ff-2fc6481a2690.png">

- Choose DNN Based on Training Labels
  - DNN을 input과 label 둘 다 쓰여진 feature data를 학습함으로써 임베딩하여 feature data를 줄여라
  - 예를들어 house data에서 DNN은 가격, 크기, postal code등의 feature를 features 자체를 예측하기 위해서 사용할 것임
  - feature data를 같은 feature data를 예측하기 위해 사용하기 위해서 DNN은 input feature data를 임베딩함으로써 줄이게 될 것임
  - 이러한 임베딩을 similarity를 계산하는데 사용할 것임
  
  - input data 자체를 예측함으로써 input data를 임베딩하여 학습하는 DNN을 antoencoder라고 부름
  - autoencoder는 hidden layers가 input과 output layer보다 작고, input feature data를 나타내는 것을 압축시키게끔 학습하도록 되어 있음 
  - DNN이 학습될 때, last hidden layer로부터 similarity를 계산하기 위해 임베딩을 추출할 것임
  <img src="https://user-images.githubusercontent.com/32586985/75213403-538ec880-57cd-11ea-836f-5076c437038d.png">
  
  - autoencoder은 임베딩을 생산하는데 가장 단순한 선택임
  - autoencoder은 특정 feature가 다른 similarity를 결정하는 것보다 중요하다면 선택사항이 아님
  - 예를들어 house data에서는 postal code보다 price가 더 중요하게 여겨짐 
  - 이러한 경우 중요한 feature에 대해서만 학습 label로 DNN에 사용하라
  - 이 DNN은 모든 input features를 예측하는 대신 특정한 input feature를 예측할 것임
  - 이것을 predictor DNN이라고 불리움 / 아래의 feature를 선택하는 가이드라인을 참고하라
    - categorical features보다 numeric features를 라벨로 선택하라 / loss를 계산하고 해석하는데 numeric features가 더 나음
    - categorical features를 cardinality가 100이하인 label의 경우 사용하지 마라
    - 만일 하게 된다면 DNN이 input data를 임베딩 하기 위해서 감소하지 않을 것임 왜냐하면 DNN은 low-cardinality categorical label에 대해서는 쉽게 예측하기 때문임
    - DNN의 input으로부터 label로 사용된 feature를 제거하라 DNN이 완벽히 output을 예측할 것임
  - label의 선택에 따라 달려 있지만 DNN의 결과는 autoencoder DNN 혹은 predictor DNN이 될 것임

- Loss Function for DNN
  - DNN을 학습할 때, loss function을 아래의 과정을 거쳐서 만들어야함
    - DNN의 모든 output에서 loss를 계산하라 / 해당 output은
      - Numeric, use mean square error(MSE)
      - Univalent categorical, use log loss / log loss를 직접 실행할 필요는 없음, library function을 사용하기 때문에
      - Multivalent categorical, use softmax cross entropy loss / softmax cross entropy loss를 직접 실행할 필요는 없음, library function을 사용하기 때문에
    - 모든 output의 loss를 합쳐서 total loss를 계산하라
  - loss를 합칠 때, 각각의 feature는 loss의 어느정도 부분을 차지한다는것을 확실시하라
  - 예를들어, color data를 RGB 값으로 변환할때, 3개의 output이 있을 것임
  - 하지만 loss를 합칠 때 3개의 output은 color를 위한 loss는 다른 features에 비해서 3배 이상 가중치가 부여되는 것을 의미함
  - 대신에 각각의 output에 1/3을 곱하여라

- Using DNN in an Online System 
  - 온라인 머신러닝은 새로운 input data에 대해서 continuous하게 stream이 이어짐
  - DNN의 새로운 데이터에 대해서 학습을 할 필요가 있음 
  - 하지만 DNN을 scratch로부터 다시 학습시킨다면 임베딩이 다를 것임 왜냐하면 DNN은 초기에 랜덤 가중치를 부여하기 때문임
  - 대신에 항상 존재하는 가중치에 warm-start를 한 후 DNN의 새로운 데이터로 업데이트를 하여라
  
## Generating Embeddings Examples
- 이 예시는 임베딩이 어떻게 supervised similarity measure을 사용해서 생산하는지 보여줌
- 아래의 예시가 있다고 가정해보자 (manual similarity measure)
<img src="https://user-images.githubusercontent.com/32586985/75238529-830cf780-5804-11ea-9bad-946c5f652ab4.PNG">

- Preprocessing Data
  - feature data를 input으로 사용하기 이전에, 데이터를 preprocess를 해야함
  - preprocessing steps은 manual similarity measure을 생성할 때 기본이 되는 과정임
  <img src="https://user-images.githubusercontent.com/32586985/75238707-cebfa100-5804-11ea-9021-97acb9aafc76.PNG">
  
- Choose Predictor or Autoencoder
  - 임베딩을 생성하기 위해서, autoencoder 혹은 predictor을 선택할 수 있음
  - default choice는 autoencoder임 / predictor은 데이터세트의 특정한 features가 similarity를 결정할 때만 선택함
  - Train a Predictor
    - 이 features 같은 경우 예시들 사이에서 similarity를 선택하는데 중요한 DNN을 위한 training labels로써 선택할 수 있음
    - price가 houses사이에 similarity를 결정하는데 가장 중요한 것이라고 가정해보자
    - price를 training label로 선택하고, DNN에서 input feature data로부터 제거함
    - input data로써 모든 다른 features를 사용함으로써 DNN을 학습시켜라
    - 학습을 위해서 loss function은 예측과 실제 값 사이의 MSE로 될 것임
  - Train an Autoencoder
    - autoencoder를 우리의 dataset으로써 다음의 과정을 따름
      - 1.autoencoder의 hidden layers은 input과 output layer보다 작아야함
      - 2.Supervised Similarity Measure로 표현된 각각의 output에 대해서 loss를 계산하라 
      - 3.각각의 output에 대해서 loss를 합한 loss function을 만들어라 / 모든 feature에 대해 동일하게 loss의 가중치를 부여하라 / color data는 RGB로써 생산되므로, RGB 각각의 1/3의 output의 가중치를 부여하라 
      - 4.DNN을 학습하라

- Extracting Embeddings from the DNN  
  - DNN을 학습한 이후, predictor던 autoencoder던 DNN으로부터 예시를 임베딩하기 위한 것을 추출하라
  - input으로써 feature data의 예시를 사용함으로써 임베딩을 추출하고, 최종적인 hidden layer의 output을 읽어라
  - 이 output은 embedding vector로부터 기인한 것임
  - 비슷한 houses에 대한 vector는 비슷하지 않은 houses보다 vector가 close together 해야함

## Measuring Similarity from Embeddings
- 어떠한 예제로도 임베딩을 할 것임
- similarity measure은 이러한 임베딩과 returns을 그들의 similarity를 측정하는 수로 가져옴
- 임베딩은 단순히 vectors of numbers라는 것을 기억하라
- 만일 두 개의 벡터에서의 similarity를 찾을려면 아래의 방식을 참조하라
<img src="https://user-images.githubusercontent.com/32586985/75239959-e8fa7e80-5806-11ea-8900-04ab63d8d5cb.PNG">

- Choosing a Similarity Measure
  - cosine과 반대로, 내적은 vector길이의 일부분을 차지함
  - 이것은 매우 중요함 / 왜냐하면 예제에서 training set에서 임베딩 vector를 큰 길이로 갖는 경우가 있게 빈번하게 등장하기 때문에
  - popularity를 계산하고 싶다면 내적을 선택해야함 / 하지만 risk는 popular examples이 similarity metric을 skew할 수도 있기 때문임
  - 이 skew를 밸런스화 하기 위해서 길이를 거듭제곱 알파가 1보다 작고 내적을 (a알파제곱)(b알파제곱)cos(세타)로 내적을 계산할 수 있음
  - vector length를 similarity measure로 어떻게 변화시켰는지 잘 이해시키기 위해서 vector 길이를 1로 일반화하고 3개의 measures를 각각의 비율로 생각함
  <img src="https://user-images.githubusercontent.com/32586985/75241112-db45f880-5808-11ea-8f0f-6174fa6b7a18.PNG">
  
### 프로그래밍 실습
- Supervised Similarity Measure
- k-means with a supervised similarity measure
- 1.Load and clean data
  <img src="https://user-images.githubusercontent.com/32586985/75250424-aee7a780-581b-11ea-8840-6b2b5da121ce.png">

- 2.Process Data
  - DNN을 사용하면 data를 수동으로 process할 필요 없음 / DNN이 데이터를 transform함
  - 하지만 가능하다면 similarity 계산을 distort하는 features를 제거해야함
  - 아래의 예시에서는 similarity와 관련없은 features에 대해서 제거를 함
  ```python
     choc_data.drop(columns=['review_date','reference_number'],inplace=True)
     choc_data.head()
  ```
  <img src="https://user-images.githubusercontent.com/32586985/75250642-1ef62d80-581c-11ea-964e-08b6bf64a97c.png">
  
- 3.Generate Embeddings from DNN
  - DNN을 feature data의 학습함으로써 임베딩을 생산할 준비가 됨 
  - set up functions을 통해서 임베딩을 생산하는 DNN을 학습함
  - DNN train
    - predictor DNN과 autoencoder DNN을 선택할 수 있음 / predictor DNN의 경우 특정한 한 feature만, autoencoder DNN의 경우 모든 feature에 대해서
    - 다른 parameters를 바꿀 필요는 없음 / l2_regularization:L2_regularization의 가중치 조절, hidden_dims:hidden layer의 차원 조절
    <img src="https://user-images.githubusercontent.com/32586985/75251086-3d105d80-581d-11ea-9470-762743d83b2d.png">
    <img src="https://user-images.githubusercontent.com/32586985/75251095-413c7b00-581d-11ea-9965-ed7ca5285d72.png">
    
- 4.Cluster Chocolate Dataset
  - chocolate을 cluster하기 위해서 k-means clustering functions을 설정하라
  - k = 160이라는 cluster의 수를 사용할 예정임
  - k-means의 모든 반복에서 output은 모든 예제로부터의 centroids는 감소하는데 k-means가 항상 converges하듯이 길이의 합이 어떻게 되는지 보여줌
  <img src="https://user-images.githubusercontent.com/32586985/75251350-c4f66780-581d-11ea-819c-c2955b8e76a2.png">
  
- Inspect Clustering Result
  - parameter를 바꿈으로써 서로 다른 clusters에서의 초콜릿을 관찰하라 
  - cluster를 볼 때 다음의 질문을 생각하라
    - clusters가 의미있는가?
    - clustering의 결과가 manual similarity measure나 supervised similarity measure보다 더 나은 결과를 도출하는가?
    - clusters의 수를 바꾸는 것이 clusters를 더 의미있게 만드는가 아니면 덜 의미있게 만드는가?
  <img src="https://user-images.githubusercontent.com/32586985/75251705-82815a80-581e-11ea-9eb1-96252a29349a.png">
  
  - cluster가 의미있는가?
    - cluster의 수가 거의 100을 근접하여 넘을 정도로 증가할 때 cluster가 의미있어짐 / 100 이하의 clusters에서는 dissimilar chocolates가 그룹화 하는 경향을 보임 / numeric features를 grouping하는 것이 categorical features보다 더 의미가 있음 / DNN을 정확히 categorical feature로 encoding하는 것은 가능하지 않음 왜냐하면 예시가 1800를 넘을 경우 categorical features가 가지고 있는 몇 십개의 values에 대해서 충분히 encode하지 못함
  - clustering의 결과가 manual similarity measure이나 supervised similarity measure보다 더 나은 결과를 도출하는가?
    - clusters가 manual similarity measure보다는 의미가 있음 왜냐하면 measure을 chocolates사이에서 similarity를 정확히 도출하게 수정할 수 있기 떄문임 / Manual design도 dataset이 복잡하지 않다면 가능함 / supervised similarity measure의 경우 데이터를 DNN에 맡기고 similarty를 encode하는 것도 DNN의 의존함 / 이러한 경우 데이터세트가 작을 경우 DNN이 similarity encode를 정확히 하는데 어려움을 겪음 
  - clusters의 수를 바꾸는 것이 clusters를 더 의미있게 만드는가 아니면 덜 의미있게 만드는가?
    - clusters의 수를 증가시키는 것이 clusters를 한계까지 더 의미있게 만듬 / 정확한 clusters로 인해 dissimilar chocolates이 사라질 수 있음

- 5.Quality Metrics for Clusters
  - set up functions을 설정하여 metrics를 계산함
  - metrics를 계산하는 것에 다음이 포함됨
    - cardinality of your clusters
    - magnitude of your clusters 
    - cardinality vs magnitude
  - Observe
    - plot은 많은 clusters를 위해 cluster metrics를 관찰하는 것이 쉽지 않다는 것을 보여줌 / 하지만 plot은 clustering의 quality의 일반적인 아이디어를 제공해줌 / outlying clusters의 수도 있음
    - cluster cardinality와 cluster magnitude사이의 상관관계는 manual similarity measure보다 낮음 / lower correlation은 몇 개의 초콜릿은 cluster하기 어려운 것을 보여줌 (large example-centroid distances에 따르면)
  - clusterQualityMetrics(choc_embed)
  <img src="https://user-images.githubusercontent.com/32586985/75252789-b198cb80-5820-11ea-9537-223b28b3f8f1.png">
  
- Find Optimum Number of Clusters
  <img src="https://user-images.githubusercontent.com/32586985/75252959-01779280-5821-11ea-89ad-67ea6f5fe689.png">

- Summary (supervised similarity measure의 특징으로도 볼 수 있음)
  - Eliminates redundant information in correlated features
    - DNN은 redundant한 정보를 제거함 / 이 특성을 증명하기 위해서 DNN을 정확한 데이터로 학습하고 manual similarity measure의 결과와 비교해 봐야함
  - Does not provides insight into calculated similarities
    - embeddings이 나타내는 것이 알 수 없기 때문에 clustering 결과에 대한 insight이 없음
  - Suitable for large datasets with complex features
    - 데이터세트가 DNN을 정확히 학습하기 적다면, DNNs이 학습을 위해서 큰 데이터세트가 필요하단 것을 증명해야함 / 이점은 input data를 이해할 필요가 없다는 것임 / large dataset은 이해하기 어렵기 때문에 이러한 두 특성이 연관성이 깊음
  - Not suitable for small datasets
    - 작은 데이터세트는 DNN을 학습하는데 충분한 정보가 없음
    
## Similarity Measure Summary
<img src="https://user-images.githubusercontent.com/32586985/75241905-62e03700-580a-11ea-80a7-aa316158aa2f.PNG">
