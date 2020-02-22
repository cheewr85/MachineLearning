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
  
