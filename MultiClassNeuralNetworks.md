## 다중 클래스 신경망 
- 여러 가능한 항목 중에서 선택할 수 있는 다중 클래스 분류를 살펴봄 
- 예:
  - 이 개는 비글인가요,바셋하운드인가요,아니면 블러드하운드인가요?
  - 이 꽃은 시베리안 아이리스인가요, 더치 아이리스인가요, 블루 플래그 아이리스인가요, 아니면 드워프 비어디드 아이리스인가요?
  - 이 비행기는 보잉 747인가요, 에어버스 320인가요, 보잉 777인가요, 아니면 엠브라에르 190인가요?
  - 이 이미지는 사과, 곰, 사탕, 개, 계란 중 무엇의 이미지인가요?
- 수백만개의 클래스 중에서 선택해야 하는 다중 클래스 문제도 있음 
- 일대다 다중 클래스
  - 각각의 가능한 클래스의 고유한 출력을 만듬
  - '내 클래스'대'다른 모든 클래스'의 신호에 대해 학습
  - 심층망에서나 별도의 모델을 사용하여 실행할 수 있음 
- 소프트맥스 다중 클래스
  - 제약조건 추가:모든 일대다 노드의 합이 1.0이 되어야함
  - 추가 제약조건은 빠르게 수렴을 학습하는데 도움이 됨 
  - 또한 출력을 확률로 해석할 수 있음 
- 사용 모델 및 시기
  - 다중 클래스, 단일 라벨 분류:
    - 하나의 예는 한 클래스만의 멤버가 될 수 있음
    - 클래스가 상호 배타적인 제약조건이 유용한 구조임 
    - 손실에서 인코딩하는 데 유용함 
    - 모든 가능한 클래스에 하나의 소프트맥스 손실을 사용함
  - 다중 클래스, 다중 라벨 분류:
    - 하나의 예가 두 개 이상의 클래스의 멤버가 될 수 있음 
    - 클래스 멤버에 추가로 적용되는 제약조건이 없음 
    - 각각의 가능한 클래스에 하나의 로지스틱 회귀 손실이 있음 
- 소프트맥스 옵션
  - 전체 소프트맥스
    - 무차별 대입. 모든 클래스에 관해 계산함 
  - 후보 샘플링
    - 모든 양성 라벨에 대해 계산하지만 음성 라벨의 무작위 샘플에 대해서만 계산함 
    
## 다중 클래스 신경망:일대다
- 이진 분류 활용 방법을 제공함
- 가능한 솔루션이 N개인 분류 문제의 경우 일대다 솔루션은 가능한 각 결과에 하나씩 N개의 이진 분류자로 구성됨 
- 학습하는 동안 모델은 일련의 이진 분류자를 통해 실행되며 별도의 분류 문제에 답하기 위해 각 분류자를 학습함 
- 이 방식은 총 클래스 수가 작으면 매우 합리적이지만 클래스 수가 증가하면 점점 더 비효울적이 됨
- 각 출력 노드가 다른 클래스를 나타내는 심층신경망을 사용하여 훨씬 더 효율적인 일대다 모델을 만들 수 있음 
<img src="https://user-images.githubusercontent.com/32586985/71302678-7ed1f180-23f1-11ea-9409-aa39600da6ba.PNG">


## 다중 클래스 신경망:소프트맥스
- 로지스틱 회귀는 0과 1.0 사이의 소수를 생성함/결과론적으로 확률의 합이 1.0이 됨
- 소프트맥스는 이 아이디어를 다중 클래스 문제에 적용함/다중 클래스 문제의 각 클래스에 소수 확률을 할당함/소수 확률의 합은 1.0이 되어야함
- 이 제약조건을 추가하면 제약조건을 추가하지 않은 경우보다 더 빠르게 수렴을 학습할 수 있음 
- 위에 있는 예시에서의 이미지 분석의 경우 이미지가 특정 클래스에 속할 확률은 다음과 같음 
<img src="https://user-images.githubusercontent.com/32586985/71302723-0fa8cd00-23f2-11ea-9fec-d2f39236bfb0.PNG">

- 소프트맥스는 출력 레이어 바로 앞의 신경망 레이어를 통해 구현됨/소프트맥스 레이어의 노드 수는 출력 레이어와 같아야함 
<img src="https://user-images.githubusercontent.com/32586985/71302729-35ce6d00-23f2-11ea-9521-5199117273a9.PNG">

- 소프트맥스 방정식
- 이 수식은 기본적으로 로지스틱 회귀 수식을 다중 클래스로 확장함 
<img src="https://user-images.githubusercontent.com/32586985/71302745-572f5900-23f2-11ea-9ef0-f09334382117.PNG">

## 소프트맥스 옵션
- 변형된 소프트맥스가 존재
  - 전체 소프트맥스는 지금까지 설명한 소프트맥스임/모든 가능한 클래스의 확률을 계산하는 소프트맥스임 
  - 후보 샘플링은 소프트맥스가 양성 라벨의 확률은 모두 계산하지만 음성 라벨의 경우 무작위 샘플의 확률만 계산함
    - 예를 들면 입력 이미지가 비글인지, 블러드하운드인지 확인하는 경우 개가 아닌 예의 확률은 제공할 필요가 없음 
- 전체 소프트맥스는 클래스 수가 적으면 매우 적은 비용이 들지만 클래스 수가 증가하면 엄청나게 많은 비용이 듬 
- 클래스 수가 많은 문제에서 후보 샘플링을 사용하면 효율성이 향상됨 

## 라벨 1개 대 라벨 여러 개
- 소프트맥스에서는 각 예가 정확히 한 클래스의 멤버라고 가정함 
- 동시에 여러 클래스의 멤버인 예도 있음/이러한 예의 경우
  - 소프트맥스를 사용할 수 없음
  - 로지스틱 회귀를 여러 개 사용해야 함
- 정확히 하나의 항목(과일 한 개)이 포함된 이미지를 예로 들어봄
  - 소프트맥스는 해당 항목이 배,오렌지,사과 또는 다른 과일일 확률을 확인할 수 있음 
  - 모든 종류의 항목(여러 종류의 과일 접시)이 포함된 이미지 예의 경우 대신 로지스틱 회귀를 여러 개 사용해야함 
  

## 프로그래밍 실습
- 신경망으로 필기 입력된 숫자 분류하기
- 각각의 입력 이미지를 올바른 숫자에 매핑하는 것/몇 개의 히든 레이어를 포함하며 소프트맥스 레이어가 맨 위에서 최우수 클래스를 선택하는 NN만듬
- 설정/이 데이터는 원본 MNIST 학습 데이터에서 20000개 행을 무작위로 추출한 샘플임
```python
   from __future__ import print_function
   
   import glob
   import math
   import os
   
   from IPython import display
   from matplotlib import cm
   from matplotlib import gridspec
   from matplotlib import pyplot as plt
   import numpy as np
   import pandas as pd
   import seaborn as sns
   from sklearn import metrics
   %tensorflow_version 1.x
   import tensorflow as tf
   from tensorflow.python.data import Dataset
   
   tf.logging.set_verbosity(tf.logging.ERROR)
   pd.options.display.max_rows = 10
   pd.options.display.float_format = '{:.1f}'.format
   
   mnist_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv", sep=",", header=None)
   
   # Use just the first 10,000 records for training/validation.
   mnist_dataframe = mnist_dataframe.head(10000)
   
   mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
   mnist_dataframe.head()
```
<img src="https://user-images.githubusercontent.com/32586985/71303093-77154b80-23f7-11ea-9630-69221ea37bc8.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71303095-81cfe080-23f7-11ea-898c-87cd25dc216c.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71302952-3ddbdc00-23f5-11ea-98e8-77a2d385c9ae.PNG">

- 첫 번째 열은 클래스 라벨을 포함함/나머지 열은 특성 값을 포함하며 28x28=784개 픽셀 값마다 각각 하나의 특성 값이됨
- 이 784개의 픽셀 값은 대부분 0이지만, 1분 정도 시간을 들여 모두 0은 아니라는 것을 확인됨
- 이러한 예는 비교적 해상도가 낮고 대비가 높은 필기 입력 숫자임
- 0~9 범위의 숫자 10개가 각각 표현되었으며 가능한 각 숫자에 고유한 클라스 라벨이 지정됨/이 문제는 10개 클래스를 대상으로 하는 다중 클래스 분류 문제임
- 라벨과 특성을 해석하고 몇 가지 예를 살펴봄/이 데이터 세트에는 헤더 행이 없지만 loc를 사용하여 원래 위치를 기준으로 열을 추출할 수 있음
```python
   def parse_label_and_features(dataset):
     """Extracts labels and features.
     
     This is a good place to scale or transform the features if needed.
     
     Args:
       dataset: A Pandas 'Dataframe', containing the label on the first column and 
         monochrome pixel values on the remaining columns, in row major order.
     Returns:
       A 'tuple' '(labels, features)':
         labels: A Pandas 'Series'.
         features: A Pandas 'DataFrame'.
     """
     labels = dataset[0]
     
     # DataFrame.loc index ranges are inclusive at both ends.
     features = dataset.loc[:,1:784]
     # Scale the data to [0, 1] by dividing out the max value, 255.
     features = features / 255
     
     return labels, features
```
```python
   training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])
   training_examples.describe()
```
<img src="https://user-images.githubusercontent.com/32586985/71303157-07ec2700-23f8-11ea-8435-46d709844803.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71303159-14707f80-23f8-11ea-8d47-75c3b2ece8d6.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71303162-1f2b1480-23f8-11ea-9124-542c6f9cd2f5.PNG">
```python
   validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])
   validation_examples.describe()
```
<img src="https://user-images.githubusercontent.com/32586985/71303178-5c8fa200-23f8-11ea-8a40-ef1451292c61.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71303181-66b1a080-23f8-11ea-9a95-eab3c43b3b1f.PNG">
<img src="https://user-images.githubusercontent.com/32586985/71303186-6fa27200-23f8-11ea-9a70-8e7b84f8c506.PNG">
```python
   rand_example = np.random.choice(training_examples.index)
   _, ax = plt.subplots()
   ax.matshow(training_examples.loc[rand_example].values.reshape(28, 28))
   ax.set_title("Label: %i" % training_targets.loc[rand_example])
   ax.grid(False)
```
<img src="https://user-images.githubusercontent.com/32586985/71303192-821cab80-23f8-11ea-8bb3-b8e1bdca00aa.PNG">

### 작업1:MNIST용 선형 모델 구축
- 비교 기준이 될 모델을 만듬/LinearClassifier는 K개 클래스마다 하나씩 k개의 일대다 분류자 집합을 제공함 
- 이 작업에서는 정확성을 보고하고 시간별 로그 손실을 도식화할 뿐 아니라 혼동행렬도 표시함
- 혼동행렬은 다른 클래스로 잘못 분류된 클래스를 보여줌
- log_loss 함수를 사용하여 모델의 오차를 추적함/이 함수는 학습에 사용되는 LinearClassifier 내장 손실 함수와 다르므로 주의해야함 

