## 임베딩(강의)
- 고차원 벡터의 변환을 통해 생성할 수 있는 상대적인 저차원 공간을 가리킴
- 단어를 나타내는 희소 벡터와 같이 커다란 입력값에 대해 머신러닝을 더 쉽게 수행할 수 있음 
- 임베딩이 잘 동작하는 경우 의미가 유사한 입력값들을 임베딩 공간 안에 서로 근접하게 위치시켜 입력값의 특정 의미를 포착함 
- 모델의 관계없이 학습과 재사용이 가능함 

- 협업 필터링에서 필요한 경우
  - 입력:사용자 500,000명이 선택한 영화 1,000,000편
  - 작업:사용자에게 영화 추천
  - 이 문제를 해결하려면 어떤 영화가 서로 비슷한지 파악하는 방법이 필요함
  - 유사성별로 영화 정리(1차원)
    - 사람들이 어떤 영화를 상대적으로 더 좋아하게 만드는 복잡한 요소를 모두 파악하기가 어려움 
  <img src="https://user-images.githubusercontent.com/32586985/71352983-120a5300-25bb-11ea-9a31-2fe5b9d57ae9.PNG">
  
  - 유사성별로 영화 정리(2차원)
  <img src="https://user-images.githubusercontent.com/32586985/71353083-5b5aa280-25bb-11ea-9c33-32d00ce9883c.PNG">
  
- 하지만 파악하고 싶은 다른 요소가 아주 많으므로 2차원 이상을 원하게 됨/실제로는 많은 차원으로 함   
  - 지금은 눈에 보이게 하기 위해서 2차원으로 한정하여 예시를 보임
  - 2차원 임베딩/X축에 왼쪽에는 어린이용 영화가 배치, 오른쪽으로 갈수록 성인물에 가까움/Y축에 위쪽에는 블록버스터 영화가 아래쪽에는 예술영화
  - 여러 구조가 체계적으로 표시되며 근처에 있는 영화는 서로 비슷한류로 보임
  <img src="https://user-images.githubusercontent.com/32586985/71353236-c73d0b00-25bb-11ea-9a58-1a984883b19e.PNG">
  
- D차원 임베딩
  - 영화에 대한 사용자의 관심분야를 대략 d개의 측면에서 설명할 수 있다고 가정해보자
  - 각각의 영화는 차원 d의 값이 해당 측면에 대한 각 영화의 일치도를 나타내는 d차원 지점이 됨 
  - 임베딩은 데이터를 통해 학습할 수 있음 
  - 위의 예시 중 몇 개를 실제 데이터 값으로 가정해봄
  
- 심층망에서 임베딩 학습
  - 임베딩 레이어는 차원당 단위 하나를 갖는 히든 레이어에 불과하므로 별도의 학습 과정이 필요하지 않음 
  - 지도 정보(예:사용자가 동일한 영화 2개를 시청함)를 통해 원하는 작업에 맞게 학습된 임베딩을 조정함 
  - 히든 단위는 최종 목표를 최적화하도록 d차원 공간에서 항목을 정리하는 방법을 직관적으로 발견함 

- 입력 표현 
  - 각 예(이 행렬의 행)는 사용자가 시청한 특성(영화)의 희소 벡터임
  - 이 예의 밀집표현은 다음과 같음/(0, 1, 0, 1, 0, 0, 0, 1)
  - 아래의 예시에 표시된 행렬은 일반적으로 생각하는 협업 필터링 입력을 나타냄 
  - 사용자마다 한 개의 행이 있으며 영화마다 한 개의 열이 있음/체크표시는 사용자가 그 영화를 봤다는 의미임
  - 각 예는 이 행렬의 한 행이 될 것이므로 노란색으로 강조표시한 맨 아래 행에 초점을 맞춤
  - 아래의 예시는 시간과 공간 측면에서 효율적이진 않음 
  <img src="https://user-images.githubusercontent.com/32586985/71353639-00c24600-25bd-11ea-81e0-484971c44e95.PNG">
  
- 따라서 아래와 같은 예시를 사용함 
<img src="https://user-images.githubusercontent.com/32586985/71353778-6f070880-25bd-11ea-8fd5-48b4e48dbfbc.PNG">

  - 각 특성을 0에서(영화의 수 -1)까지의 정수로 매핑하는 사전을 만듬/사전은 각 특징을 매핑함 
  - 희소 벡터를 사용자가 시청한 영화로만 효율적으로 표현함
  - 아래와 같이 표현 가능 
  <img src="https://user-images.githubusercontent.com/32586985/71353875-a8d80f00-25bd-11ea-8f0c-e595ef01502f.PNG">
  
- 심층망의 임베딩 레이어
  - 주택 판매가를 평가하는 회귀 문제
    - 일반적으로 회귀 문제로 처리함/제곱손실을 최적화 하고자 함/임베딩 레이어를 작성할 대상은 판매,광고에서 사용하는 단어
    - 주택의 크기를 알아내기 위해 비슷하게 사용되는 단어를 이해해야함/희소 임베딩을 원하므로 몇 개만 삽입할것임,단어 사용
    - 3차원 임베딩에 대해서 알아봄, 실제로는 훨씬 더 많이 필요/임베딩 레이어는 녹색, 실제로는 히든 레이어이며,3차원을 원하므로 3개의 유닛이 있음
    - 위도, 경도, 방 수와 같은 기타 입력 데이터가 있을 수 있음/분홍색은 다른 히든 레이어를 원하는만큼 사용할 수 있다는 표시,표준 히든레이어임
    - 유닛 수를 결정할 수 있고, 이 유닛들은 마지막에는 단일 유닛으로 모이게 됨
    - 회귀 문제가 실제 값을 제공하며 판매 가격을 사용하여 L2 손실을 최적화함 
    - 역전파를 사용하는 동안 임베딩 레이어를 사용 
  <img src="https://user-images.githubusercontent.com/32586985/71354134-564b2280-25be-11ea-95b8-9360cb18ac3e.PNG">

  - 필기 입력된 숫자를 평가하는 다중 클래스 분류
    - 0부터 9까지의 숫자가 있음, 실제로 올바른 숫자 라벨이 있는 학습데이터가 있음 
    - 임베딩을 만들려는 대상은 그림의 원시 비트맵임/이 비트맵은 흰색 또는 검은색, 0 또는 1임
    - 3차원 임베딩을 그대로 사용/분홍색은 추가 히든 레이어임, 로지트 레이어
    - 10개의 숫자가 있으며, 기본적으로 각 숫자가 해당 숫자일 확률을 계산하는 확률 분포에 대해 배움 
    - 정답을 알고 있는 원 핫 대상 확률 분포를 하나 사용하여 소프트 맥스 손실을 최적화함
    - 역전파에 대해 학습하며 이미지를 삽입하는 방법을 배움 
    <img src="https://user-images.githubusercontent.com/32586985/71354431-4aac2b80-25bf-11ea-81dd-ca857c0b5653.PNG">
    
  - 추천 영화를 평가하는 협업 필터링
    - 사용자가 10편의 영화를 본다고 가정할 시
    - 무작위로 3개의 영화를 선택하여 분리후 다음 추천할 영화로 레이블을 지정함,본 영화이므로 좋은 추천영화가 됨/나머지 7개의 영화를 학습 데이터로 사용 
    - 문자 인식과 매우 유사/희소성 표현을 얻는 방법을 알고 있으므로 내 학습 데이터인 7개의 영화를 임베딩 레이어로 가져옴 
    - 장르,감독,사용자 등 영화에 대한 모든 특징을 사용할 수 있고, 추가 히든 레이어로 가져올 수 있음/로지트 레이어 포함
    - 이 로지트 레이어는 매우 큼/50만편의 영화가 있다면 50만개의 로지트 레이어가 생김
    - 하지만 50만편의 영화 중에 좋아할 것으로 생각되는 영화의 분포를 얻게 되며, 좋아한다고 생각하여 분류한 영화에서 소프트 및 맥스 손실을 최적화함
    - 역전파와 표준 학습에서 이 작업을 할 때 앞에서 설명한 영화 임베딩을 배울 것임 
    <img src="https://user-images.githubusercontent.com/32586985/71354618-dd4cca80-25bf-11ea-8a82-94f8b18af506.PNG">

- 기하학적 보기와 반응 
  - 심층망 
    - 각 히든 단위는 하나의 차원(잠재 특성)에 대응함
    - 영화와 히든 레이어간의 에지 가중치가 좌표값임 
    <img src="https://user-images.githubusercontent.com/32586985/71354928-c8246b80-25c0-11ea-91e5-04aa52f305fa.PNG">
  
  - 단일 영화 임베딩의 기하학적 보기
  <img src="https://user-images.githubusercontent.com/32586985/71354995-f99d3700-25c0-11ea-9401-c8d77572cd97.PNG">
  
  - 왼쪽에 있는 심층신경망에서 영화선택/50만편 중 하나를 나타냄/검은색으로 만듬
  - 3개의 히든 유닛 있다고 하였고 3차원 임베딩 사용/검은색 노드는 각 유닛에 연결하는 엣지를 가짐
  - 첫번째 유닛에 빨간색,두번째는 자홍색,세번째는 갈색/신경망 학습후 이들 엣지는 가중치가 되며 각 엣지는 실제 값이 연결됨,이것이 임베딩임
  - 빨간색이 x,자홍색이 y,갈색이 z값임/위의 특정 영화는 0.9,0.2,0.4로 삽입됨 
  
- 임베딩 차원 개수 선택
  - 고차원 임베딩 입력값 간의 관계를 더 정확하게 표현할 수 있음 
  - 하지만 차원이 많아지면 과적합 확률이 높아져 학습 속도가 느려짐 
  - 경험적 법칙은 시작점으로는 좋지만 유효성 확인 데이터를 사용하여 조정해야함
  <img src="https://user-images.githubusercontent.com/32586985/71355207-aecfef00-25c1-11ea-83c3-d30f2db9d6c2.PNG">

- 도구로서의 임베딩
  - 임베딩은 유사한 항목이 서로 근접하도록 항목(예:영화,텍스트)을 저차원의 실제 벡터로 매핑함
  - 의미 있는 유사성 척도를 생성하기 위해 임베딩을 밀집데이터(예:오디오)에 적용할 수도 있음 
  - 다양한 데이터 형식(예:텍스트,이미지,오디오)을 공동으로 임베딩하면 서로 간의 유사성이 정의됨 


### 추가설명
- 협업 필터링에서 필요한 경우
  - 협업 필터링은 다른 여러 사용자의 관심분야를 바탕으로 특정 사용자의 관심 분야를 예측하는 작업임
  
- 범주형 입력 데이터
  - 범주형 데이터란 선택사항이 유한한 집합에 속한 하나 이상의 이산 항목을 표현하는 입력 특성을 가리킴 
  - 대부분의 요소가 0인 희소 텐서를 통해 가장 효율적으로 표현함
  - 여러가지 표현에 있어서 의미론적으로 유사한 항목이 벡터 공간에서 비슷한 거리에 있도록 각 희소 벡터를 숫자 벡터로 표현하는 방법 필요
  - 가장 간단한 방법은 어휘의 모든 단어에 대해 노드가 있는 거대한 입력 레이어를 정의하는 것 
  - 적어도 데이터에 나타나는 각 단어에 대해 노드가 있는 입력 레이어를 정의하는 것
  - 예를 들어 말이라는 단어를 1247번째에 할당한 다음 망에 말을 전달하는 경우 1247번째 입력 노드에 1을, 나머지 노드 전체에 0을 복사할 수 있음 
  - 이러한 유형의 표현은 색인 하나에만 0이 아닌 값이 있기 때문에 원-핫 인코딩이라고 부름 
  - 벡터에 더 큰 텍스트 뭉치에 있는 단어의 수가 포함될 수 있음/BOW 표현으로 알려짐/여러 개의 노드의 값이 0이 아닐 수 있음 
  - 어떤 방식으로든 0이 아닌 값을 결정하더라도 단어당 노드 1개 방식으로는 0이 아닌 값이 상대적으로 거의 없는 매우 큰 벡터인 지극히 희소한 입력 벡터 발생 
  - 희소 표현에는 모델의 효과적인 학습을 어렵게 만들 수 있는 몇 가지 문제점이 있음 
    - 망 크기
      - 입력 벡터가 거대해지면 신경망에 엄청나게 많은 가중치가 만들어짐/M개의 단어가 있고 N개의 노드가 있는경우 MxN개의 가중치를 학습시켜야 함
      - 그리고 가중치의 수가 커지면 다음과 같은 문제가 발생함
        - 데이터의 양:모델의 가중치가 많을수록 효과적인 학습을 위해 더 많은 데이터가 필요함
        - 계산량:가중치가 많을수록 모델을 학습하고 사용하는 데 더 많은 계산이 필요함/하드웨어가 이를 지원하지 못할 가능성이 높음 
    - 벡터 간의 의미 있는 관계 부족
      - 이미지 분류자에 RGB 채널의 픽셀 값을 공급하는 경우 '가까운' 값에 대해 언급할 필요가 있음 
      - 불그스름한 파란색은 의미론적으로든 벡터 간 기하학적 거리로든 순수한 청색에 가까움 
      - '말'에 대한 1247번째에 1을 가진 벡터는 '텔레비전'에 대한 238번째 1을 갖는 벡터보다 '영양'에 대한 50430번쨰 1을 갖는 벡터에 더 가깝지 않음 
    - 솔루션:임베딩
      - 크기가 큰 희소벡터를 의미론적 관계를 보존하는 저차원 공간으로 변환하는 임베딩을 사용하는 것이 해결책
- 저차원 공간으로 변환 
  - 고차원 데이터를 저차원 공간에 매핑하여 희소한 입력 데이터의 핵심 문제를 해결할 수 있음 
  - 작은 다차원 공간에서도 의미론적으로 유사한 항목은 묶고 유사하지 않은 항목은 서로 떨어뜨리는 작업을 자유롭게 수행할 수 있음 
  - 벡터 공간의 위치(거리와 방향)는 좋은 임베딩을 통해 의미론을 인코딩할 수 있음
  - 아래에 나온 실제 임베딩의 시각화 자료는 의미론적인 관계를 캡처하는 기하하적인 관계를 나타냄
  - 이러한 종류의 의미 있는 공간은 머신러닝 시스템이 패턴을 감지하여 학습 작업을 향상 시킬 수 있는 기회를 제공함 
  <img src="https://user-images.githubusercontent.com/32586985/71355984-21da6500-25c4-11ea-95ab-398fb8e49a96.PNG">
  
- 망 축소하기
  - 풍부한 의미론적 관계를 인코딩하기에 충분한 차원이 필요하지만 그와 동시에 시스템을 더 빠르게 학습할 수 있게 할만큼의 작은 임베딩 공간도 필요함
  - 유용한 임베딩은 대략 수백 차원에 달할 수 있음
- 검색표로서의 임베딩
  - 임베딩은 하나의 행렬이고, 행렬의 각 열은 어휘 항목 하나에 대응함/단일 어휘 항목에 대한 밀집 벡터를 얻으려면 해당 항목에 대응하는 열을 검색함 
  - 희소한 BOW(bag of words) 벡터는 어떻게 변환해야할까?/여러 개의 어휘 항목(예:문장 또는 단락의 모든 단어)을 나타내는 희소 벡터에 대한 밀집 벡터를 얻으려면
  - 개별 항목에 대해 임베딩을 검색한 다음 이를 전부 더하면 됨 
  - 희소 벡터에 어휘 항목의 수가 포함되어 있으면 각 임베딩에 해당 항목의 수를 곱한 다음 이를 합계에 추가할 수 있음 
- 행렬 곱셈으로서의 임베딩 검색
  - 방금의 검색,곱셈,덧셈 절차는 행렬 곱셈과 동일함
  - 1xN 크기의 희소표현 S와 NxM 크기의 임베딩 표 E가 주어지면 행렬 곱셈 SxE를 통해 1xM 밀집 벡터를 얻을 수 있음 
  
- 임베딩 획득하기
  - 표준 차원 축소 기법
    - 저차원 공간에서 고차원 공간의 중요한 구조를 캡처할 수 있는 여러 가지 수학적 기법이 존재하는데 이 기법을 통해 머신러닝 시스템용 임베딩에 사용 
  - Word2vec
    - 단어 임베딩을 학습을 위해 만든 알고리즘 
    - 분포 가설에 기반하여 의미론적으로 유사한 단어를 기하학적으로 가까운 임베딩 벡터로 매핑함 
    - 분포 가설은 주로 단어가 인접하는 단어 간에는 의미론적으로 유사한 경향이 있다고 봄 
    - Word2vec은 실제로 함께 등장하는 단어 그룹과 무작위로 그룹화된 단어를 구분하도록 신경망을 학습시켜 이와 같은 문맥상의 정보를 활용함 
    - 입력 레이어는 하나 이상의 문맥 단어와 함께 대상 단어의 희소 표현을 취함/이 입력은 더 작은 히든 레이어 하나에 연결됨 
    - 음의 예를 만드는 경우도 있음/하지만 분류자가 진짜 목표는 아님/모델을 학습한 후에 임베딩을 획득하게 됨
    - 입력 레이어와 히든 레이어를 연결하는 가중치를 사용하여 단어의 희소 표현을 더 작은 벡터에 매핑할 수 있음/다른 분류자에서 재사용가능 
  - 더 큰 모델의 일부로서 임베딩 학습 
    - 임베딩을 대상 작업을 위한 신경망의 일부로서 학습할 수도 있음/특정 시스템에 맞게 임베딩을 효과적으로 맞춤화할 수 있지만 임베딩을 별도로 학습하는 것보다 시간이 오래 걸림 
    - 일반적으로 희소 데이터 또는 임베딩하려는 밀집 데이터가 있는 경우 크기가 d인 특수 유형의 은닉 단위인 임베딩 단위를 만들 수 있음 
    - 이 임베딩 레리어는 다른 특성 및 히든 레이어와 결합할 수 있음
    - 모든 DNN(심층신경망)이 그러하든, 최종 레이어는 최적화 중인 손실임 
    <img src="https://user-images.githubusercontent.com/32586985/71356643-34559e00-25c6-11ea-89e8-35ef7d9c89bb.PNG">
    
    - d차원 임베딩을 학습할 때, 각 항목은 d차원 공간의 지점에 매핑되어 유사한 항목이 이 공간에서 서로 가까이 위치함
    - 아래 그림은 임베딩 레이어에서 학습한 가중치와 기하학적 보기 간의 관계를 보여줌 
    - 입력 노드와 d차원 임베딩 레이어에 있는 노드 간의 에지 가중치는 d개 축 각각의 좌표값과 일치함 
    <img src="https://user-images.githubusercontent.com/32586985/71356748-77b00c80-25c6-11ea-81a2-e8aa3ef3e5bf.PNG">
    
    

## 프로그래밍 실습
- 설정 
```python
   from __future__ import print_function
   
   import collections
   import io
   import math
   
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   %tensorflow_version 1.x
   import tensorflow as tf
   from IPython import display
   from sklearn import metrics
   
   tf.logging.set_verbosity(tf.logging.ERROR)
   train_url = 'https://download.mlcc.google.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
   train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
   test_url = 'https://download.mlcc.google.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
   test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)
```
<img src="https://user-images.githubusercontent.com/32586985/71406939-f9627180-267c-11ea-9e23-19825d4e6e5f.PNG">

- 감정 분석 모델 만들기
  - 이 데이터로 감정 분석 모델을 학습시켜 리뷰가 전반적으로 긍정적(라벨 1)인지 아니면 부정적(라벨 0)인지를 예측해 보겠음 
  - 이를 위해 문자열 값인 단어를 어휘,즉 데이터에 나올 것으로 예상되는 각 단어의 목록을 사용하여 특성 벡터로 변환함 
  - 어휘의 각 단어는 특성 벡터의 좌표에 매핑됨/예의 문자열 값인 단어를 이 특성 벡터로 변환하기 위해
  - 예 문자열에 어휘 단어가 나오지 않으면 각 좌표의 값에 0을 입력하고 어휘 단어가 나오면 1을 입력하도록 인코딩함 
  - 어휘에 나오지 않는 단어는 무시됨
- 입력 파이프라인 구축
```python
   def _parse_function(record):
     """Extracts features and labels.
     
     Args:
       record: File path to a TFRecord file
     Returns:
       A 'tuple' '(labels, features)':
         features: A dict of tensors representing the features
         labels: A tensor with the corresponding labels.
     """
     features = {
       "terms": tf.VarLenFeature(dtype=tf.string), # terms are strings of varying lengths
       "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32) # labels are 0 or 1
     }
     
     parsed_features = tf.parse_single_example(record, features)
     
     terms = parsed_features['terms'].values
     labels = parsed_features['labels']
     
     return {'terms':terms}, labels
```
  - 함수가 정상적으로 작동하는지 확인하기 위해 학습 데이터에 대한 TFRecordDataset를 생성하고 위 함수를 사용하여 데이터를 특성 및 라벨에 매핑함
  ```python
     # Create the Dataset object.
     ds = tf.data.TFRecordDataset(train_path)
     # Map features and labels with the parse function.
     ds = ds.map(_parse_function)
     
     ds
  ```
  <img src="https://user-images.githubusercontent.com/32586985/71407404-3e3ad800-267e-11ea-880a-322a9fc4bc30.PNG">
  
- 다음 셀을 실행하여 학습 데이터 세트에서 첫 예를 확인함 
```python
   n = ds.make_one_shot_iterator().get_next()
   sess = tf.Session()
   sess.run(n)
```
<img src="https://user-images.githubusercontent.com/32586985/71407478-75a98480-267e-11ea-9b1b-1e90f63ed7a5.PNG">

- train() 메소드에 전달할 수 있는 정식 입력 함수를 만듬
```python
   # Create an input_fn that parses the tf.Examples from the given files,
   # and split them into features and targets.
   def _input_fn(input_filenames, num_epochs=None, shuffle=True):
   
     # Same code as above; create a dataset and map features and labels.
     ds = tf.data.TFRecordDataset(input_filenames)
     ds = ds.map(_parse_function)
     
     if shuffle:
       ds = ds.shuffle(10000)
     
     # Our feature data is variable-length, so we pad and batch
     # each field of the dataset structure to whatever size is necessary.
     ds = ds.padded_batch(25, ds.output_shapes)
     
     ds = ds.repeat(num_epochs)
     
     
     # Return the next batch of data.
     features, labels = ds.make_one_shot_iterator().get_next()
     return features, labels
```

### 작업1:희소 입력 및 명시적 어휘와 함께 선형 모델 사용
- 첫 번째 모델로서 50개의 정보 단어를 사용하여 LinearClassifier 모델을 만듬
- 다음 코드는 단어에 대한 특성 열을 만듬/categorical_column_with_vocabulary_list 함수는 문자열과 특성 벡터 간의 매핑을 포함하는 특성열을 만듬 
```python
   # 50 informative terms that compose our model vocabulary.
   informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                        "excellent", "poor", "boring", "awful", "terrible",
                        "definitely", "perfect", "liked", "worse", "waste",
                        "entertaining", "loved", "unfortunately", "amazing",
                        "enjoyed", "favorite", "horrible", "brilliant", "highly",
                        "simple", "annoying", "today", "hilarious", "enjoyable",
                        "dull", "fantastic", "poorly", "fails", "disappointing",
                        "disappointment", "not", "him", "her", "good", "time",
                        "?", ".", "!", "movie", "film", "action", "comedy",
                        "drama", "family")
   terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms", vocabulary_list=informative_terms)                       
```
- LinearClassifier를 생성하고 학습 세트로 학습시킨 후 평가 세트로 평가함
```python
   my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
   my_optimizer = tf.contrib,estimator.clip_gradients_by_norm(my_optimizer, 5.0)
   
   feature_columns = [ terms_feature_column ] 
   
   classifier = tf.estimator.LinearClassifier(
     feature_columns=feature_columns,
     optimizer=my_optimizer,
   )
   
   classifier.train(
     input_fn=lambda: _input_fn([train_path]),
     steps=1000)
     
   evaluation_metrics = classifier.evaluate(
     input_fn=lambda: _input_fn([train_path]),
     steps=1000)
   print("Training set metrics:")
   for m in evaluation_metrics:
     print(m, evaluation_metrics[m])
   print("---")
   
   evaluation_metrics = classifier.evaluate(
     input_fn=lambda: _input_fn([test_path]),
     steps=1000)
   
   print("Test set metrics:")
   for m in evaluation_metrics:
     print(m, evaluation_metrics[m])
   print("---")
```
<img src="https://user-images.githubusercontent.com/32586985/71408199-b1454e00-2680-11ea-992d-f49b8050dd8d.PNG">

### 작업2:심층신경망(DNN) 모델 사용
- 작업1에 선형모델 대신 DNN모델을 사용해 보겠음
```python
   classifier = tf.estimator.DNNClassifier(
     feature_columns=[tf.feature_column.indicator_column(terms_feature_column)],
     hidden_units=[20,20],
     optimizer=my_optimizer,
   )
   
   try:
     classifier.train(
       input_fn=lambda: _input_fn([train_path]),
       steps=1000)
     
     evaluation_metrics = classifier.evaluate(
       input_fn=lambda: _input_fn([train_path]),
       steps=1)
     print("Training set metrics:")
     for m in evaluation_metrics:
       print(m, evaluation_metrics[m])
     print("---")
     
     evaluation_metrics = classifier.evaluate(
       input_fn=lambda: _input_fn([test_path]),
       steps=1)
       
     print("Test set metrics:")
     for m in evaluation_metrics:
       print(m, evaluation_metrics[m])
     print("---")
   except ValueError as err:
     print(err)
```
<img src="https://user-images.githubusercontent.com/32586985/71408427-890a1f00-2681-11ea-9775-2573f2c05387.PNG">

### 작업3:DNN 모델에 임베딩 사용
- 임베딩 열을 사용하여 DNN 모델을 구현함/임베딩 열은 희소 데이터를 입력으로 취하고 저차원 밀집 벡터를 출력으로 반환함 
- 다음과 같은 사양으로 DNNClassifier를 정의함
  - 각각 20개 유닛을 포함하는 히든 레이어 2개
  - Adagrad 최적화, 학습률 0.1
  - gradient_clip_norm을 5.0으로 지정
```python
   terms_embedding_column = tf.feature_column.embedding_column(terms_feature_column, dimension=2)
   feature_column = [ terms_embedding_column ]
   
   my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
   my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
   
   classifier = tf.estimator.DNNClassifier(
     feature_columns=feature_columns,
     hidden_units=[20,20],
     optimizer=my_optimizer
   )
   
   classifier.train(
     input_fn=lambda: _input_fn([train_path]),
     steps=1000)
     
   evaluation_metrics = classifier.evaluate(
     input_fn=lambda: _input_fn([train_path]),
     steps=1000)
   print("Training set metrics:")
   for m in evaluation_metrics:
     print(m, evaluation_metrics[m])
   print("---")
   
   evaluation_metrics = classifier.evaluate(
     input_fn=lambda: _input_fn([test_path])
     steps=1000)
     
   print("Test set metrics:")
   for m in evaluation_metrics:
     print(m, evaluation_metrics[m])
   print("---")  
```
<img src="https://user-images.githubusercontent.com/32586985/71409115-c4a5e880-2683-11ea-84ed-2cb23f860d6a.PNG">

### 작업4:임베딩이 실제로 적용되는지 확인 
- 모델에서 내부적으로 임베딩을 실제로 사용하는지 확인하려는 과정 
- 모델의 텐서 확인 
```python
   classifier.get_variable_names()
```
- 밑의 결과를 통해 임베딩 레이어가 있음을 확인할 수 있음/모델의 다른 부분과 함께 동시에 학습됨
<img src="https://user-images.githubusercontent.com/32586985/71409197-0898ed80-2684-11ea-9cdd-ae1002dae98b.PNG">

- 임베딩 50차원 벡터를 2차원으로 투영하는 행렬임
```python
   classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights').shape
   
   # 결과
   (50, 2)
```

### 작업5:임베딩 조사
- 실제 임베딩 공간을 조사하여 각 단어가 결국 어느 위치에 배치되었는지 확인해봄
  - 1.아래의 코드를 실행하여 작업3에서 학습시킨 임베딩을 확인함
  ```python
     import numpy as np
     import matplotlib.pyplot as plt
     
     embedding_matrix = classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')
     
     for term_index in range(len(informative_terms)):
       # Create a one-hot encoding for our team.  It has 0s everywhere, except for
       # a single 1 in the coordinate that corresponds to that term.
       term_vector = np.zeros(len(informative_terms))
       term_vector[term_index] = 1
       # We'll now project that one-hot vector into the embedding space.
       embedding_xy = np.matmul(term_vector, embedding_matrix)
       plt.text(embedding_xy[0],
                embedding_xy[1],
                informative_terms[term_index])
       
     # Do a little setup to make sure the plot displays nicely.
     plt.rcParams["figure.figsize"] = (15, 15)
     plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
     plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
     plt.show()
  ```
  <img src="https://user-images.githubusercontent.com/32586985/71409646-b658cc00-2685-11ea-8d03-66aa88987316.PNG">
  
  - 위의 작업 3의 코드를 재실행하여 임베딩 시각화를 다시 실행함/이전 실행보다 그래프가 달라짐
  <img src="https://user-images.githubusercontent.com/32586985/71409820-3ed76c80-2686-11ea-8df8-07e0aa4273fe.PNG">
  
  - 작업 3의 코드를 10단계만 사용하여 모델을 다시 학습하여, 임베딩 시각화 실행
  - 위의 두 작업과는 매우 상이한 그래프가 나옴 
  <img src="https://user-images.githubusercontent.com/32586985/71409946-abeb0200-2686-11ea-8a9e-acf245b27728.PNG">

### 작업6:모델 성능 개선 시도
- 초매개변수 변경 또는 Adam등의 다른 옵티마이저 사용, 이 전략으로 향상되는 정확성은 1~2%에 불과할 수 있음
- informative_terms에 더 많은 단어 추가/이 어휘 파일에서 단어를 더 추출할 수도 있고,categorical_column_with_vocabulary_file 특성 열을 통해 전체 어휘를 사용할 수도 있음 
```python
   # Download the vocabulary file.
   terms_url = 'https://download.mlcc.google.com/mledu-datasets/sparse-data-embedding/terms.txt'
   terms_path = tf.keras.utils.get_file(terms_url.split('/')[-1], terms_url)
```
<img src="https://user-images.githubusercontent.com/32586985/71410123-74308a00-2687-11ea-99c2-28d0653cf8d4.PNG">

```python
   # Create a feature column from "terms", using a full vocabulary file.
   informative_terms = None
   with io.open(terms_path, 'r', encoding='utf8') as f:
     # Convert it to a set first to remove duplicates.
     informative_terms = list(set(f.read().split()))
     
   terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms",vocabulary_list=informative_terms)
   
   terms_embedding_column = tf.feature_column.embedding_column(terms_feature_column, dimension=2)
   feature_columns = [ terms_embedding_column ] 
   
   my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
   my_optimizer = tf.contrib.estimator.clip_gradient_by_norm(my_optimizer, 5.0)
   
   classifier = tf.estimator.DNNClassifier(
     feature_columns=feature_columns,
     hidden_units=[10,10],
     optimizer=my_optimizer
   )
   
   classifier.train(
     input_fn=lambda: _input_fn([train_path])
     steps=1000)
   
   evaluation_metrics = classifier.evaluate(
     input_fn=lambda: _input_fn([train_path]),
     steps=1000)
   print("Training set metrics:")
   for m in evaluation_metrics:
     print(m, evaluation_metrics[m])
   print("---")
   
   evaluation_metrics = classifier.evaluate(
     input_fn=lambda: _input_fn([test_path]),
     steps=1000)
   
   print("Test set metrics:")
   for m in evaluation_metrics:
     print(m, evaluation_metrics[m])
   print("---")  
```
<img src="https://user-images.githubusercontent.com/32586985/71410418-87902500-2688-11ea-8333-82d93d923c8b.PNG">


- 임베딩을 사용한 DNN 솔루션이 원래의 선형 모델보다 우수할 수 있지만, 선형 모델도 성능이 그다지 나쁘지 않았으며 학습 속도는 상당히 더 빨라짐
- 선형 모델의 학습 속도가 더 빠른 이유는 업데이트할 매개변수 또는 역전파할 레이어의 수가 더 적기 때문임
- 응용 분야에 따라서는 선형 모델의 빠른 속도가 큰 장점이 될 수 있고, 선형 모델도 품질 면에서 충분하고도 남을 수 있음
- 다른 분야에서는 DNN이 제공하는 추가적인 모델 복잡성과 용량이 더 중요할 수 있음
- 모델 아키텍처를 정의할 때는 어떠한 모델이 적합한지 판단할 수 있도록 문제를 충분히 탐구해야 함
