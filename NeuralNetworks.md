## 신경망 
- 비선형 분류 문제
  - 비선형/b+w1x1+w2x2형태의 모델로 라벨을 정확하게 예측할 수 없다는 의미임
  - 결정 표면은 선이 아님
  <img src="https://user-images.githubusercontent.com/32586985/70856668-306db000-1f24-11ea-903a-bd4410423072.PNG">
  
- 복잡한 비선형 분류 문제의 데이터 세트
<img src="https://user-images.githubusercontent.com/32586985/70856679-4ed3ab80-1f24-11ea-91a6-72de222aeb7d.PNG">

- 신경망이 비선형 문제를 해결하는 데 어떻게 도움이 되는지 알아보기 위해 선형 모델을 그래프로 나타냄
- 각 파란색 원은 입력 기능을 나타내고, 녹색 원은 입력의 가중합을 나타냄 
<img src="https://user-images.githubusercontent.com/32586985/70856703-d0c3d480-1f24-11ea-85f6-98516ea05fd1.PNG">

### 히든 레이어
- 중간값의 히든 레이어를 추가함/각 노란색 노드는 파란색 입력 노드 값의 가중합임/출력은 노란색 노드의 가중합임
- 여전히 입력의 선형 조합임 
<img src="https://user-images.githubusercontent.com/32586985/70856719-0799ea80-1f25-11ea-9feb-c88e6a56a863.PNG">

- 가중합의 두번째 히든 레이어를 추가함 
- 여전히 선형 모델임/출력을 입력의 함수로 표현하고 단순화하면 입력의 또 다른 가중합을 얻게됨
- 이 합계는 비선형 문제를 효과적으로 모델링하지 않음 
<img src="https://user-images.githubusercontent.com/32586985/70856736-65c6cd80-1f25-11ea-957e-df2136cf9cc5.PNG">

### 활성화 함수
- 비선형 문제를 모델링하기 위해 비선형성을 직접 도입할 수 있음/각 히든 레이어의 노드가 비선형 함수를 통과하도록 할 수 있음 
- 아래의 그래프로 나타낸 모델에서 히든 레이어1의 각 노드 값이 비선형 함수로 변환한 후에 다음 레이어의 가중 합으로 전달됨 
- 이 비선형 함수를 활성화 함수라고 함
- 활성화 함수를 추가하였으므로 레이어를 추가하면 효과가 더 큼
- 비선형성을 누적하면 입력과 예측 출력 간의 매우 복잡한 관계를 모델링 할 수 있음 
- 각 레이어는 원시 입력에 적용되는 더 높은 수준의 복잡한 함수를 효과적으로 학습함
<img src="https://user-images.githubusercontent.com/32586985/70856750-b76f5800-1f25-11ea-9121-2525ddb95a2e.PNG">

- 일반적인 활성화 함수
  - 시그모이드 활성화 함수는 가중합을 0과 1사이의 값으로 변환함
  <img src="https://user-images.githubusercontent.com/32586985/70856771-0e752d00-1f26-11ea-83c7-6a37064f7f69.PNG">
  
  - 구성은 다음과 같음
  <img src="https://user-images.githubusercontent.com/32586985/70856776-2c429200-1f26-11ea-85ab-51e5fa760b33.PNG">
  
- 정류 선형 유닛(ReLU)활성화 함수
  - 시그모이드와 같은 매끄러운 함수보다 조금 더 효과적이지만, 훨씬 쉽게 계산할 수 있음 
  <img src="https://user-images.githubusercontent.com/32586985/70856790-6875f280-1f26-11ea-8a4e-1498662db158.PNG">
  
  - ReLU의 반응성 범위가 더 유용함
  <img src="https://user-images.githubusercontent.com/32586985/70856791-6ad84c80-1f26-11ea-9dfd-3b82996579ab.PNG">
  
- 어떠한 수학 함수라도 활성화 함수의 역할을 할 수 있음
- σ가 활성화 함수(ReLU,시그모이드등)를 나타낸다고 가정한다면
- 결과적으로 네트워크 노드 값은 다음 수식으로 나타냄
<img src="https://user-images.githubusercontent.com/32586985/70856818-e0dcb380-1f26-11ea-9d57-96965155a5ae.PNG">


## 실습
- 처음 만들어보는 신경망
  - 과제1.주어진 모델은 두 개의 입력 특성을 하나의 뉴련으로 결합함/이 모델이 비선형성을 학습할 수 있을까?
  - 활성화가 선형으로 설정되어 있으므로 이 모델은 어떠한 비선형성도 학습할 수 없음/손실은 매우 높음 
  <img src="https://user-images.githubusercontent.com/32586985/70856905-0e762c80-1f28-11ea-9f43-8d5d358260bf.PNG">
  
  - 과제2.히든 레이어의 뉴런 수를 1개에서 2개로 늘려보고 선형 활성화에서 ReLU와 같은 비선형 활성화로 변경한다면 비선형성을 학습하나?
  - 비선형 활성화 함수는 비선형 모델을 학습할 수 있음/뉴런이 2개인 히든 레이어는 모델을 학습하는데 시간이 오래 걸림 
  - 이 실습은 비결정적이므로 일부 시도에서는 효과적인 모델을 학습하지 못함/다른 시도에서는 상당히 효과적으로 학습할 수 있음 
  <img src="https://user-images.githubusercontent.com/32586985/70856939-965c3680-1f28-11ea-9d95-2167e468b2a1.PNG">
  
  - 과제3.히든 레이어 및 레이어당 뉴런을 추가하거나 삭제하여 실험을 계속해본다/자유롭게 설정을 변경
  - 테스트 손실을 0.177이하로 얻는데 사용할 수 있는 가장 적은 노드 및 레이어수는 얼마인가?
  - 히든 레이어 3개의 테스트 손실이 매우 낮음
    - 1레이어에는 뉴런이 3개 있음/2레이어에는 뉴런이 3개 있음/3레이어에는 뉴런이 2개있음
  - 정규화도 L1정규화로 함
  <img src="https://user-images.githubusercontent.com/32586985/70856987-e687c880-1f29-11ea-90f7-773c71522f5b.PNG">
  
- 신경망 초기화
- XOR 데이터를 사용하여, 학습용 신경망의 반복성과 초기화의 중요성을 살펴봄
  - 과제1.주어진 모델을 4~5회 실행함/매번 시도하기 전에 네트워크 초기화 버튼을 눌러 임의로 새롭게 초기화함
  - 최소 500단계를 실행하도록 함/각 모델 출력이 어떤 형태로 수렴하나?/이 결과가 비볼록 최적화에서 초기화의 역할에 어떤 의미를 가지는가?
  - 각 시도마다 학습된 모델의 형태가 달라짐/테스트 손실 수렴 결과는 최저와 최고가 거으 2배까지 차이가 날 정도로 다양했음 
  <img src="https://user-images.githubusercontent.com/32586985/70857147-cd344b80-1f2c-11ea-977f-37b11b25342b.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857153-d9b8a400-1f2c-11ea-9d01-e1e46a3e0ec7.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857157-e3daa280-1f2c-11ea-88ac-0be4c2abb63a.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857162-efc66480-1f2c-11ea-8d4d-a01895213dd0.PNG">
  
  - 과제2.레이어 한 개와 추가 노드를 몇 개 더 추가하여 모델을 약간 더 복잡하게 만듬/과제1의 시도를 반복/결과에 안정성이 보강되나?
  - 레이어와 추가 노드를 추가하여 더 반복적인 결과를 얻음/매 시도마다 결과 모델은 거의 같은 형태였음
  - 테스트 손실 수렴 결과는 매 시도마다 변화가 적음 
  <img src="https://user-images.githubusercontent.com/32586985/70857209-b510fc00-1f2d-11ea-989e-35b3483907a8.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857223-c8bc6280-1f2d-11ea-9eb8-e8df883ce879.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857228-cce88000-1f2d-11ea-8b93-d5f7206d5354.PNG">
  <img src="https://user-images.githubusercontent.com/32586985/70857231-d2de6100-1f2d-11ea-9733-e130503ae981.PNG">
  
- 나선형 신경망 
- 잡음이 있는 나선형임/선형 모델은 실패하고 직접 정의된 특성 교차도 구성이 어려울 수 있음 
