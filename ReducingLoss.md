# 손실 줄이기
모델을 학습하기 위해서 모델의 손실을 줄이기 위한 방법  

## 손실을 줄이는 방법
- 가중치와 편향에 대한 도함수 (y-y')^2는 주어진 예제의 손실 변화 정도를 보여줌
  - 계산하기 간편하며 볼록 모양 
- 손실을 최소화하는 방향으로 작은 보폭을 반복하여 취함 
  - 기울기 보폭이라고 함(음의 기울기 보폭)
  - 이 최적화 전략을 경사하강법이라고 함  

## 가중치 초기화
- 볼록 문제에서는 가중치가 임의의 값을 가질 수 있음(예: 모두0)
  - 볼록:그릇 모양을 생각하면 됨  
  - 최소값은 단 하나임 
- 예고(foreshadowing): 신경망에서는 해당 없음
  - 볼록하지 않음: 계란판 모양을 생각
  - 최소값이 둘 이상 있음
  - 초기값에 따라 크게 달라짐

## SGD와 미니 배치 경사 하강법
- 각 보폭마다 전체 데이터 세트에 대해 기울기를 계산할 수 있지만 그럴 필요는 없음
- 적은 양의 데이터 샘플에서 기울기 계산이 잘 작동함 
  - 모든 보폭에서 새로운 무작위 샘플을 얻음
- 확률적 경사하강법:한 번에 하나의 예
- 미니 배치 경사하강법:10~1000개의 예로 구성된 배치
  - 손실과 기울기는 배치 전반에 걸쳐 평균 처리됨
  
## 반복 방식
- 반복 학습을 통해 최적 모델을 찾는것 
- 처음에는 임의의 지점에서 시작해서 시스템이 손실 값을 알려줄 때까지 기다림
- 그 이후 다른 값을 추정해서 손실 값을 확인함 
- 최적의 모델을 가능한 한 가장 효율적으로 찾는것 

- 모델을 학습하는 데 사용하는 반복적인 시행착오 과정/반복 방식의 모델 학습 
<img src="https://user-images.githubusercontent.com/32586985/68591019-20048880-04d3-11ea-8c6e-fd410f44a79c.png">

- 반복 전략은 주로 대규모 데이터 세트에 적용하기 용이함  
- 하나 이상의 특성을 입력하여 하나의 예측(y')을 출력하는 모델  
<img src="https://user-images.githubusercontent.com/32586985/68591308-c6e92480-04d3-11ea-9ae1-1494e3856eb0.png">

- 선형 회귀 문제에서는 초기값이 중요하진 않음 임의의 값을 정해도 됨
  - b=0,w1=0
  - 최초 특성 값을 10이라고 가정
  
  ```octave
     y' = 0 + 0(10)
     y' = 0
     % 다음과 같이 출력됨 
  ```     
  
- 위의 다이어그램에서 손실 계산 과정은 이 모델에서 사용할 손실함수임  
- 제곱 손실 함수를 사용한다고 가정한다면 y': 특성 x에 대한 모델의 예측 값/y:특성 x에 대한 올바른 라벨 두개의 입력 값 사용
- 다이어그램의 매개변수 업데이트 계산 과정에 도달/머신러닝 시스템은 손실 함수의 값을 검토하여 b와 w1의 새로운 값을 생성함 
- 새로운 값을 만든 다음 머신러닝 시스템이 이러한 모든 특성을 모든 라벨과 대조하여 재평가하여 손실 함수의 새로운 값을 생성하여 새 매개변수 값을 출력한다고 가정
- 손실 값이 가장 낮은 모델 매개변수를 발견할때까지 반복 학습함
- 전체 손실이 변하지 않거나 매우 느리게 변할 때까지 계속 반복함/이때 모델이 수렴했다고 말함


## 경사하강법  
- w1의 가능한 모든 값에 대해  손실을 계산할 시간과 컴퓨팅 자료가 있다고 가정  
- 손실과 w1을 대응한 도표는 항상 블록 함수 모양을 함 
- 회귀 문제에서는 볼록 함수 모양의 손실 대 가중치 도표가 산출됨  
<img src="https://user-images.githubusercontent.com/32586985/68592031-7a9ee400-04d5-11ea-9bd5-58bec6f7c5b3.png">

- 볼록 문제에는 기울기가 정확하게 0인 지점인 최소값이 하나 존재함/이 최소값에서 손실 함수가 수렴함  
- 전체 데이터 세트에 대해 모든 손실 함수를 계산하는 것은 수렴 지점을 찾는데 비효율적인 방법
- 더 효율적인 방법인 경사하강법이 있음

- 경사하강법의 첫 번째 단계는 w1에 대한 시작 값(시작점)을 선택하는 것 
- 시작점은 별로 중요하지 않음/w1을 0으로 설정하거나 임의의 값을 선택함  
<img src="https://user-images.githubusercontent.com/32586985/68592295-09136580-04d6-11ea-9bad-7963ff173e28.png">

- 시작점에서 손실 곡선의 기울기를 계산함
- 기울기는 편미분의 벡터로서 어느 방향이 더 정확한지 혹은 더 부정확한지 알려줌 
- 단일 가중치에 대한 손실의 기울기는 미분값과 같음  
- 기울기는 벡터이므로 방향과 크기의 특성을 모두 가지고 있음  
- 기울기는 항상 손실 함수 값이 가장 크게 증가하는 방향을 향함
- 경사하강법 알고리즘은 가능한 한 빨리 손실을 줄이기 위해 기울기 반대 방향으로 이동함
- 경사하강법은 음의 기울기를 사용함  
<img src="https://user-images.githubusercontent.com/32586985/68592518-a1114f00-04d6-11ea-95ae-53f8413b6749.png">

- 손실 함수 곡선의 다음 지점을 결정하기 위해 기울기의 크기의 일부를 시작점에 더함 
- 기울기 보폭을 통해 손실 곡선의 다음 지점으로 이동함  
<img src="https://user-images.githubusercontent.com/32586985/68592748-272d9580-04d7-11ea-8021-b5f47d1fa189.png">

- 이 과정을 반복해 최소값에 점점 접근함 

