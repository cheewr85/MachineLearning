## 텐서플로우 도구함의 계층구조
<img src="https://user-images.githubusercontent.com/32586985/68846333-419e8380-0710-11ea-8649-ff3abffef7a8.PNG">

도구함 | 설명
---- | ----
에스티메이터(tf.estimator)|높은 수준의 OOP API
tf.layers/tf.losses/tf.metrics|일반 모델 구성요소용 라이브러리
텐서플로우|낮은 수준의 API

### 두 요소로 구성됨  
- 그래프 프로토콜 버퍼
- 분산된 그래프를 실행하는 런타임 
  - 여러 CPU와 GPU에서 구현됨
- 문제를 해결하는 최고 수준의 추상화를 사용해야함 
  - 추상화 수준이 높을수록 더 사용하기 쉽지만 (설계상) 유연성이 떨어짐
- 특별한 모델링 문제를 해결하기 위해 더 유연한 추상화가 필요하면 한 수준 아래로 이동함
  - 각 수준은 낮은 수준의 API를 사용하여 제작되므로 계층구조를 낮추는 것이 합리적임


### tf.estimator API
- 현 과정에서 사용할 API
- 코드 행 수를 크게 줄일 수 있음 
- scikit-learn API와 호환됨/Python에서 사용하는 오픈소스 ML 라이브러리
- tf.estimator로 구현된 선형 회귀 프로그램의 형식
```python
   import tensorflow as tf
   
   # Set up a linear classifier.
   classifier = tf.estimator.LinearClassifier()
   
   # Train the model on some example data.
   classifier.train(input_fn=train_input_fn, steps=2000)
   
   # Use it to predict.
   predictions = classifier.predict(input_fn=predict_input_fn)
```


