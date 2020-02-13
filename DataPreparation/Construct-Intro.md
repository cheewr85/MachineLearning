## 데이터세트 구축
- 데이터 세트를 구축하기 위해 아래와 같이 따라야함
  - 1.raw data를 모음
  - 2.feature과 label의 source를 명시하라
  - 3.샘플링 전략을 선택하라
  - 4.데이터를 구분하라
- 해당 과정은 ML 문제에 대해서 어떻게 framed을 했는지에 달려 있음  

### 데이터 수집
- 데이터 세트의 크기 및 품질
  - 데이터 세트의 품질을 측정하고 향상시키며, 유용한 결과를 나오게 하기 위해서는 문제를 해결하는 것에 달려있음

- 데이터 세트의 크기
  - 가장 직관적인 방식으로는 모델에 대해서 최소한 학습할 수 있는 정도보다 그 이상의 예시를 측정하여 학습하는 것임
  - 간단한 모델의 많은 데이터 세트가 일반적임 
  - 많은의 기준은 프로젝트가 무엇이냐에 따라 달려 있음 
  - 이러한 데이터 세트의 대략적인 크기를 비교하라
  <img src="https://user-images.githubusercontent.com/32586985/74428091-b1511580-4e9b-11ea-99e1-9af5a3dddcad.PNG">
  
- 데이터 세트의 품질
  - 많은 데이터를 쓰지 않는 것은 좋은 데이터가 아닌것과 같이 품질 역시 같음
  - 품질을 측정하는 것은 어렵기 때문에 경험적인 접근법이난 최고의 결과를 나타내는 옵션을 선택하는 것을 고려해봐야 함
  - 이러한 마인드셋으로 품질이 좋은 데이터가 문제를 해결하는데 있어서 성공적일 것임
  - 다른 한편으론 데이터가 좋기 위해서는 몇 가지 선행되는 과제가 필요할 것임
  - 데이터를 모으는데 있어서 품질이라는 정의에 좀 더 집중하는 것이 도움이 될 것임
  - better-performing model로 여겨지는 품질의 측면이 있음
    - reliability
    - feature representation
    - minimizing skew
    
- Reliability    
  - Reliability는 당신의 데이터를 얼마나 신뢰하는지에 대한 척도를 말함
  - reliable한 데이터 세트를 통해 학습한 모델은 unreliable한 데이터 세트로 학습한 모델보다 더 유용한 예측을 생산해냄
  - reliability를 측정하는데 있어서 반드시 결정해야 할 것이 있음
    - 얼마나 label의 에러가 있는가? / 만일 데이터를 사람이 labeled 했으면 가끔씩 실수가 있을 수 있음
    - features들이 noisy한가? / GPS 측정시 데이터 세트에 대한 모든 noise를 확인할 수 없음
    - 데이터가 정확하게 문제를 filter하는가? / 시스템을 구축하였지만 발전시키는 과정에서 혼선이 생겨 원하는 결과와 정반대가 될 수 있음
  - unreliable하게 데이터를 만드는 것
    - values를 빠뜨리는 것
    - 예시를 똑같이 쓰는 것
    - 좋지 않은 label
    - 좋지 않은 feature values
    
- Feature Representation
  - Representation은 유용한 features에 대한 데이터를 mapping함 / 해당 사항에 대해서 생각해보아야함
    - 모델에 대해 데이터가 어떻게 보이는가?
    - 수 적인 values에 대해서 일반화 할 수 있는가?
    - 이상점에 대해서 조절할 수 있는가?

- Training versus Prediction
  - 오프라인으로 좋은 결과치를 얻었다면 이것은 live 경험에는 도움이 되지 않음
  - 이러한 문제는 training/serving skew를 나타냄
  - 즉, metrics에서의 training time과 serving time에 계산의 차이로 나타남
  - 이러한 차이는 커보이지 않더라도 결과에 대해서는 큰 차이가 보임 
  - 항상 모델에 예측 시간을 어떠한 데이터가 사용가능한지 고려해야함
  - 학습하는 동안 serving을 하는데 이용가능한 특징을 사용하고 serving traffic이 나타나는 training set에 대해서 명확히하라
  - 예측하려는만큼 학습시켜라! / 즉 training task가 prediction task에 잘 맞을수록 더욱 더 ML 시스템이 잘 구현될 것임

