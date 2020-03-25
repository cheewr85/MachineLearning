## Step 4:Build, Train, and Evaluate Your Model
- 이제 모델을 building, training, evaluating하는 것을 할 것이고 / Step 3에서 n-gram모델이나 sequence model중 하나를 S/W 비율을 사용해서 선택해 사용할 것임 / 이제 classification algorithm을 쓰고 학습할 시기임 / TensorFlow의 keras API를 사용할 것임 
- keras와 함께 ML모델을 만드는것은 layers를 한 곳에 합치는것이고, data-processing building blocks, Lego bricks를 모으는 것 같은 것임 
- 이 layers은 our input에 우리가 원하는 정도의 수행을 할 수 있는 transformation의 sequence를 특정화하는 걸 허락하게함
- learning algorithm을 single text input과 single classification의 outputs으로 가지게 함으로써 Sequential model API를 사용함으로써 layers의 선형적인 stack을 만들 수 있음
<img src="https://user-images.githubusercontent.com/32586985/77385642-0b62c600-6dcc-11ea-8760-7049888cb046.PNG">

- input layer과 intermediate layers이 서로 다르게 constructed될 것이고, n-gram과 sequence model을 만드는데 달려있음 / 하지만 model type의 irrespective가 last layer에 주어진 문제에 같게 나타날 것임 

- Constructing the Last Layer
  - 오직 2개의 classes(binary classification)만을 가지고 있을 때, 모델은 single probability score가 output이어야만함 
  - 예를들어 주어진 input sample에 outputtin이 0.2라면 20%만이 이 sample이 first class(class 1)에 확실히 있고 80%가 second class(class 0)이 있음을 의미함 
  - 이러한 probability score의 output을 가지는것과 last layer의 activation function은 sigmoid function이어야하고, 모델을 학습하는데 사용된 loss function은 binary cross-entropy여야만 함(Figure 10의 왼쪽)
  - 만일 2 classes이상 있다면 (multi-class classification), 모델은 각 class마다 one probability score가 output이어야 함 
  - 이러한 scores의 합은 반드시 1이어야함 / 예를들어 outputting {0: 0.2, 1: 0.7, 2: 0.1}은 sample의 20%만이 class 0에 있고, 70%는 class 1에 있고, 10%는 class 2에 있다는것을 의미함 
  - 이러한 scores의 output은 last layer의 activation function은 softmax여야만하고 모델을 학습하는데 사용되는 loss function은 categorical cross-entropy여야만 함(Figure 10의 오른쪽)
  <img src="https://user-images.githubusercontent.com/32586985/77481776-b5dbf700-6e67-11ea-916e-4027fff4bae5.PNG">
  
  - 아래의 코드는 input으로써 classes의 수를 가지는 function을 정의하고, layer units의 정확한 수를 outputs로 하며(1 unit for binary classification, otherwise 1 unit for each class)그리고 정확한 activation function을 outputs로 함 
  ```python
     def _get_last_layer_units_and_activation(num_classes):
     """Gets the # units and activation function for the last network layer.

     # Arguments
         num_classes: int, number of classes.

     # Returns
         units, activation values.
     """
     if num_classes == 2:
         activation = 'sigmoid'
         units = 1
     else:
         activation = 'softmax'
         units = num_classes
     return units, activation
  ```
  
  - 다음의 2개의 sections은 n-gram models과 sequence models의 model layers을 남긴 creation을 확인해볼 것임
  - S/W 비율이 작다면, n-gram models이 sequence models보다 더 잘 수행하는 것을 알 수 있고 sequence models은 small,dense vectors의 수가 클 때 더 잘 수행됨 / 이것은 embedding relationships이 dense space에서 학습하기 때문이고, 그리고 이것은 많은 samples에서의 나타남

- Build n-gram model(Option A)
  - n-gram models로써 tokens이 독립적으로 process하는 models을 언급한 것임(not taking into account word order) / Simple multi-layer perceptrons(including logistic regression), gradient boosting machines and support vector machines models모두 이 카테고리 안에 있고, 그것은 text ordering에 대해서 어떠한 정보도 leverage 할 수 없음 
  - 위에서 언급한 몇몇 n-gram models의 수행능력을 비교했고, multi-later perceptrons(MLPs)가 일반적으로 다른 options보다 더 잘 수행됨을 확인했음 / MLPs는 정의하고 이해하기 쉬우며, 좋은 accuracy를 제공하며, 상대적으로 적은 계산을 필요로함 
  - 아래의 코드는 Dropout layers for regularization의 couple을 추가함으로써 two-layer MLP models을 (tf.keras에) 정의해줌 (training samples을 overfitting하는것을 막기위해서)
  ```python
     from tensorflow.python.keras import models
     from tensorflow.python.keras.layers import Dense
     from tensorflow.python.keras.layers import Dropout
     
     def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
         """Creates an instance of a multi-layer perceptron model.

     # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

     # Returns
         An MLP model instance.
     """
     op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
     model = models.Sequential()
     model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

     for _ in range(layers-1):
         model.add(Dense(units=units, activation='relu'))
         model.add(Dropout(rate=dropout_rate))

     model.add(Dense(units=op_units, activation=op_activation))
     return model     
  ```

- Build sequence model(Option B)
  - sequence models의 tokens의 adjacency로부터 학습할 수 있는 모델을 볼 수 있음
  - 이것은 models의 CNN, RNN classes를 포함함 / Data는 이러한 모델의 sequence vectors로써 미리 생성됨 
  - Sequence models은 일반적으로 배우기 위한 parameters의 많은 수를 가지고 있음 
  - 이 모델에서 first layer은 dense vector space에서의 단어 사이에 관계를 학습하는 embedding layer임 / 많은 예제를 통해서 word relationships을 배우는게 좋음
  - 주어진 dataset의 words는 dataset에서 unique하지 않을 수 있음 / 그래서 다른 datasets을 사용해서 우리의 dataset의 단어들 사이에 관계를 학습할 수 있음 / 이렇게 하기 위해서, 우리는 우리의 embedding layer에 또 다른 dataset으로부터 학습된 embedding으로부터 transfer할 수 있음 / 이러한 embedding을 pre-trained embeddings라고함 / pre-trained embedding을 사용함으로써 models이 learning process에서 head start를 줌을 알 수 있음 
  - GloVe같은 large corpora를 사용하여 학습을 할 수 있는 pre-trained embeddings이 있음 / GloVe는 multiple corpora로 학습함 / GloVe embeddings버전을 이용해서 sequence models을 학습하는 것을 테스트할 수 있고 만약 pre-trained embeddings에 대해서 가중치를 고정시키고 
  network의 나머지 부분을 학습시킨다면 models은 잘 작동하지 않을 것임을 확인할 수 있음 / 이것은 embedding layer이 학습된 context가 우리가 사용하고 있는 context와는 다를 수 있기 때문일 수도 있음 
  - Wikipedia data를 학습한 GloVe embeddings은 IMDb dataset의 language patterns와 align되지 않을 수 있음 / 이 관계는 updating이 필요로 함
  - 예를들어 embedding weights가 contextual tuning을 할 수도 있을 때 아래의 2가지를 고려함
    - 처음 실행시, embedding layer weights를 고정하고 network의 나머지를 학습하게끔 함 / 이 실행 마지막에는 model weights가 그것의 uninitialized values보다 더 나은 상태에 도달하게 됨 / 두 번째 실행시, embedding layer가 학습할 수 있고, network의 모든 weights를 적합한 adjustments를 만듬 / 이러한 과정을 fine-tuned embedding를 이용한다고 함 
    - Fine-tuned embeddings는 더 나은 정확성을 생산함 / 하지만 이것은 network를 학습하는데 필요로 한 compute power의 증가되는 expense를 일으킴 / 충분한 수의 samples가 있다면, scratch로부터 embedding을 잘 학습할 수 있음 / S/W > 15K를 관찰하며, scratch로부터 시작해 fine-tuned embedding을 사용함으로서 같은 정확성에 대한 효율적으로 생산할 수 있음 
  - CNN, sepCNN(Depthwise Separable Convolutional Network), RNN(LSTM & GRU), CNN-RNN, stacked RNN, 다양한 모델 구조들 같은 것으로 서로다른 sequence models을 비교할 수 있음 / sepCNNs, convolutional network variant는 더욱 더 data-efficient하고 compute-efficient이며 다른 모델들보다 더 잘 구동되는 것을 발견함 
  ```python
     from tensorflow.python.keras import models
     from tensorflow.python.keras import initializers
     from tensorflow.python.keras import regularizers

     from tensorflow.python.keras.layers import Dense
     from tensorflow.python.keras.layers import Dropout
     from tensorflow.python.keras.layers import Embedding
     from tensorflow.python.keras.layers import SeparableConv1D
     from tensorflow.python.keras.layers import MaxPooling1D
     from tensorflow.python.keras.layers import GlobalAveragePooling1D

     def sepcnn_model(blocks,
                     filters,
                     kernel_size,
                     embedding_dim,
                     dropout_rate,
                     pool_size,
                     input_shape,
                     num_classes,
                     num_features,
                     use_pretrained_embedding=False,
                     is_embedding_trainable=False,
                     embedding_matrix=None):
         """Creates an instance of a separable CNN model.

         # Arguments
             blocks: int, number of pairs of sepCNN and pooling blocks in the model.
             filters: int, output dimension of the layers.
             kernel_size: int, length of the convolution window.
             embedding_dim: int, dimension of the embedding vectors.
             dropout_rate: float, percentage of input to drop at Dropout layers.
             pool_size: int, factor by which to downscale input at MaxPooling layer.
             input_shape: tuple, shape of input to the model.
             num_classes: int, number of output classes.
             num_features: int, number of words (embedding input dimension).
             use_pretrained_embedding: bool, true if pre-trained embedding is on.
             is_embedding_trainable: bool, true if embedding layer is trainable.
             embedding_matrix: dict, dictionary with embedding coefficients.

         # Returns
             A sepCNN model instance.
         """
         op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
         model = models.Sequential()

         # Add embedding layer. If pre-trained embedding is used add weights to the
         # embeddings layer and set trainable to input is_embedding_trainable flag.
         if use_pretrained_embedding:
             model.add(Embedding(input_dim=num_features,
                                 output_dim=embedding_dim,
                                 input_length=input_shape[0],
                                 weights=[embedding_matrix],
                                 trainable=is_embedding_trainable))
         else:
             model.add(Embedding(input_dim=num_features,
                                 output_dim=embedding_dim,
                                 input_length=input_shape[0]))

         for _ in range(blocks-1):
             model.add(Dropout(rate=dropout_rate))
             model.add(SeparableConv1D(filters=filters,
                                       kernel_size=kernel_size,
                                       activation='relu',
                                       bias_initializer='random_uniform',
                                       depthwise_initializer='random_uniform',
                                       padding='same'))
           model.add(SeparableConv1D(filters=filters,
                                     kernel_size=kernel_size,
                                     activation='relu',
                                     bias_initializer='random_uniform',
                                     depthwise_initializer='random_uniform',
                                     padding='same'))
           model.add(MaxPooling1D(pool_size=pool_size))

           model.add(SeparableConv1D(filters=filters * 2,
                                     kernel_size=kernel_size,
                                     activation='relu',
                                     bias_initializer='random_uniform',
                                     depthwise_initializer='random_uniform',
                                     padding='same'))
           model.add(SeparableConv1D(filters=filters * 2,
                                     kernel_size=kernel_size,
                                     activation='relu',
                                     bias_initializer='random_uniform',
                                     depthwise_initializer='random_uniform',
                                     padding='same'))
           model.add(GlobalAveragePooling1D())
           model.add(Dropout(rate=dropout_rate))
           model.add(Dense(op_units, activation=op_activation))
           return model
  ```
- Train Your Model
  - model architecture를 설계할 때, model을 학습할 필요가 있음 / Training은 model의 현재 상태를 기반으로 한 prediction을 통해서 만드는 것을 포함하고 prediction이 어떻게 부정확하게 계산했는지를 포함하고 그리고 이러한 error를 최소화하기 위해서 그리고 model의 predict를 더 잘하게 만들기 위한 것을 network의 parameters와 weights를 updating하는 것을 포함함 / 우리의 모델이 converged되고 더 이상 학습을 안 할 때까지 이 과정을 반복할 것임 / 이 과정을 위해 선택된 3개의 key parameters가 있음 
    - Metric:metric을 이용하는 model의 performance를 어떻게 측정하는지 볼 것임 / 실험에서 metric으로써 accuracy를 사용할 것임
    - Loss function:학습은 etwork weights를 tuning함으로써 최소화하려는 시도를 생산하는 loss value를 계산하는데 사용되는 함수임 
    - Optimizer:어떻게 loss function의 output에 기반하여 updated되는지 결정하는 함수임 / 실험에서 Adam optimizer를 사용할 것임
  - Keras에서는 complie method를 사용하여 model의 learning parameters를 pass할 것임
  <img src="https://user-images.githubusercontent.com/32586985/77545229-79eb7500-6eed-11ea-8297-52f1a24976a8.PNG">
  
  - 실제 training은 fit method를 사용할 때 일어남 / dataset의 크기에 따라서 이 방법은 대부분의 compute cycles이 쓸 것임 / 각각의 training iteration에서 training data로부터 samples의 batch_size의 수는 loss를 계산하는데 사용되고, 이 값에 근거하여 가중치가 한 번 updated됨 / training process는 전체 training dataset에서 보여지는 모델의 epoch로 완료됨 / 각각의 epoch끝에 여기서 validation dataset을 모델이 얼마나 잘 학습하는지 평가하는 것으로 사용할 수 있음 / epoch의 이미 정해진 수를 위한 dataset을 사용함으로서 학습을 반복할 수 있음 / validation accuracy가 consecutive epochs사이에 stabilizes할 때 더 빨리 멈춤으로써, 모델이 더이상 학습을 보이지 않을때 이것을 최적화할 수 있음 
  <img src="https://user-images.githubusercontent.com/32586985/77545895-71e00500-6eee-11ea-9a09-dc7028afef6d.PNG">
  
  - 아래의 코드는 Table 2 & 3에서 parameters를 선택하여 사용한 training process임 
  ```python
      def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = explore_data.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val = vectorize_data.ngram_vectorize(
        train_texts, train_labels, val_texts)

    # Create model instance.
    model = build_model.mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('IMDb_mlp_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]
  ```

## Step 5:Tune Hyperparameters
- 
