## 텍스트분류

### Introduction
  - Text classification algorithms은 scale에서 텍스트 데이터를 생성하는 소프트웨어 시스템에서의 다양성의 핵심임
  - Email software은 들어오는 mail이 inbox에 들어가거나 spam folder에 필터링 되는지 결정하는데 text classification을 사용함
  - topic classification과 text document를 이미 설정된 topic 중 하나로 categorizing하는 두 가지의 예시가 있음
  - 많은 topic classification 문제에서 이 categorization은 text의 keyword를 주로 기반함
  <img src="https://user-images.githubusercontent.com/32586985/76317931-4bcc3980-6320-11ea-8660-72c086d3711a.PNG">
  
  - 또 다른 text classification의 흔한 종류는 그것이 표현한 의견의 종류인 text content의 polarity를 정의하는 것을 목표로 하는 sentiment analysis임 
  - 이것은 like/dislike rating의 binary form으로 만들거나, options의 set을 더 세분화하거나, star rating을 1~5로 정도로
  - sentiment analysis예제는 사람들이 블랙팬서 영화를 좋아하는지 확인하기 위해서 Twitter posts를 분석하거나 Walmart review를 통해서 Nike 신발의 새로운 브랜드에 대한 대중들의 의견이 지속될지를 포함함 
  - 해당 과정을 통해서 text classification 문제를 해결하는데 최고의 적용법과 몇개의 머신러닝의 키를 배우게 될 것임
    - The high-level, end-to-end workflow for solving text classification problems using machine learning
    - How to choose the right model for your text classification problem 
    - How to implement your model of choice using TensorFlow 
  - Text Classification Workflow
    - 아래의 워크플로우를 통해서 machine learning 문제를 해결할것임
      - Step1:Gather Data
      - Step2:Explore Your Data
      - Step2.5:Choose a Model
      - Step3:Prepare Your Data
      - Step4:Build,Train,and Evaluate Your Model
      - Step5:Tune Hyperparameters
      - Step6:Deploy Your Model
      <img src="https://user-images.githubusercontent.com/32586985/76318969-da8d8600-6321-11ea-8473-57663501f405.PNG">
      
### Step 1:Gather Data
  - 데이터를 모으는것은 어떠한 supervised machine learning 문제를 해결하는데 있어서 중요한 과정임
  - text classifier은 제대로 만들어진 dataset으로부터만 좋은 성능을 발휘함
  - 만약 해결하고 싶거나 흥미있는 특정 문제가 없다면, 일반적으로 text classification을 탐색하게 된다면, 충분한 양의 오픈소스 데이터 세트를 제공할 것임   
  - 이와 반대로, 만약 특정 문제를 다루고 싶다면, 필요한 데이터에 대해서 모을 필요가 있음 
  - Many organizations이 그들 데이터를 접근하는데 public APIs를 제공할 것임 / 예를 들면 Twitter API, NY Times API등 
  - 데이터를 모을 때 고려해야하는 부분이 있음
    - Public API를 사용할때, 그것을 사용하기 전에 API의 한계점을 이해해야함 / 예를들어 몇 API는 queries를 만드는데 한계를 설정을 할 수 있음
    - training examples이 더 많을수록(samples이라고 언급될 것임) 더 잘 작동할 것임 / 이것은 모델이 더 잘 generalize하는데 도움을 줄 것임
    - 모든 class의 샘플의 수를 확실히하거나 topic이 과도하게 imbalanced되지 않게 하여라 / 각각의 class에서 samples의 수를 comparable하게하라
    - sample이 가능한 inputs의 space를 cover하게 확실시하라 / common case뿐만 아니라
  - 이 과정에서는 Internet Movie Database(IMDb) movie reviews dataset을 workflow로 설정함 
  - 이 dataset은 IMDb 웹사이트에 사람들이 올린 movie review뿐만아니라 reviewer들이 영화를 좋아하는지 싫어하는지를 상응하는 positive나 negative한 label을 포함함  
  - 이것은 sentiment analysis problem에서의 classic한 예제임

### Step 2:Explore Your Data
  - model을 만들고 학습하는 것은 workflow의 일부분의 불가함
  - data의 특성을 이해하는 것은 모델을 더 잘 만들게 할 수 있음
  - 이것은 단순히 높은 정확성을 포함하는 것을 의미할 수 있음
  - 이것은 또한 training을 위한 더 적은 데이터를 필요로 하는 것을 의미하고 혹은 더 적은 computational resources를 필요로 하는 것을 의미함
  - Load the Dataset / First up, let's load the dataset into Python
  ```python
     def load_imdb_sentiment_analysis_dataset(data_path, seed=123):
         """Loads the IMDb movie reviews sentiment analysis dataset.

         # Arguments
             data_path: string, path to the data directory.
             seed: int, seed for randomizer.

         # Returns
             A tuple of training and validation data.
             Number of training samples: 25000
             Number of test samples: 25000
             Number of categories: 2 (0 - negative, 1 - positive)

        # References
            Mass et al., http://www.aclweb.org/anthology/P11-1015

            Download and uncompress archive from:
            http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
        """
        imdb_data_path = os.path.join(data_path, 'aclImdb')

        # Load the training data
        train_texts = []
        train_labels = []
        for category in ['pos', 'neg']:
            train_path = os.path.join(imdb_data_path, 'train', category)
            for fname in sorted(os.listdir(train_path)):
                if fname.endswith('.txt'):
                    with open(os.path.join(train_path, fname)) as f:
                        train_texts.append(f.read())
                    train_labels.append(0 if category == 'neg' else 1)

        # Load the validation data.
        test_texts = []
        test_labels = []
        for category in ['pos', 'neg']:
            test_path = os.path.join(imdb_data_path, 'test', category)
            for fname in sorted(os.listdir(test_path)):
                if fname.endswith('.txt'):
                    with open(os.path.join(test_path, fname)) as f:
                        test_texts.append(f.read())
                    test_labels.append(0 if category == 'neg' else 1)

        # Shuffle the training data and labels.
        random.seed(seed)
        random.shuffle(train_texts)
        random.seed(seed)
        random.shuffle(train_labels)

        return ((train_texts, np.array(train_labels)),
                (test_texts, np.array(test_labels)))
  ```
  - Check the Data
    - 데이터를 loading한 후 몇 번의 check를 하는 것이 좋은 practice임 / 몇 개의 samples를 고르고 manually check를 하여서 예상과 일치하는 부분이 있는지 확인하라  
    - 예를들어 몇 개의 random samples를 print해서 본 후 sentiment label이 sentiment of the review와 대응하는지 확인하라
    - IMDb 데이터세트로부터 무작위로 선택한 리뷰를 보면
      - 2시간중에 좋은 부분은 10분정도 있었고 어떠한 특별함이 없다고 중간쯤 느꼈을 때 나는 그곳을 떠나야만 했다
    - 예상한 sentiment가 sample's label과 일치함
  - Collect Key Metrics
    - data를 한 번 verified할 때, text classification problem을 특성화하는데 도움을 주는 중요한 metrics를 모아라
      - 1.Number of samples:Total number of examples you have in the data
      - 2.Number of classes:Total number of topics or categories in the data
      - 3.Number of samples per class:Number of samples per class(topic/category) / In a balanced dataset, all classes will have a similar number of samples / In an imbalanced dataset, the number of samples in each class will vary widely 
      - 4.Number of words per sample:Median number of words in one sample
      - 5.Frequency distribution of words:Distribution showing the frequency (number of occurrences) of each word in the dataset
      - 6.Distribution of sample length:Distribution showing the number of words per sample in the dataset
    - 어떠한 metrics가 IMDb reviews dataset에서 가치가 있는지 확인해보자
    <img src="https://user-images.githubusercontent.com/32586985/76322202-6ef9e780-6326-11ea-829a-1ad74cc1cdcf.PNG">
    
    - 아래의 예시로 몇 개 metrics를 계산하고 분석함
    ```python
       import numpy as np
       import matplotlib.pyplot as plt

       def get_num_words_per_sample(sample_texts):
           """Returns the median number of words per sample given corpus.

           # Arguments
               sample_texts: list, sample texts.

           # Returns
               int, median number of words per sample.
           """
           num_words = [len(s.split()) for s in sample_texts]
           return np.median(num_words)

       def plot_sample_length_distribution(sample_texts):
           """Plots the sample length distribution.

           # Arguments
               samples_texts: list, sample texts.
           """
           plt.hist([len(s) for s in sample_texts], 50)
           plt.xlabel('Length of a sample')
           plt.ylabel('Number of samples')
           plt.title('Sample length distribution')
           plt.show()
    ```
    <img src="https://user-images.githubusercontent.com/32586985/76322536-df086d80-6326-11ea-9bc6-3e5b1fc779d8.PNG">
    <img src="https://user-images.githubusercontent.com/32586985/76322570-e891d580-6326-11ea-8f3c-9c820dac9ff7.PNG">
    
### Step 2.5:Choose a Model
  - 이 시점에서는 dataset을 모으고 data의 주요한 특성에 대한 insight를 얻었을 것임
  - Step2에서 얻은 metrics를 바탕으로 어떤 classification model을 사용해야하는지 생각해야함
  - 이것은 어떻게 우리가 numeric input으로 예상되는 algorithm으로 text data를 나타내는지(data preprocessing이나 vectorization으로 불리우는)나 어떤 모델의 타입을 써야하는 것과 모델을 사용하는데 어떤 configuration parameters를 사용해야는지 등 생각해봐야함을 의미함    
  - 수십년간의 연구 덕택에, data preprocessing과 model configuration options의 large arry로 접근할 수 있음
  - 하지만 viable options의 very large array의 이용가능성은 특정문제의 scope와 complexity를 동시에 크게 증가시킴
  - 최고의 선택은 명확하지 않을수도 있고, navie한 해결책은 사용가능한 모든 options을 철저히 해야할 수 있고, 직관적으로 몇 개에 선택을 제외시켜야 할 수 있음  
  - 하지만 그것은 너무 expensive함
  - 이 과정에서는 text classification model을 선택하는 과정을 크게 단순화해서 접근할 것임
  - dataset이 주어진다면, 우리의 목표는 학습에 필요한 시간을 계산을 최소화하는동안 최대의 정확성에 근접하게 도달할 수 있는 알고리즘을 찾는것임
  - 서로다른 종류의 문제(특히 sentiment analysis와 topic classification problems)를 넘어서 12개의 dataset을 사용하면서, 각각의 dataset사이에 서로 다른 data preprocessing techniques와 서로 다른 model architectures로 바꾸면서, 많은 실험(~450K)를 실행해야함 
  - 이것은 optimal chocies의 영향을 미치는 dataset parameters를 정의하는데 도움을 줌
  - 아래의 실험의 요약을 통해서 model은 알고리즘과 플로우차트를 선택함
  
  - Algorithm for Data Preparation and Model Building 
  ```python
     1. Calculate the number of samples/number of words per sample ratio.
     2. If this ratio is less than 1500, tokenize the text as n-grams and use a simple multi-layer perceptron (MLP) model to classify them (left branch in the flowchart below):
        a. Split the samples into word n-grams; convert the n-grams into vectors.
        b. Score the importance of the vectors and then select the top 20K using the scores.
        c. Build an MLP model.
     3. If the ratio is greater than 1500, tokenize the text as sequence and use a sepCNN model to classify them (right branch in the flowchart below):
        a. Split the samples into words; select the top 20K words based on their frequency.
        b. Convert the samples into word sequence vectors.
        c. If the original number of samples/number of words per sample ratio is less than 15K, using a fine-tuned pre-trained embedding with the sepCNN model will likely provide the best results.
     4. Measure the model performance with different hyperparameter values to find the best model configuration for the dataset.  
  ```
    - 만약 플로우차트가 아래와 같다면, yellow boxes는 data와 model preparation processes를 가르킴 
    - Grey boxes와 green boxes는 각각의 process를 고려하는 선택을 가르킴 
    - Green boxes는 각각의 process의 recommended choice를 가르킴
    - 이 플로우차트를 첫 experiment에서 starting point를 설계하기 위해서 사용할 수 있음 / low computation costs로 좋은 accuracy를 줄 수 있음 
    - 초기 모델을 지속적인 반복을 통해서 계속해서 향상시킬 수 있음
    <img src="https://user-images.githubusercontent.com/32586985/76325713-f9dce100-632a-11ea-935c-542f43640141.PNG">
    
    - 플로우차트를 보며 핵심 질문은 아래와 같음
      - 1.어떠한 learning algorithm과 모델을 사용해야하는가?
      - 2.어떻게 data를 text와 label사이의 관계를 효과적으로 학습하게끔 준비할 수 있을까?
    - 2번째 질문의 정답은 1번째 질문에 달려있음 / 어떤 모델을 고르는에 따라서 모델이 preprocess data를 얻는 방식이 다를것임
    - 모델은 두 개의 카테고리로 광범위하게 분류할 것임 / word를 ordering information(sequence models)로 사용하거나 아니면 단지 words(n-gram models)의 bags(sets)으로 text를 봄 
    - sequence models의 종류는 convolutional neural networks(CNNs), recurrent neural network(RNNs), 그들의 variations을 포함하고 있음 
    - n-gram 모델의 타입은 logistic regression, simple multi-layer perceptrons(MLPs, or fully-connected neural networks), gradient boosted trees와 support vector machines를 포함하고 있음
    - 실험에 따르면, samples의 수 (S)의 비율과 sample당 words의 수(W)의 비율이 모델이 잘 수행하는 것과 연관되어 있는 것으로 관찰됨
    - 이 비율의 값이 작다면 (<1500), small multi-layer perceptron이 n-grams를 input으로(Option A라고 부르는)더 잘 수행하거나 적어도 sequence models로써 수행하는 것을 가질 것임
    - MLPs는 정의하고 이해하는데 쉬우며, sequence models보다 time을 계산하는것이 덜 함 
    - 이 비율의 값이 크다면 (>=1500), sequence model (Option B)를 사용하라 
    - 이 과정을 따르면, samples/words-per-sample ration에 기반하여 선택한 model type을 위한 relevant subsections(labeled A or B)를 넘길 수 있음 
    - IMDb review dataset의 따르면, samples/words-per sample의 비율이 144정도되고, 이것은 MLP model을 만드는것을 의미함 
    - 만약 위에 플로우차트를 따르게 된다면 문제에 대한 최적화된 결론이 나오지 않을 수 있음을 이해해야함 / 이유는 아래와 같음
      - 목표가 다를 수 있음 / 가장 짧은 가능한 계산 시간을 이용하여 최고의 정확성을 측정하는것임
      - 대체 flow가 더 좋은 결과를 나타낼 수 있음 area under the curve(AUC)로 optimizing할 때 그럴 수 있음
      - 일반적이고 흔한 알고리즘을 선택할 것임 field가 계속해서 발전할 때 / 새로운 cutting-edge 알고리즘과 enhancements가 데이터와 연관될 수 있고 더 나은 수행을 할 수 있음
      - 몇 개의 dataset을 플로우차트를 유효화할때 사용될 때, dataset의 대체 플로우를 사용하는 걸 선호하는 특정 특성화가 있을 수 있음 

### Step3: Prepare Your Data
  - 모델의 데이터를 넣기전에, 모델이 이해할 수 있는 format으로 변환을 해야함 
  - 먼저 우리가 모은 data samples 대해서 특정한 순서일 수도 있음 / 어떠한 정보도 texts와 labels사이의 관계에 영향을 주는 samples의 순서와 연관되는 것을 원하지 않음 / 예를들어 만약 dataset이 class에 의해서 sort되고 그리고 그때 training/validation sets으로 나뉘어진다면 이러한 sets은 전체적인 data의 분포에 대해서 representative되진 않을 것임 
  - 모델을 확실시 하기 위한 간단한 최고의 방법은 데이터가 무엇인가를 하기전에 항상 데이터를 순서대로 섞는것에 의해 모델의 영향을 주지않는 것임 / 만약 데이터가 training과 validation sets으로 이미 나뉘어졌다면 validation data를 training data를 transform하는것과 같은 방식으로 transform을 확실히 해라 / 만약 training과 validation sets으로 이미 나뉘어지지 않았다면 shuffling한 후에 samples를 split할 수 있음 / 일반적으로 samples의 80%를 training으로 사용하고 20%를 validation으로 사용함 
  - 그 다음에는 machine learning algorithms을 input으로써 numbers를 take하라 / 이것은 text를 numerical vectors로 변환할 것이라는 것을 의미함 
  - 이 과정을 위해 2가지 과정이 필요함
    - Tokenization / texts를 texts와 labels사이의 관계를 good generalization할 수 있는 words나 더 작은 sub-texts로 나누어라 / 이것은 dataset의 vocabulary를 결정함 (데이터에 나타나는 unique tokens의 set)
    - Vectorization / 이러한 texts를 characterize하기 위한 good numerical measure를 정의하라 
  - n-gram vectors와 sequence vectors를 두 개의 과정에서 수행시 어떻게 되는지 보자 / feature selection과 normalization techniques를 사용함으로써 vector representations을 어떻게 최적화하는지 보자
  - N-gram vectors(Option A)
    - n-gram models를 통해서 tokenization과 vectorization이 어떻게 되는지 볼 것임 / feature selection과 normalization techniques를 사용함으로써 n-gram representation을 어떻게 최적화하는지 볼 것임 
    - n-gram vector에서는 text는 unique n-grams의 collection으로 나타날 것임 / n의 인접한 token의 그룹(일반적으로 words) / 만약 'The mouse ran up the clock'을 예를들면 word unigrams(n=1)은 'the','mouse','ran','up','clock'임 / word bigrams(n=2)일 때는 'the mouse', 'mouse ran', 'ran up', 'up the', 'the clock'등이 됨 
    - Tokenization
      - 단어를 unigrams+bigrams으로 tokenizing하는것은 계산시간이 덜 걸린다는동안 좋은 정확성을 제공하는것을 확인함 
    - Vectorization
      - text samples를 n-grams으로 나눌때, 이러한 n-grams을 machine learning models이 생산할 수 있는 numerical vectors로 바꿀 수 있음 / 아래의 예시는 2개의 texts로 생성된 bigrams과 unigrams에 할당된 indexes를 보여줌 
      ```python
         Texts: 'The mouse ran up the clock' and 'The mouse ran down'
         Index assigned for every token: {'the': 7, 'mouse': 2, 'ran': 4, 'up': 10,
           'clock': 0, 'the mouse': 9, 'mouse ran': 3, 'ran up': 6, 'up the': 11, 'the
         clock': 8, 'down': 1, 'ran down': 5}
      ```
    - n-grams으로 indexes가 할당될 때 우리는 일반적으로 아래의 옵션중 하나를 사용하여 vectorize함   
      - One-hot encoding
        - 모든 sample text는 presence를 가르키는 vector혹은 text의 token의 absence를 나타냄 
        ```python
           'The mouse ran up the clock' = [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
        ```
      - Count encoding
        - 모든 sample text는 text의 token의 count를 가르키는 vector로써 나타냄 / 'the'라는 unigram과 상응하는 element는 'the'라는 단어가 2번 나왔기 때문에 2로 나타냄을 알고 있어라
        ```python
           'The mouse ran up the clock' = [1, 0, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1]
        ```
      - Tf-idf encoding
         - 두 개의 접근법을 넘어선 문제는 모든 문서에서 비슷한 결과가 발생하는 단어가 penalized되지 않는 것임 (예를들어, 단어가 dataset안의 text sample로써 특별하지 않을 경우)
         - 예를들어, 'a'같은 단어는 모든 texts에서 빈번하게 나타남 / 그래서 다른 의미있는 단어가 유용하지 않는것보다 더 token이 높게 count 됨
         ```python
            'The mouse ran up the clock' = [0.33, 0, 0.23, 0.23, 0.23, 0, 0.33, 0.47, 0.33,
            0.23, 0.33, 0.33] (See Scikit-learn TfidfTransformer)
         ```
    - 많은 vector의 표현방법이 있지만 위의 세가지 경우가 일반적임 
    - tf-idf encoding이 다른 두 개의 encoding보다 더 좋은 성능을 발휘한 것을 보았고 이 방법을 n-gram을 vectorizing할 때 방법으로 쓰는 것으로 추천함 / 하지만 이것은 더 많은 메모리를 차지하고 계산하는데 시간이 더 걸리며, 많은 dataset이 있다는 것을 기억해야함 
    - Feature selection
      - dataset의 모든 texts를 uni+bigram token의 단어로 변환할 때, 우리는 수천개의 token이 생길 수도 있음 / 이 모든 tokens이나 features가 label prediction에 기여하는 것은 아님 / 그래서 몇몇 tokens을 없앨것임 / 예를들어, dataset를 넘어서 극히 드물게 나타나는 것들 / 그리고 feature importance를 측정하고(각각의 token이 얼마나 label predictions의 기여하는가) 그리고 가장 informative한 tokens만을 포함시킴 
      - features와 labels과 상응하는 것을 가지고 importance score feature를 output으로 하는 통계학적인 함수가 몇 개 있음 / f_classif, chi2 2개의 함수를 일반적으로 씀 / 이 실험은 이러한 함수들이 동등하게 잘 수행되는지 확인할 것임
      - 더 중요한 것은 아래의 사진을 보면 20000 features 사이에 정확히 절정을 찍고, 이에 3배에 해당하는 더 많은 features를 추가할 때 기여를 매우 적게 하고 종종 overfitting을 하거나 performance를 감소시킴 
      <img src="https://user-images.githubusercontent.com/32586985/77307520-bf1c7500-6d3c-11ea-9260-a89ac778e65d.PNG">
      
    - Normalization
      - Normalization은 모든 feature samples의 값을 small하고 similar한 값으로 변환함 / 이것은 learning algorithms에서 gradient descent convergence를 간단하게 함 / 우리가 본 것을 통해서 데이터를 생성하는 동안에 normalization은 text classification 문제에서 더 많은 값을 추가하는 것처럼 보이진 않음 / 이 과정은 스킵하는 걸 추천함
      - 아래의 코드를 아래의 과정을 모두 포함하여 추가하라
        - Tokenize text samples into word uni+bigrams,
        - Vectorize using tf-idf encoding
        - Select only the top 20,000 features from the vector of tokens by discarding tokens that appear fewer than 2 times and using f_classif to calculate feature importance
        ```python
           from sklearn.feature_extraction.text import TfidfVectorizer
           from sklearn.feature_selection import SelectKBest
           from sklearn.feature_selection import f_classif

           # Vectorization parameters
           # Range (inclusive) of n-gram sizes for tokenizing text.
           NGRAM_RANGE = (1, 2)

           # Limit on the number of features. We use the top 20K features.
           TOP_K = 20000

           # Whether text should be split into word or character n-grams.
           # One of 'word', 'char'.
           TOKEN_MODE = 'word'

           # Minimum document/corpus frequency below which a token will be discarded.
           MIN_DOCUMENT_FREQUENCY = 2

           def ngram_vectorize(train_texts, train_labels, val_texts):
               """Vectorizes texts as n-gram vectors.

               1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

               # Arguments
                   train_texts: list, training text strings.
                   train_labels: np.ndarray, training labels.
                   val_texts: list, validation text strings.

               # Returns
                   x_train, x_val: vectorized training and validation texts
               """
               # Create keyword arguments to pass to the 'tf-idf' vectorizer.
               kwargs = {
                       'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
                       'dtype': 'int32',
                       'strip_accents': 'unicode',
                       'decode_error': 'replace',
                       'analyzer': TOKEN_MODE,  # Split text into word tokens.
                       'min_df': MIN_DOCUMENT_FREQUENCY,
               }
               vectorizer = TfidfVectorizer(**kwargs)

               # Learn vocabulary from training texts and vectorize training texts.
               x_train = vectorizer.fit_transform(train_texts)

               # Vectorize validation texts.
               x_val = vectorizer.transform(val_texts)

               # Select top 'k' of the vectorized features.
               selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
               selector.fit(x_train, train_labels)
               x_train = selector.transform(x_train).astype('float32')
               x_val = selector.transform(x_val).astype('float32')
               return x_train, x_val
        ```
      - n-gram vector가 나타내는 것과 같이, word order와 grammar에 대한 많은 정보를 제거함 (최상의 상태는 n>1일떄 몇몇 partial ordering information에 대해서 유지하는것임)
      - 이것은 bag-of-words 접근법이라고 함 / 이 representation은 모델이 logistic regression, multi-layer perceptrons, gradient boosting machines, supprort vector machines같이 account를 ordering하는 것을 가지지 않을 때 conjunction하기도 함 
  
  - Sequence Vectors(Option B)
    - 연속되는 문단에서 연속된 모델에서 tokenization과 vectorization을 어떻게 하는지 볼 것임 / 또한 feature selection과 normalization techniques를 사용해서 sequence representation을 어떻게 최적화하는지 볼 것임
    - 몇몇 text samples에서 word order가 text의 의미에서 중요함 / 예를들어 'I used to hate my commute. My new bike changed that completely라는 문장은 순서대로 읽는다면 이해할 수 있다 / CNNs,RNNs같은 모델은 sample에서 단어의 순서로부터 의미를 추론할 수 있음 / 이러한 모델에서 text를 order를 보존하면서 tokens의 순서로써 text를 나타낼 수 있음
    - Tokenization
      - Text는 characters의 순서뿐 아니라 words의 순서로써 나타내짐 / word-level representation을 사용하는것이 character tokens보다 더 좋은 성능을 제공함을 발견함 / 이것은 또한 industry에 따르는 일반적인 norm임 / character tokens을 사용하는겅느 texts가 일반적이지 않게 많은 typos를 가지고 있을때만 확실함
    - Vectorization
      - text samples를 sequences of words로써 변환할 때, 우리는 이 sequences를 numerical vectors로 바꿀 수 있음 / 아래의 예시를 보게 된다면 indexes가 2개의 texts에서 생성된 unigrams으로써 할당됨을 볼 수 있고 그리고 그때 첫번째 text의 token indexes의 sequence가 변환됨을 알 수 있음 / 
      ```python
         Texts: 'The mouse ran up the clock' and 'The mouse ran down'
         Index assigned for every token: {'clock': 5, 'ran': 3, 'up': 4, 'down': 6, 'the': 1, 'mouse': 2}.
         NOTE: 'the' occurs most frequently, so the index value of 1 is assigned to it.
         Some libraries reserve index 0 for unknown tokens, as is the case here.
         Sequence of token indexes: 'The mouse ran up the clock' = [1, 2, 3, 4, 1, 5]
      ```
    - token sequences를 vectorize할 수 있는 두 가지의 옵션이 있음 
      - One-hot encoding:n이 vocabulary의 크기일때 n차원 공간의 word vector를 사용함으로써 sequences를 나타낼 수 있음 / representation은 characters로 tokenizing을 할 때 잘 작동하고, 그래서 vacabulary가 작음 / words를 tokenizing할 때, vocabulary는 수천개의 tokens을 가질 것이고, one-hot vectors를 매우 밀집되고 비효율적으로 만들것임
      ```python
         'The mouse ran up the clock' = [
           [0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0]
         ]
      ```
      - Word embeddings:Words는 그 자체로 의미가 연결되어 있음 / 결과적으로 우리는 밀집된 vector space에 word tokens을 나타낼 수 있고 words 사이에 location과 distance가 얼마나 의미적으로 similar한 지 가르킴 / 이것을 word embeddings이라고 함
      <img src="https://user-images.githubusercontent.com/32586985/77384629-34ce2280-6dc9-11ea-98e2-a3daeec6978b.PNG">
      
      - sequence models은 종종 그들의 첫번째 layer로써 embedding layer를 가지고 있음 / 이 layer은 학습하는 시기동안 word index sequences를 word embedding vectors로써 변환하여 학습을하고 이것은 각각의 word index가 word's location이 semantic space에서 나타나는 real values의 dense vector를 mapped하는 것을 가지는 것임
      <img src="https://user-images.githubusercontent.com/32586985/77384756-8b3b6100-6dc9-11ea-9534-7da47f274f9c.PNG">
      
    - Feature selection
      - data에 있는 모든 단어들이 label predictions의 기여하는 것은 아님 / learning process를 우리의 vocabulary로부터 irrelevant words나 rare을 discarding함으로써 learning process를 최적화할 수 있음 / 실제로 자주 사용하는 2만개의 features를 사용하는것이 효율적인것을 확인함 / 이것은 n-gram models에서도 똑같이 적용됨 
      - sequence vectorization에서의 과정을 보자 / 아래의 코드의 과제는 아래와 같음
        - Tokenizes the texts into words
        - Creates a vocabulary using the top 20,000 tokens
        - Converts the tokens into sequence vectors
        - Pads the sequences to a fixed sequence length
        ```python
           from tensorflow.python.keras.preprocessing import sequence
           from tensorflow.python.keras.preprocessing import text

           # Vectorization parameters
           # Limit on the number of features. We use the top 20K features.
           TOP_K = 20000

           # Limit on the length of text sequences. Sequences longer than this
           # will be truncated.
           MAX_SEQUENCE_LENGTH = 500

           def sequence_vectorize(train_texts, val_texts):
               """Vectorizes texts as sequence vectors.

               1 text = 1 sequence vector with fixed length.

               # Arguments
                   train_texts: list, training text strings.
                   val_texts: list, validation text strings.

               # Returns
                   x_train, x_val, word_index: vectorized training and validation
                       texts and word index dictionary.
               """
               # Create vocabulary with training texts.
               tokenizer = text.Tokenizer(num_words=TOP_K)
               tokenizer.fit_on_texts(train_texts)

               # Vectorize training and validation texts.
               x_train = tokenizer.texts_to_sequences(train_texts)
               x_val = tokenizer.texts_to_sequences(val_texts)

               # Get max sequence length.
               max_length = len(max(x_train, key=len))
               if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

               # Fix sequence length to max value. Sequences shorter than the length are
               # padded in the beginning and sequences longer are truncated
               # at the beginning.
               x_train = sequence.pad_sequences(x_train, maxlen=max_length)
               x_val = sequence.pad_sequences(x_val, maxlen=max_length)
               return x_train, x_val, tokenizer.word_index 
        ```
    - Label vectorization
      - sample text data를 어떻게 numerical vectors로 변환했는지 보았음 / 비슷한 과정이 labels에 반드시 적용되어야함 / 우리는 labels을 [0, num_classes - 1]의 범위로 values를 간단히 변환할 수 있음 / 예를들어 만약 3개의 classes가 있다면 이것을 나타내는것으로 0,1,2의 값을 사용할 수 있음 / network는 이러한 값들을 나타내는 one-hot vecotrs를 사용할 것임(labels 사이에 부정확한 관계를 추론하는것을 피하기 위해서) 이 표현은 neural network에서 사용한 last-layer activation function과 loss function에 기인함
    
