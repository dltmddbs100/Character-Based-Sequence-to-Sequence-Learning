 **Character Based Sequence to Sequence Learning**
 ---
 
 개별 문자를 통해 문자 사전을 구축하고 해당 사전을 토대로 입력 Sequence에 대해 학습해 Sequence형태의 출력을 실행한다. 
 
 여기서 실행하는 분석은 총 2가지이며 2가지 모두 개별 문자에 기반한 NLP 학습이다. 데이터의 직접적인 생성부터 학습을 통한 모델 구축까지의 과정을 내포한다. 인터넷이 연결되어있지 않은 폐쇄망에서 text에 관련해 어떤 것을 해보면 좋을지 생각하다가 Keras 문서에 있는 코드들을 참조하게 되었으며, 언급했듯이 데이터 수급이 어려운 환경에서도 데이터를 직접 생성해 Sequence모델을 실행해볼 수 있는 좋은 예시로써 실행해보았다. 


## **< 3-digit addition learning >**

### **1\. Data Generation**

 첫번째 코드는 Keras 문서에 있는 Code Example 중 [keras.io/examples/nlp/addition\_rnn/](https://keras.io/examples/nlp/addition_rnn/) 를 참조하였으며 일부 함수의 경우 편의에 맞게 변형하였다. 해당 모델은 숫자 연산 중 세자릿 수의 덧셈을 학습하는 모델로써 Input으로 문자형태를 입력받는데 이는 곧 연산에 대한 수식이된다. 

```python
# Making word based dictionary & Encoding words to one-hot vector
class CharacterTable:
  def __init__(self, chars):
    self.chars = sorted(set(chars))
    self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
    self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

  def encode(self, C, num_rows):
    x = np.zeros((num_rows, len(self.chars)))
    for i, c in enumerate(C):
      x[i, self.char_indices[c]] = 1
    return x

  def decode(self, x, calc_argmax=True):
    if calc_argmax:
      x = x.argmax(axis=-1)
    return "".join(self.indices_char[x] for x in x)
```

 CharacterTable이라는 Class는 개별 문자들에 정수 인코딩을 수행해 각 문자들이 매칭된 개별 숫자를 가지도록 한다. 해당 숫자를 통해 문자열을 One-hot vector로 encoding되는 encode라는 함수를 가진다. 반대로 encoding된 vector를 다시 원본 문자열로 변환하는 decode함수를 정의함으로써 특정 one-hot vector가 어떤 원본 문자열로부터 파생되었는지 확인하는 것에 사용된다. 0부터 9까지의 숫자들이 사용되며 덧셈 연산기호인 '+' 와 공백 역시 문자열로 인식해 고정된 길이를 가지도록 한다.


### **2\. Vectorization**

 학습 및 평가에 사용할 데이터를 생성한다. 총 50,000여개의 연산질문과 그 답에 해당하는 데이터를 random sampling을 통해 생성한다. 해당 과정에서 중복되는 질문은 제거되며 질문과 답 모두 고정된 길이를 가진다. 길이를 맞추기위해 남는 부분은 공백형태의 문자열로 투입되어 일종의 padding과정을 수행한다.

```python
# (데이터수, 질문 최대길이, 사용되는 문자(숫자))의 영행렬을 생성
def vectorization(questions, expected, chars, x_maxlen, y_maxlen, ctable):
  x = np.zeros((len(questions), x_maxlen, len(chars)))
  y = np.zeros((len(questions), y_maxlen, len(chars)))

  # 생성된 영행렬에 one-hot vector을 삽입
  for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, x_maxlen)
  for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, y_maxlen)

  return x, y
```

```python
# Split Train set, Validation set
def train_val_split(x,y):
  split_at = len(x) - len(x) // 10
  (x_train, x_val) = x[:split_at], x[split_at:]
  (y_train, y_val) = y[:split_at], y[split_at:]

  print("Training Data:")
  print(x_train.shape)
  print(y_train.shape,'\n')

  print("Validation Data:")
  print(x_val.shape)
  print(y_val.shape)
  return x_train, x_val, y_train, y_val
```

 구축된 데이터를 바탕으로 One-hot vector로 만들어주는 vectorization 함수이다. 생성된 zero vector에 encoding된 문자열의 index에 따라 1을 넣어 vectorization을 수행한다. 그리고 50,000개 중 45,000개를 Train셋, 5,000개를 Validation셋으로 분할하여 학습 및 평가 데이터에 대한 준비를 마친다. 2차원 One-hot vector가 Input, Output으로 사용된다.


### **3\. Modeling**

```python
# Build LSTM Sequence Model
def model_basic(num_layers):
  model=Sequential()
  model.add(LSTM(128,input_shape=(maxlen, len(chars))))
  model.add(RepeatVector(digits+1)) # target값이 4의 행을 가지므로 변환
  for _ in range(num_layers):
    model.add(LSTM(128, return_sequences=True))
  model.add(Dense(len(chars),activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

  return model
```

 모델링 단계에서는 2번의 LSTM을 사용하였다. 첫번째 LSTM layer는 해당 모델에서 Encoder 역할을 수행하며 for 문에 있는 두번째 layer는 Decoder 역할을 수행한다. 중간에 사용된 RepeatVector는 target값의 shape의 형태로 변환한다. 최상위층에 Dense layer를 통해 softmax연산을 수행해 각 위치의 확률값을 산출한다.

```python
def fitting_visualize(x_train, y_train, x_val, y_val, model, epochs, batch_size, ctable):
  for epoch in range(1,epochs+1):
    print()
    print("Iteration",epoch)
    model.fit(x_train,y_train, batch_size=batch_size, epochs=1, validation_data=(x_val, y_val))

    for i in range(10):
      ind = np.random.randint(0, len(x_val)) # x_val 개수인 5000개 중 랜덤 index값 설정
      rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])] # 해당 index 값의 x_val, y_val을 설정

      # 모델로 예측확률의 최대값을 preds에 할당
      preds = np.argmax(model.predict(rowx), axis=-1) 

      q = ctable.decode(rowx[0]) # decode로 실제 질문을 추출
      correct = ctable.decode(rowy[0]) # decode로 실제 답을 추출
      guess = ctable.decode(preds[0], calc_argmax=False) # decode로 예측결과 답을 추출
    
      print("Q", q, end=" ")
      print("T", correct, end=" ")
      if correct == guess:
        print("v " + guess)
      else:
        print("x " + guess)
```

 첫번째 epoch에서는 Train셋의 학습 정확도와 validation set의 학습 정확도가 모두 떨어지며 loss 역시 크게 나타난다. Decoding된 문제와 답을 보면 역시 제대로 맞추지 못하는 것이 나타난다.
 
 하지만 학습이 진행됨에 따라 loss가 줄어들고 오분류율이 낮아진다. 실제로 10번의 epoch만에 val\_accuracy는 0.9390을 달성하며 실제로 10개의 문제중 9개를 맞히고 있어 학습이 성공적으로 이루어지고 있는 것을 확인할 수있다. 최종적으로 35번째 epoch에 정확도는 약 99%를 달성했으며 모든 질문에대해 올바른 output을 산출한다. 

<br/>

## **< Date_format_conversion_learning >**
 두번째로, 년도 변환 학습이다. 이 역시 개별 문자에 근거한 Sequence 모델로써 위의 분석과 완벽히 동일한 논리를 가진다. 다만 위에서 덧셈에 대한 연산을 학습시켰다면, 여기서는 정제되지 않은 형태의 날짜를 학습해 일정한 format을 가진 형태로 변환해주는 역할을 수행한다. 그러다보니 사용되는 문자가 더 많다. 실제로 Python은 다양한 함수들을 통해 비정제된 날짜데이터를 정형화된 format으로 쉽게 가공할 수 있다. 하지만 날짜데이터가 동일한 유형의 비정제된 형태가 아닌 다양한 유형으로 구성된 형태라면 case를 나누어 각각에 대해 직접적인 정제를 수행해주어야하지만 학습을 통한 딥러닝에서는 이러한 형태들을 종합해 쉽게 일관된 format을 반환할 수 있다.


### **1\. Data Generation**

```python
# Parameters for the model and dataset.
training_size = 50000

# Maximum length of Answer
maxlen = 20

# 사용되는 '월' 문자열 생성
month=['january','february','march','april','may','june','july','august','september','october','november','december']

# 해당하는 '월'을 숫자로 매핑하는 dictionary
month_to_ind=dict((c,i+1) for i,c in enumerate(month))
```

```python
# 문자형식의 날짜와 매칭되는 답 데이터들을 생성
# 총 3가지 유형으로의 날짜를 생성
def data_generation(size):
  questions=[]
  answers=[]

  for i in range(size):
    seed = np.random.randint(0,3)
    
    if seed==0:
      q = np.random.choice(month)+' '+str(np.random.randint(1,32))+'th, '+str(np.random.randint(1900,2022))
      a = q.split()[2]+'-'+str(month_to_ind[q.split()[0]])+'-'+q.split()[1][:-3]

    if seed==1:
      q = str(np.random.randint(1900,2022))+' '+np.random.choice(month)+' '+str(np.random.randint(1,32))+'th'
      a = q.split()[0]+'-'+str(month_to_ind[q.split()[1]])+'-'+q.split()[2][:-2]

    if seed==2:
      q = str(np.random.randint(1,32))+'th '+np.random.choice(month)+' '+str(np.random.randint(1900,2022))
      a = q.split()[2]+'-'+str(month_to_ind[q.split()[1]])+'-'+q.split()[0][:-2]
    
    q += ' '*(20-len(q))
    a += ' '*(10-len(a))

    questions.append(q)
    answers.append(a)

  return questions, answers 
```
 모든 경우의 수에 있어서 Question 문자열이 가지는 최대 길이는 20, Answer 문자열은 10의 길이를 갖는다. 연산학습모델에서와 마찬가지로 최대길이를 설정하고 random seed를 통해 총 3가지 유형으로 날짜를 의미하는 문자열을 생성해 data generation을 수행한다.


### **2\. Vectorization**

```python
# 산출되는 숫자가 3자리 또는 4자리로 ' '이 포함되는 경우가 있어 ' '의 문자열도 고려
chars='0123456789-abcdefghijklmnopqrstuvwxyz, '
ctable=CharacterTable(chars)

x, y = vectorization(questions, answers, chars, maxlen, 10, ctable)
```

 One-hot encoding을 통해 vectorization된 데이터는 1과 0으로 개별문자들의 위치에 따라 행렬을 생성한다.


### **3\. Bidirectional LSTM Model**

 모델의 전반적인 구조는 이전에 사용한 구조와 거의 유사하나, 여기서는 Bidirectional LSTM을 사용하였다. **Bidirectional LSTM은 기존 LSTM에서 이전에 들어온 정보를 예측에 활용한 것에 더해 이후에 얻는 정보들도 활용하는 sequence에 대한 양방향을 모두 이용한 방식**이다.

 Time step 이 1부터 t 까지 있다고 가정할 때 Forward LSTM 에서는 Input을 time step 이 1 일때부터 t 까지 순차적으로 주고 학습한다. 반면 Backward LSTM 에서 Input을 t 일때부터 1까지 역으로 Input 주고 학습하게된다. 이를 통해 각 time step 마다 두 모델에서 나온 2개의 hidden vector는 학습된 가중치를 통해 하나의 hidden vector로 생성된다. 순방향 LSTM과 역방향 LSTM을 concat하여 사용하는 형태다. 

```python
# Build Bidirectional LSTM Sequence Model
def bd_lstm_model(num_layers):
  model=Sequential()
  model.add(Bidirectional(LSTM(128),input_shape=(maxlen, len(chars))))
  model.add(RepeatVector(10)) # target값이 10의 행을 가지므로 변환
  for _ in range(num_layers):
    model.add(LSTM(128, return_sequences=True))
  model.add(Dense(len(chars),activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

  return model

model=bd_lstm_model(1)
```

```python
epochs=5
batch_size=32

fitting_visualize(x_train, y_train, x_val, y_val, model, epochs, batch_size, ctable)
```

 총 5번의 epochs동안 학습을 수행했다. 학습이 진행됨에 따라 빠르게 validation loss는 감소하여 학습이 올바르게 진행된다. 4epoch만에 Train셋과 Validation셋의 Accuracy가 모두 1로 수렴하였다. Bidirectional LSTM은 이와같이 입력되는 데이터의 순서가 뒤바뀌는 것에 크게 영향을 받지않는 데이터에 대해서 높은 수준의 성능을 보여준다. 실제로 연산학습에서 사용한 구조의 모델로 동일한 hyper-parameter로 학습을 진행했을때, val\_acc와 train\_acc가 1이 되는 시점은 5epoch를 초과하였으며 동일한 epoch에서 성능을 비교했을때도 적합이 더딘 모습이 쉽게 확인되는 것을 볼 수 있다.
