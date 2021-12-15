import numpy as np

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


# Create a zero matrix of (number of data, maximum length of questions, used characters (numbers))
def vectorization(questions, expected, chars, x_maxlen, y_maxlen, ctable):
  x = np.zeros((len(questions), x_maxlen, len(chars)))
  y = np.zeros((len(questions), y_maxlen, len(chars)))

  # Insert one-hot vector into the created zero matrix
  for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, x_maxlen)
  for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, y_maxlen)

  return x, y


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


# Inference on new input
def model_inference(test, model, ctable):
  test += ' '*(20-len(test))
  t = np.zeros((1,maxlen,len(chars)))
  for i, sentence in enumerate(test):
    t[0][i]=ctable.encode(sentence,maxlen)[0]

  prediction = ctable.decode(np.argmax(model.predict(t.reshape(1,20,39)),axis=-1)[0],calc_argmax=False)
  
  return prediction


# Visualize fitting process
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