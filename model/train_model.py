import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

df = pd.read_csv("./data.txt")
df.head()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df.head()

y = df[['pred_x_tom', 'pred_y_tom', 'pred_photo_tom' ,'pred_x_jerry' ,'pred_y_jerry','pred_photo_jerry']]
x = df[['x_tom', 'y_tom', 'photo_tom' ,'x_jerry' ,'y_jerry','photo_jerry']]
y_scaler = MinMaxScaler()
x_scaler = MinMaxScaler()
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y)
print(x)
print(y)

data_len = len(x)
train_len = (int)(1 * data_len)
test_len = data_len - train_len
X_train = x[:train_len]
Y_train = y[:train_len]
X_test = x[train_len + 1:]
Y_test = y[train_len + 1:]

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(1, input_shape=(1, 6)))
model.add(Dense(6))
model.compile(loss='mean_squared_error', optimizer='adam')
print(X_train.shape)
X_train_final = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
print(X_train_final.shape)
model.fit(X_train_final, Y_train, epochs=150, batch_size=1, verbose=2)

trainPredict = model.predict(X_train_final)
trainPredict = np.reshape(trainPredict, (X_train.shape[0], X_train.shape[1]))
trainPredict = y_scaler.inverse_transform(trainPredict)

np.savetxt("model_results.txt", trainPredict)

from random import random, randint
all_data = None
iter = 0;
while iter < 20:
  r = randint(0, 450)
  first = X_train_final[r]
  print(first)
  first = np.reshape(first, (1, 1, 6))
  predict1 = model.predict(first)
  if all_data is None:
    all_data = np.array(first[0])  
  else:
    all_data = np.vstack((all_data, first[0]))
  i = 0;
  while i < 20:
    predict1 = np.reshape(predict1, (1, 1, 6))
    predict2 = model.predict(predict1)
    all_data = np.vstack((all_data, np.array(predict2[0])))
    predict1 = predict2
    i = i + 1;
  iter = iter + 1

all_data = np.reshape(all_data, (len(all_data), 6))

all_data = y_scaler.inverse_transform(all_data)
print(len(all_data))
np.savetxt("model_predict3.txt", all_data)