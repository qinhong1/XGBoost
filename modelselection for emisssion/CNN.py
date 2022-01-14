import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution1D,Flatten,MaxPooling1D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from asses import *

seed = 7
np.random.seed(seed)
df = pd.read_excel('D:\wxy\HQ\ML CDs\CD.xlsx',0)
df1 = pd.read_excel('D:\wxy\HQ\ML CDs\CD.xlsx',1)
X = df.values
X1 = df1.values
np.random.shuffle(X)
X_train = X[0:480,0:5].reshape((-1,5,1))
y_train = X[0:480,-2]
X_test = X[480:,0:5].reshape((-1,5,1))
y_test = X[480:,-2]
X1 = X1.reshape((-1,5,1))

model = Sequential()
model.add(Convolution1D(32, 2, activation='relu', input_shape=(5,1)))
model.add(Convolution1D(32, 2, activation='relu'))
#model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=1))

adam = optimizers.Adam(lr=0.05,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
model.compile(loss = 'mse',optimizer = adam)
print('Training -----------')
for step in range(2000):
    cost = model.train_on_batch(X_train, y_train)
    if step % 50 == 0:
        print("After %d trainings, the cost: %f" % (step, cost))

print('\nTesting ------------')
cost = model.evaluate(X_test, y_test, batch_size=10)
print('test cost:', cost)

y_pred = model.predict(X_test)
y_pred = y_pred.flatten()
y_act = model.predict(X1)
#mse = sum((y_test-y_pred.flatten())*(y_test-y_pred.flatten()))/80
print(rmse(y_test,y_pred))
print(R2cal(y_test,y_pred))
print(Rcal(y_test,y_pred))
#model.save('model.h5')
df2 = pd.DataFrame(y_act)
df2.to_csv('CDs-3.csv')