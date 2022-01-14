import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from asses import *

seed = 7
np.random.seed(seed)
df = pd.read_excel('CDs1.xlsx',0)
X = df.values
np.random.shuffle(X)
X_train = X[0:320,0:5]
y_train = X[0:320,-1]
X_test = X[320:,0:5]
y_test = X[320:,-1]

model = Sequential()
model.add(Dense(125, input_dim=5,activation='tanh'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(1))

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.0005,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
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
mse = sum((y_test-y_pred.flatten())*(y_test-y_pred.flatten()))/80
print(rmse(y_test,y_pred))
print(R2cal(y_test,y_pred))
print(Rcal(y_test,y_pred))