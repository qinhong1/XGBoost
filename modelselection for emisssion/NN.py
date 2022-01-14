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
y_train = X[0:320,-2]
X_test = X[320:,0:5]
y_test = X[320:,-2]

model = Sequential()
model.add(Dense(16, input_dim=5,activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(1))

sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
model.compile(loss = 'mse',optimizer=sgd)
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