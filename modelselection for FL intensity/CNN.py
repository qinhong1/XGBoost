import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution1D,Flatten,MaxPooling1D,normalization
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from asses import *

seed = 7
np.random.seed(seed)
df = pd.read_excel('D:\wxy\HQ\ML CDs\CD.xlsx',0)
df1 = pd.read_excel('D:\wxy\HQ\ML CDs\CD.xlsx',1)
X1 = df1.values
X = df.values
X1 = X1.reshape(-1,5,1)
np.random.shuffle(X)
ss = StandardScaler()
X_dataset = X[:,0:5]
Y_dataset = X[:,-1].reshape(-1,1)
#Y_dataset = ss.fit_transform(Y_dataset.reshape(-1,1))
X_train = X_dataset[0:480,:].reshape((-1,5,1))
y_train = Y_dataset[0:480,:]
X_test = X_dataset[480:,:].reshape((-1,5,1))
y_test = Y_dataset[480:,:]


model = Sequential()
model.add(Convolution1D(32, 2, activation='relu', input_shape=(5,1)))
#model.add(Dropout(0.1))
model.add(Convolution1D(32, 2, activation='relu'))
#model.add(Dropout(0.1))
#model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
#model.add(Dense(16, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=1))

#sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9,nesterov=True)
adam = optimizers.Adam(lr=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
model.compile(loss = 'mse',optimizer = adam)
print('Training -----------')
for step in range(4000):
    cost = model.train_on_batch(X_train, y_train)
    if step % 50 == 0:
        print("After %d trainings, the cost: %f" % (step, cost))

print('\nTesting ------------')
cost = model.evaluate(X_test, y_test, batch_size=10)
print('test cost:', cost)

y_pred = model.predict(X_test)
y_pred = y_pred.flatten()

print(rmse(y_test,y_pred))
print(R2cal(y_test,y_pred))
y_test = y_test.flatten()
print(Rcal(y_test,y_pred))

y_act = model.predict(X1)
df2 = pd.DataFrame(y_act)
df2.to_csv('CDs-2.csv')