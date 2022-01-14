import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from asses import *
import os
from sklearn.preprocessing import StandardScaler
 
seed = 7
np.random.seed(seed)
df = pd.read_excel('CD.xlsx',0)
#df1 = pd.read_excel('CD.xlsx',1)
#X1 = df1.values
X = df.values
np.random.shuffle(X)
ss = StandardScaler()
X_dataset = X[:,0:5]
Y_dataset = X[:,-1]
Y_dataset = Y_dataset.reshape(-1,1)
#Y_dataset = ss.fit_transform(Y_dataset.reshape(-1,1))
X_train = X_dataset[0:320,:]
y_train = Y_dataset[0:320,:]
X_test = X_dataset[320:,:]
y_test = Y_dataset[320:,:]

 
model = xgb.XGBRegressor(max_depth = 4, learning_rate = 0.1, n_estimators=200)
model.fit(X_train, y_train)
#y_pred = model.predict(X1)
#print(rmse(y_test,y_pred))
#print(R2cal(y_test,y_pred))
#y_test = y_test.flatten()
#print(Rcal(y_test,y_pred))
#y_act = model.predict(X1)
#y_pred = ss.inverse_transform(y_pred)
#y_test = ss.inverse_transform(y_test)
#np.savetxt('y_pred FL.txt',y_pred)
#df3 = pd.DataFrame(y_pred)
#df3.to_csv('CDs-xgb.csv')

for i in range(0,20):
    xgb.plot_tree(model, num_trees=i)
    fig = plt.gcf()
    fig.set_size_inches(120,120)
    filename =str(i)+'.png'
    fig.savefig(filename)
    plt.close()
