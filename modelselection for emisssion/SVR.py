import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from asses import *

seed = 7
np.random.seed(seed)
df = pd.read_excel('D:\wxy\HQ\ML CDs\CD.xlsx',0)
df1 = pd.read_excel('D:\wxy\HQ\ML CDs\CD.xlsx',1)
X1 = df1.values
X = df.values
np.random.shuffle(X)
X_train = X[0:400,0:5]
y_train = X[0:400,-1]
X_test = X[400:,0:5]
y_test = X[400:,-1]

model = SVR(C=1e6, gamma=0.005)
model.fit(X_train, y_train)

y_pred = model.predict(X1)
df3 = pd.DataFrame(y_pred)
df3.to_csv('CDs-SVR.csv')


#print(rmse(y_test,y_pred))
#print(R2cal(y_test,y_pred))
#print(Rcal(y_test,y_pred))

#np.savetxt(r'y_test.txt',y_test)
#np.savetxt(r'y_pred.txt',ans)