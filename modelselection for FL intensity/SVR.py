import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from asses import *

seed = 7
np.random.seed(seed)
df = pd.read_excel('CDs1.xlsx',0)
X = df.values
np.random.shuffle(X)
ss = StandardScaler()
X_dataset = X[:,0:5]
Y_dataset = X[:,-1]
Y_dataset = ss.fit_transform(Y_dataset.reshape(-1,1))
X_train = X_dataset[0:320,:]
y_train = Y_dataset[0:320,:]
X_test = X_dataset[320:,:]
y_test = Y_dataset[320:,:]

model = SVR(C=1000, gamma=0.01)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(rmse(y_test,y_pred))
print(R2cal(y_test,y_pred))
y_test = y_test.flatten()
print(Rcal(y_test,y_pred))

#np.savetxt(r'y_test.txt',y_test)
#np.savetxt(r'y_pred.txt',ans)