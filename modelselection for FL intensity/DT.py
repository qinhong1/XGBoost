import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from asses import *
from sklearn.preprocessing import StandardScaler
 
seed = 7
np.random.seed(seed)
df = pd.read_excel('CD.xlsx',0)
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

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(rmse(y_test,y_pred))
print(R2cal(y_test,y_pred))
y_test = y_test.flatten()
print(Rcal(y_test,y_pred))