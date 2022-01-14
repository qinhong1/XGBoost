import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
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

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(rmse(y_test,y_pred))
print(R2cal(y_test,y_pred))
print(Rcal(y_test,y_pred))