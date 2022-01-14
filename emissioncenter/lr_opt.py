import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
 
seed = 7
np.random.seed(seed)
df = pd.read_excel('CDs1.xlsx',0)
X = df.values
np.random.shuffle(X)
X_train = X[0:320,0:5]
y_train = X[0:320,-2]
X_test = X[320:,0:5]
y_test = X[320:,-2]
 
for i in [0.001,0.01,0.02,0.05,0.1,0.2,0.5]:
    model = xgb.XGBRegressor(max_depth = 3, learning_rate = i, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = sum((y_test-y_pred)*(y_test-y_pred))/80
    print("the mse is: {} when the learning_rate is {}".format(np.sqrt(mse),i))