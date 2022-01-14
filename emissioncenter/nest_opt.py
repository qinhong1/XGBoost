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
 
for i in [100,200,300,400,500,600,700,800,900,1000]:
    model = xgb.XGBRegressor(max_depth = 4, learning_rate = 0.1, n_estimators=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = sum((y_test-y_pred)*(y_test-y_pred))/80
    print("the mse is: {} when the n_estimators is {}".format(np.sqrt(mse),i))