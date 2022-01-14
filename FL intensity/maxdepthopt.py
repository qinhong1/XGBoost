import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
 
seed = 7
np.random.seed(seed)
df = pd.read_excel('CD.xlsx',0)
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
 
for i in [3,4,5,6,7,8,9,10]:
    model = xgb.XGBRegressor(max_depth = i, learning_rate = 0.1, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = sum((y_test.flatten()-y_pred)*(y_test.flatten()-y_pred))/80
    print("the mse is: {} when the max_depth is {}".format(np.sqrt(mse),i))