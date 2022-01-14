import xgboost as xgb
from xgboost import plot_importance, plot_tree
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from asses import *
from graphviz import Digraph
 
seed = 7
np.random.seed(seed)
df = pd.read_excel('CDs1.xlsx',0)
X = df.values
#X = X[0:200,:]
np.random.shuffle(X)
t = int(400*0.8)
X_train = X[0:t,0:5]
y_train = X[0:t,-2]
X_test = X[t:,0:5]
y_test = X[t:,-2]

model = xgb.XGBRegressor(max_depth = 3, learning_rate = 0.3, n_estimators=100,importance_type = 'weight')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(rmse(y_test,y_pred))
print(R2cal(y_test,y_pred))
print(Rcal(y_test,y_pred))
#np.savetxt('y_pred.txt',y_pred)
#np.savetxt('y_test.txt',y_test)
#print(X_train.shape)

#xgb.plot_tree(model, num_trees=6)
#fig = plt.gcf()
#fig.set_size_inches(120,120)
#fig.savefig('tree.png')
#plot_importance(model)
#plt.show()
print(model.feature_importances_)
plt.bar(range(len(model.feature_importances_)),model.feature_importances_)
plt.show()