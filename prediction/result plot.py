import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_excel(r'D:\wxy\HQ\ML CDs\CDs1.xlsx', 4, header=None)
X = df.values
max = X.max()
min = X.min()
#x = np.round(((X-min)/(max-min)),decimals=2)
x = np.absolute(np.round(X,decimals=2))

sns.set_theme()
ax = sns.heatmap(x, annot=True, annot_kws={'size':8,'weight':'bold'}, cmap = 'YlGnBu')
plt.rc('font',family='Arial')
plt.show()

