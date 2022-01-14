import numpy as np
import pandas as pd

df = pd.read_excel('CDs1.xlsx',0)
x = df.values
N = x.shape[0]
df = pd.DataFrame()
for k in range(0,100,25):
    d = []
    for i in range(k,N,100):
        for j in range(0,25):
            d.append(x[i+j,-2])
    df1 = pd.DataFrame(d)
    df = pd.concat([df,df1], axis=1)
df.to_csv('time.csv',index=False, header=None)
