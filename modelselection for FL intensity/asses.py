import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def Rcal(a,b):
    A = np.cov(a,b)
    r = A[0,1]/np.sqrt(A[0,0]*A[1,1])
    return r

def R2cal(a,b):
    return r2_score(a, b)
    

def rmse(a,b):
    return np.sqrt(mean_squared_error(a, b))