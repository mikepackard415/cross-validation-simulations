from mse import *

#Bayesian Information Criterion
def IC(x, y, k):
    s, t = mse(x, y, x, y)
    bic = t * np.log(s/t) + (k+2) * np.log(t)
    return [bic]
