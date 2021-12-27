from mse import *

#Bayesian Information Criterion
def IC(x,y,k):
    sq=mse(x,y,x,y)
    s=sq[0]
    t=sq[1]
    bic =t*np.log(s/t)+(k+2)*np.log(t)
    return [bic]
