import numpy as np

# Generate data (Static, i.i.d.)
def dgp_static_iid(n, b, m):
    '''
    Function to generate static i.i.d. data.

    Inputs:
        n (int): The sample size.
        b (list): The parameters to the generation function.
        m (int): The number of variables to generate.
    '''
    xs = np.random.uniform(0, 1, (m, n))
    e = np.random.normal(0, 0.5, n)
    
    b1, b2, b3 = b
    y = b1 * xs[0] + \
        b2 * xs[1] + \
        b3 * xs[2] + e
    
    y = y.reshape((1, n))
    datasets = np.concatenate((y, xs), axis = 0)
    
    return datasets

# Generate data (Static, AR errors)
def dgp_static_ar(n,b,m):    
    
    xs = np.random.uniform(0, 1, (m, n))
    
    u=np.random.normal(0,((1-0.95**2)*0.25)**0.5,n-1) 
    e=np.random.normal(0,0.5,1)
    for j in range (n-1):
        e = np.append(e, 0.95 * e[j] + u[j])
    
    b1, b2, b3 = b
    y = b1 * xs[0] + \
        b2 * xs[1] + \
        b3 * xs[2] + e

    y = y.reshape((1, n))
    datasets = np.concatenate((y, xs), axis = 0)
        
    return datasets

# Generate data (AR Models)
def dgp_ar(n,b,m):    
    
    e = np.random.normal(0, 0.5, n+100)
    y = np.random.normal(0, 0.5, 3)
    
    b1, b2, b3 = b
    for j in range(n+100):
        y = np.append(y,(y[j+2] * b1 + 
                         y[j+1] * b2 +
                         y[j]   * b3 + e[j]))
    
    xs = np.empty((m, n-m))
    for i in range(m):
        xs[i] = y[-n-i-1+m:-i-1]
    
    y=y[-n+m:]
    
    y = y.reshape((1, n-m))
    datasets = np.concatenate((y, xs), axis = 0)
        
    return datasets
