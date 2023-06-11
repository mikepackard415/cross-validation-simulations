from mse import *
import sys

# Cross Validation Function
def cv(x, y, n, p):
    '''
    This function takes x data, y data, the sample size (n), and an indicator of whether autoregressive models are being used (p).
    It executes the four cross-validation procedures: v-block equal, v-block unequal, hv-block equal, and hv-block unequal.
    Returns the mean squared error (mse) of each procedure.
    '''

    nc = int(n**0.5)
    nv = (n-nc) * ((n-nc) % 2==1) + (n-nc+1) * ((n-nc)%2==0) # Always odd
    if p >= n-nv:
        sys.exit('ERROR: Number of models is too large.')
    v=(nv-1)/2
    h=int(n/4)
    nvh=nv-2*h
    vh=(nvh-1)/2
    
    sse_v_eq=0
    sse_v_un=0
    sse_hv_eq=0
    sse_hv_un=0
    n_v_eq=0
    n_v_un=0
    n_hv_eq=0
    n_hv_un=0
        
    for i in range(n-p):
        lo=max(0,int(i-v))
        hi=min(n,int(i+v+1))
        lo_h=max(0,int(i-vh))
        hi_h=min(n,int(i+vh+1))
        
        y_test=y[lo:hi]
        x_test=x[lo:hi]
        y_test_h=y[lo_h:hi_h]
        x_test_h=x[lo_h:hi_h]
        
        if lo == 0:
            y_train=y[hi:n]
            x_train=x[hi:n]
        elif hi == n:
            y_train=y[0:lo]
            x_train=x[0:lo]
        else:
            y_train=np.concatenate((y[0:lo],y[hi:n]),axis=0)
            x_train=np.concatenate((x[0:lo],x[hi:n]),axis=0)

        reg   = mse(x_train, y_train, x_test,   y_test)
        reg_h = mse(x_train, y_train, x_test_h, y_test_h)

        sse_v_un  = sse_v_un+reg[0]
        n_v_un    = n_v_un+reg[1]
        sse_hv_un = sse_hv_un+reg_h[0]
        n_hv_un   = n_hv_un+reg_h[1]
        
        if len(y_test)==nv:
            sse_v_eq = sse_v_eq+reg[0]
            n_v_eq   = n_v_eq+reg[1]
        if len(y_test_h)==nvh:
            sse_hv_eq = sse_hv_eq+reg_h[0]
            n_hv_eq   = n_hv_eq+reg_h[1]

    v_eq  = sse_v_eq  / n_v_eq
    v_un  = sse_v_un  / n_v_un
    hv_eq = sse_hv_eq / n_hv_eq
    hv_un = sse_hv_un / n_hv_un
    
    return [v_eq, v_un, hv_eq, hv_un]
