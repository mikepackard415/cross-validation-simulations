import numpy as np
import cv, bic
from mse import *

def compare_models(n, dgp_ar, m, datasets, holdout=0, delta=0.5):
    '''
    # This function compares sum of squared errors (sse) from competing models. Returns the number of the model selected by each procedure.
    # If testing prediction on a holdout set, this function instead returns the sse for the model selected by each procedure.
    '''
    p = 0
    holdout_buffer = 0
    if dgp_ar:
        p = m
        if holdout == 50:
            holdout_buffer = 20
        elif holdout == 25:
            holdout_buffer = 10
    
    vblock_equal    = np.empty((m))
    vblock_unequal  = np.empty((m))
    hvblock_equal   = np.empty((m))
    hvblock_unequal = np.empty((m))
    shwartz_bic     = np.empty((m, 1))
    
    if holdout:
        remove_holdout = datasets[:, :-(holdout+holdout_buffer)]
    else:
        remove_holdout = datasets
    
    y = datasets[0]
    xs = datasets[1:]
    
    for i in range(m):
        x = np.concatenate([xs[j].reshape((n-holdout-holdout_buffer-p, -1)) for j in range(i+1)], axis=1)
        cv_ = cv.cv(x, y, n-holdout-holdout_buffer, p, delta)
        
        vblock_equal[i]    = cv_[0]
        vblock_unequal[i]  = cv_[1]
        hvblock_equal[i]   = cv_[2]
        hvblock_unequal[i] = cv_[3]
        shwartz_bic[i]     = bic.IC(x, y, i+1)[0]
        
    p1 = np.where(vblock_equal    == np.amin(vblock_equal))[0]    + 1
    p2 = np.where(vblock_unequal  == np.amin(vblock_unequal))[0]  + 1
    p3 = np.where(hvblock_equal   == np.amin(hvblock_equal))[0]   + 1
    p4 = np.where(hvblock_unequal == np.amin(hvblock_unequal))[0] + 1
    p5 = np.where(shwartz_bic     == np.amin(shwartz_bic))[0]     + 1

    picks=[p1[0], p2[0], p3[0], p4[0], p5[0]]
    
    if not holdout:
        return np.array(picks)

    # Get the unique set of models selected
    models = set(picks)
    rmse_results = {}
    
    # For each model that was selected, find the MSE
    y = datasets[0]
    xs = datasets[1:]
    for i in models:
        
        x = np.concatenate([xs[j].reshape((n-p, -1)) for j in range(i)], axis=1)
        
        x_train = x[:-(holdout+holdout_buffer)]
        y_train = y[:-(holdout+holdout_buffer)]
        x_test = x[-holdout:]
        y_test = y[-holdout:]
        
        sse, n_holdout = mse(x_train, y_train, x_test, y_test, dgp_ar)
        assert n_holdout == holdout
        
        rmse_results[i] = (sse / n_holdout) ** 0.5
     
    rmse_of_picks = [rmse_results[pick] for pick in picks]
    return np.array(rmse_of_picks)
    
