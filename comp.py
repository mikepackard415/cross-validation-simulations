import numpy as np
import cv, bic

def compare_models(n, dgp_ar, m, datasets):
    
    p = 0
    if dgp_ar:
        p = m
    
    vblock_equal    = np.empty((m))
    vblock_unequal  = np.empty((m))
    hvblock_equal   = np.empty((m))
    hvblock_unequal = np.empty((m))
    shwartz_bic     = np.empty((m, 1))

    y = datasets[0]
    xs = datasets[1:]
    
    for i in range(m):
        x = np.concatenate([xs[j].reshape((n-p, -1)) for j in range(i+1)], axis=1)
        cv_ = cv.cv(x, y, n, p)
        
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
    return np.array(picks)
