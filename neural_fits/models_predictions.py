#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:30:00 2020

@author: kohitij
"""

import numpy as np
from scipy import stats
from regression_metrics import pls_regress, get_train_test_indices
from correlation_metrics import get_splithalves, spearmanbrown_correction, get_splithalf_corr

def get_modelpredictions(rates,model_features,ncomp=10,nrfolds=10,seed=0):
    
    nrImages = rates.shape[0]
    ypred = np.arange(nrImages, dtype=float)
    ypred[:]=np.NAN
    
    for i in range(nrfolds):
        # print('fold number is: ' + str(i))
        train, test = get_train_test_indices(nrImages,nrfolds=nrfolds, foldnumber=i, seed=seed)
        pred = pls_regress(model_features[train,:], np.nanmean(rates[train,:],axis=1), model_features[test,:],ncomp=ncomp)
        np.put(ypred, test, pred)
     
    return ypred


def get_model_neural_splithalfcorr(rates,model_features,ncomp=10,nrfolds=10,seed=0):
    sp1, sp2, _, _ = get_splithalves(rates,ax=1)
    shc = get_splithalf_corr(rates,ax=1)
     # model  predictions split half 1 -- 
    p1 = get_modelpredictions(sp1,model_features, nrfolds=nrfolds, ncomp = ncomp, seed=seed)
     # model  predictions split half 1 -- 
    p2 = get_modelpredictions(sp2,model_features, nrfolds=nrfolds, ncomp = ncomp, seed=seed)
    model_shc = spearmanbrown_correction(stats.pearsonr(p1.T,p2.T)[0])
    neural_shc = spearmanbrown_correction(shc['split_half_corr'])
    return model_shc, neural_shc
    
def predictivity(x,y,rho_xx, rho_yy):
    """
    

    Parameters
    ----------
    x : float np array ,
        e.g. measured firing rates  [images x trials]
    y : float np array 
        ,e.g. model predictions for [images x 1]
    rho_xx : float64 scalar
        internal reliablity of x
    rho_yy : float64 scalar
        internal reliablity of y

    Returns
    -------
    ev : float64
        % EV
    raw_corr : float64
        % raw Pearson correlated
    corrected_raw_corr : float64
        % noise corrected Pearson Correlation
    """
    numerator = stats.pearsonr(x, y)[0]
    denominator = np.sqrt(np.multiply(rho_xx, rho_yy))
    raw_corr = numerator
    corrected_raw_corr = numerator/denominator
    ev = ((corrected_raw_corr)**2)*100
    return ev, raw_corr, corrected_raw_corr
     
  
    
    
    
    
