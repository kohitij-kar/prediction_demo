import numpy as np
from scipy import stats
import random


def get_splithalf_corr(var,ax=1,type='spearman'): 
    """
    specify the variable (var) for which splits are required, 
    along which axis (ax)?
    which correlation method do you want (type)?
    """
    _,_,split_mean1, split_mean2 = get_splithalves(var, ax=ax)
    if(type=='spearman'):
        split_half_correlation = stats.spearmanr(split_mean1,split_mean2) #get the Spearman Correlation
    else:
        split_half_correlation = stats.pearsonr(split_mean1, split_mean2) #get the Pearson Correlation
    return {'split_half_corr':split_half_correlation[0],
            'p-value':split_half_correlation[1],
            'type':type
            }

def get_splithalves(var, ax=1):
    """
    

    Parameters
    ----------
    var : TYPE
        DESCRIPTION.
    ax : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    split_mean1 : TYPE
        DESCRIPTION.
    split_mean2 : TYPE
        DESCRIPTION.

    """
    np.apply_along_axis(np.random.shuffle,ax,var)   # randomly shuffle the array along the specific axis (ax)     
    split1, split2 = np.array_split(var,2, axis=ax) # split the aray into 2 halves
    split_mean1 = np.nanmean(split1,axis=ax) # split half 1
    split_mean2 = np.nanmean(split2,axis=ax) # split half 2
    return split1, split2, split_mean1, split_mean2
    
def spearmanbrown_correction(var): # Spearman Brown Correct the correlation value
    spc_var = (2*var)/(1+var)
    return spc_var


def get_correlation_noise_corrected(var1,var2,nrbs=50,correction_method='spearmanBrown'):
    """
        Parameters
    ----------
    var1 :  variable 1 for correlation (2d array): 2nd dimension has to be trials (repetitions)
    var2 : variable 2 for correlation (2d array): 2nd dimension has to be trials (repetitions)
    nrbs : number of bootstrap repeats. optional, The default is 50.
    correction_method : Split correction applied, optional, The default is 'spearmanBrown'.

    Returns
    -------
    corrected_corr : 1d array of corrected pearson correlation values
        
    """
    corrected_corr = np.empty([nrbs, 1], dtype=float)
    for i in range(nrbs):
        sh_corr_var1 = get_splithalf_corr(var1)
        sh_corr_var2 = get_splithalf_corr(var2)
        den = np.sqrt(sh_corr_var1['split_half_corr']*sh_corr_var2['split_half_corr'])
        if(correction_method=='spearmanBrown'):
            num = stats.pearsonr(np.nanmean(var1,axis=1),np.nanmean(var2,axis=1))
        else:
            var1_split= var1[:,random.sample(list(np.arange(0,np.size(var1, axis=1),1)),int(np.round(np.size(var1, axis=1)/2)))]
            var2_split = var2[:,random.sample(list(np.arange(0,np.size(var2, axis=1),1)),int(np.round(np.size(var2, axis=1)/2)))]
            num = stats.pearsonr(np.nanmean(var1_split,axis=1),np.nanmean(var2_split,axis=1))
        corrected_corr[i] = num[0]/den
    return corrected_corr
        
        
    
def main():
    if __name__ == "__main__":
        main()
     
    
    
  
