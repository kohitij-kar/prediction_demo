
import numpy as np
from sklearn.cross_decomposition import PLSRegression


def pls_regress(X_train,Y_train,X_test,ncomp=20):
    """
    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    Y_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    ncomp : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    Y_test_pred : TYPE
        DESCRIPTION.

    """
    pls2 = PLSRegression(n_components=ncomp)
    pls2.fit(X_train, Y_train)
    PLSRegression()
    Y_test_pred = pls2.predict(X_test)
    return Y_test_pred



def get_train_test_indices(totalIndices, nrfolds=10,foldnumber=0, seed=1):
    """
    

    Parameters
    ----------
    totalIndices : TYPE
        DESCRIPTION.
    nrfolds : TYPE, optional
        DESCRIPTION. The default is 10.
    foldnumber : TYPE, optional
        DESCRIPTION. The default is 0.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    train_indices : TYPE
        DESCRIPTION.
    test_indices : TYPE
        DESCRIPTION.

    """
    
    np.random.seed(seed)
    inds = np.arange(totalIndices)
    np.random.shuffle(inds)
    splits = np.array_split(inds,nrfolds)
    test_indices = inds[np.isin(inds,splits[foldnumber])]
    train_indices = inds[np.logical_not(np.isin(inds, test_indices))]
    return train_indices, test_indices







def main():
    if __name__ == "__main__":
        main()    
    
    
    
    
    
    
    
    
    
    
    
    
