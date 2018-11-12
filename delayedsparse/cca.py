"""Canonical Correlation Analysis (CCA)"""

# Author: Hirotaka Niitsuma
# @2018 Hirotaka Niirtsuma
#
# You can use this code olny for self evaluation.
# Cannot use this code for commercial and academical use.
# pantent pending
#  https://patentscope2.wipo.int/search/ja/detail.jsf?docId=JP225380312
#  Japan patent office patent number 2017-007741

import sys

import math

import numpy as np

from scipy.sparse import csr_matrix,csc_matrix,dok_matrix,issparse,coo_matrix
import scipy.sparse

from sklearn import base
from sklearn import utils

from .delayedsparse import delayedspmatrix,delayedspmatrix_t,isdelayedspmatrix
from .delayedsparse import safe_sparse_dot as sdot

from . import extmath2

class CCA(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
    def fit(self, X, Y):
        if isinstance(X,np.ndarray):
            X = np.matrix(X, dtype=float)
        if isinstance(Y,np.ndarray):
            Y = np.matrix(Y, dtype=float)

        len_f=float(X.shape[0])
        xm = X.sum(axis=0)/len_f
        self.x_mean_ = np.array(xm)[0, :]
        ym = Y.sum(axis=0)/len_f
        self.y_mean_ = np.array(ym)[0, :]
        S1=delayedspmatrix(
            lambda z: sdot(X,z)-sdot(xm,z),
            lambda z: sdot(z,X)-sdot(z.sum(axis=1),xm),
            X.shape)

        S2=delayedspmatrix(
            lambda z: sdot(Y,z)-sdot(ym,z),
            lambda z: sdot(z,Y)-sdot(z.sum(axis=1),ym),
            Y.shape)
        U1, D1, Vt1 = extmath2.randomized_svd(S1 , X.shape[1])
        U2, D2, Vt2 = extmath2.randomized_svd(S2 , Y.shape[1])
        U, D, Vt = extmath2.randomized_svd(U1.T.dot(U2) , self.n_components)
        self.x_rotations_ = (Vt1.T.dot(scipy.sparse.diags(1.0/D1,offsets=0).dot(U)))
        self.y_rotations_ = (Vt2.T.dot(scipy.sparse.diags(1.0/D2,offsets=0).dot(Vt.T)))
        return self
        
    def transform(self, X, Y=None):
        utils.validation.check_is_fitted(self, 'x_mean_')

        x_scores=   (X - self.x_mean_) .dot(self.x_rotations_)
        if Y is not None:
            y_scores=   (Y - self.y_mean_).dot(self.y_rotations_)
            return x_scores, y_scores
        return x_scores

def _test():

    ### https://stats.idre.ucla.edu/r/dae/canonical-correlation-analysis/
    xcoef_R=np.matrix([[-1.2538,-0.6215,-0.6617],[0.3513,-1.1877,0.8267],[-1.2624, 2.0273, 2.0002]])

    
    import pandas as pd
    df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/mmreg.csv")


    data_npar=df.values 
    X=data_npar[:,0:3]
    Y=data_npar[:,3:9]
    
    n_components=3
    import sklearn.cross_decomposition
    cca_sk = sklearn.cross_decomposition.CCA(n_components=n_components,scale=False,copy=True)
    cca_sk.fit(X, Y)

    cca_my=CCA(n_components=n_components)
    cca_my.fit(X, Y)

    print(np.divide(xcoef_R,cca_sk.x_rotations_))
    print(np.divide(xcoef_R,cca_my.x_rotations_))
    print(max(np.divide(cca_sk.x_rotations_,cca_my.x_rotations_).std(axis=0, ddof=1))< 0.1 )
    
    
if __name__ == '__main__':

    #_test()
        


    import sklearn.cross_decomposition
    
    if len(sys.argv)==2:
        X = scipy.sparse.load_npz('tmp.npz')
        Y = scipy.sparse.load_npz('tmp2.npz')
        dim=min(10,X.shape[1])
        if sys.argv[1] == 'delay':
            print('delayed sparse CCA')
            cca=CCA(dim)
            cca.fit(X,Y)
        elif  sys.argv[1] == 'sklearn':
            print('sklearn')
            cca=  sklearn.cross_decomposition.CCA(n_components=dim,scale=False)
            cca.fit(X.todense(),Y.todense())
      
