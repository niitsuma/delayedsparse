"""Principal Component Analysis (PCA)"""

# Author: Hirotaka Niitsuma
# @2018 Hirotaka Niirtsuma
#
# You can use this code olny for self evaluation.
# Cannot use this code for commercial and academical use.
#
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

class PCA(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        if isinstance(X,np.ndarray):
            X = np.matrix(X, dtype=float)

        self.shape=X.shape
        
        len_X_f=float(X.shape[0])
        self.ca = X.sum(axis=0)/len_X_f
        self.mean_ = np.array(self.ca)[0, :]

        S=delayedspmatrix(
            lambda z: sdot(X,z)-sdot(self.ca,z),
            
            #lambda z: sdot(z,X)-sdot(sdot(z, np.matrix(np.ones((X.shape[0],1)))),self.ca),
            lambda z: sdot(z,X)-sdot(z.sum(axis=1),self.ca),
            
            X.shape
            )
        
        self.U, self.D, self.Vt = extmath2.randomized_svd(S, self.n_components)
        self.components_=self.Vt
        return self
        

    def transform(self, X):
        utils.validation.check_is_fitted(self, 'Vt')
        ## return (X-self.ca).dot(self.Vt.T)
        return sdot(X,self.Vt.T)-sdot(self.ca,self.Vt.T)

    # @property
    # def eigenvalues_(self):
    #     utils.validation.check_is_fitted(self, 'D')
    #     return np.square(self.D).tolist()


        
def _test():
    from sklearn import datasets
    X, y = datasets.load_iris(return_X_y=True)

    # import prince
    # import pandas as pd
    # X1 = pd.DataFrame(data=X, columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
    # y1 = pd.Series(y).map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

    # pca_p = prince.PCA(n_components=2)
    # pca_p = pca_p.fit(X1)
    # #print(pca.transform(X).head())
    # print(pca_p.eigenvalues_)
    import sklearn.decomposition
    pca_sk = sklearn.decomposition.PCA(n_components=2)
    pca_sk.fit(X)

    #print(X.shape)
    pca=PCA()
    pca.fit(X)
    print(pca.eigenvalues_)
    

def _test2():
    x = np.linspace(0.2,1,100)
    y = 0.8*x + np.random.randn(100)*0.1
    X = np.vstack([x, y]).T
    np.random.shuffle(X)
    #print(X.shape)

    pca=PCA()
    pca.fit(X)

    #sys.exit()
    
    import sklearn.decomposition
    pca_sk = sklearn.decomposition.PCA(n_components=2)
    pca_sk.fit(X)
    #print(pca_sk.get_covariance())
    #print(X[:5])


    pca_sr= sklearn.decomposition.PCA(n_components=2,svd_solver='randomized')
    pca_sr.fit(X)
    
    # import prince
    # import pandas as pd
    # #X1 = pd.DataFrame(data=X, columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
    # X1 = pd.DataFrame(data=X)
    # # #y1 = pd.Series(y).map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    # pca_p = prince.PCA(n_components=2)
    # pca_p = pca_p.fit(X1)
    # print(pca_p.transform(X1).head())
    # # # print(pca_p.eigenvalues_)

    

    print(pca_sk.mean_)
    print(pca.mean_)
    #print(pca.eigenvalues_)
    print(pca_sk.transform(X)[:5])
    print(pca.transform(X)[:5])
    print(pca_sk.components_)
    print(pca.Vt)

    
    
if __name__ == '__main__':

    #_test2()

    import sklearn.decomposition
 
    if len(sys.argv)==3:
        
        X = scipy.sparse.load_npz('tmp.npz')
        dim=min(10,X.shape[1])
        if sys.argv[1] == 'delay':
            print('delayed sparse PCA')
            pca=PCA(dim)
            pca.fit(X)
        elif  sys.argv[1] == 'sklearn':
            print('sklearn')
            pca= sklearn.decomposition.PCA(n_components=dim)
            pca.fit(X.todense())
        elif  sys.argv[1] == 'randmomized':
            print('sklearn randmomized')
            pca= sklearn.decomposition.PCA(n_components=dim,svd_solver='randomized')
            pca.fit(X.todense())
    


