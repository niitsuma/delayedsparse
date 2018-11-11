"""Correspondence Analysis (CA)"""

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

from delayedsparse import delayedspmatrix,delayedspmatrix_t,isdelayedspmatrix
from delayedsparse import safe_sparse_dot as sdot

import extmath2


## http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def set_correspondenceanalysis_delayed_mat(model):

    
    model.Ut   = delayedspmatrix_t(model.U)   
    model.V   = delayedspmatrix_t(model.Vt)
    model.D_a = scipy.sparse.diags(model.D,offsets=0)
    
    model.F2 = delayedspmatrix(lambda x: sdot(sdot( model.U * model.D_a),x),
                               lambda x: sdot(x,(model.U * model.D_a)),
                            (model.U.shape[0],model.D_a.shape[1])) 

    model.G2 =delayedspmatrix(lambda x: sdot( sdot(model.V, model.D_a) ,x),
                               lambda x: sdot(x,sdot(model.V, model.D_a)),
                               (model.V[0],model.D_a.shape[1])) 

    # principal coordinates of rows
    #F = D_r_rsq * (U * D_a)
    model.F=delayedspmatrix(lambda x: sdot(  model.D_r_rsq*(model.U * model.D_a),x),
                            lambda x: sdot(x,model.D_r_rsq*(model.U * model.D_a)),
                            (model.D_r_rsq.shape[0],model.D_a.shape[1])) 
        
    # principal coordinates of columns
    # G = model.D_c_rsq * (V * D_a)
    model.G=delayedspmatrix(lambda x: sdot(  model.D_c_rsq * sdot(model.V, model.D_a) ,x),
                               lambda x: sdot(x,model.D_c_rsq * sdot(model.V, model.D_a)),
                               (model.D_c_rsq.shape[0],model.D_a.shape[1])) 
    # #model.X = X.A
    # X = model.D_r_rsq.dot(U)
    # model.X = X
    model.X=delayedspmatrix(lambda x: sdot(  model.D_r_rsq*model.U ,x),
                               lambda x: sdot(x,model.D_r_rsq*model.U),
                                (model.D_r_rsq.shape[0],model.U.shape[1]))
        
    # #model.Y = Y.A
    # Y = model.D_c_rsq.dot(V)
    # model.Y = Y
    model.Y=delayedspmatrix(lambda x: sdot(  sdot(model.D_c_rsq,model.V) ,x),
                               lambda x: sdot(x,sdot(model.D_c_rsq,model.V)),
                                (model.D_r_rsq.shape[0],model.Vt.shape[0]))
    

    model.mat_name_list=['F','G','U','V','X','Y','F2','G2']

class CA(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mat_name_list=[]
        
    def fit(self, X, y=None):
        if isinstance(X,np.ndarray):
            N = np.matrix(X, dtype=float)
        else:
            N=X
            
        self.shape=N.shape
            
        n_sum_total=N.sum()
        n_sum_total_f=float(n_sum_total)
        
        ra = np.array(N.sum(axis=1))[:, 0]/n_sum_total_f   ## =pra
        ca = np.array(N.sum(axis=0))[0, :]/n_sum_total_f   ## =pca
        self.ra_inv_sqrt = div0(1.0,np.sqrt(ra))
        self.ca_inv_sqrt = div0(1.0,np.sqrt(ca))

  
        self.D_r_rsq = scipy.sparse.diags(self.ra_inv_sqrt,offsets=0)
        self.D_c_rsq = scipy.sparse.diags(self.ca_inv_sqrt,offsets=0)
   
        r_sq = self.D_r_rsq * ra.reshape((-1,1))
        c_sq = ca.reshape((1,-1)) * self.D_c_rsq

        S=delayedspmatrix(
            lambda x: sdot(self.D_r_rsq, sdot(N,sdot(self.D_c_rsq,x/n_sum_total_f)))-sdot(r_sq,sdot(c_sq,x)),
            
            lambda x: sdot(sdot(sdot(x/n_sum_total_f,self.D_r_rsq),N),self.D_c_rsq)-sdot(sdot(x,r_sq),c_sq),
            N.shape
            )
            
        self.U, self.D, self.Vt = extmath2.randomized_svd(S, self.n_components)
  
        set_correspondenceanalysis_delayed_mat(self)
        
        return self
    
    def transform(self, X):
        utils.validation.check_is_fitted(self, 'D')
        return sdot(X,self.Y)

    
    @property
    def eigenvalues_(self):
        utils.validation.check_is_fitted(self, 'D')
        return np.square(self.D).tolist()
    
    def evaled_mats(self):
        
        self.V=(self.Vt).T
        #self.V=delayedspmatrix_t(self.Vt)
        
        self.D_r_rsq = scipy.sparse.diags(self.ra_inv_sqrt,offsets=0)
        self.D_c_rsq = scipy.sparse.diags(self.ca_inv_sqrt,offsets=0)
        self.D_a = scipy.sparse.diags(self.D,offsets=0)

        
        self.F = self.D_r_rsq * (self.U * self.D_a)
        self.G = self.D_c_rsq * (self.V * self.D_a)
        self.X = self.D_r_rsq.dot(self.U)
        self.Y = self.D_c_rsq.dot(self.V)

        self.F2 = (self.U * self.D_a)
        self.G2 = (self.V * self.D_a)

        self.mat_name_list=['F','G','U','V','X','Y','F2','G2']

        
    def save(self,filename):
        np.savez(filename,
                     U=self.U,
                     D=self.D,
                     Vt=self.Vt,
                     ra_inv_sqrt=self.ra_inv_sqrt, 
                     ca_inv_sqrt=self.ca_inv_sqrt
            )

    def load(self,filename):
        loader = np.load(filename)
        self.D=loader['D']
        self.ra_inv_sqrt=loader['ra_inv_sqrt']
        self.ca_inv_sqrt=loader['ca_inv_sqrt']

        self.D_r_rsq = scipy.sparse.diags(self.ra_inv_sqrt,offsets=0)
        self.D_c_rsq = scipy.sparse.diags(self.ca_inv_sqrt,offsets=0)

        self.U=loader['U']
        self.Vt=loader['Vt']
        
        self.mat_name_list=['F','G','U','V','X','Y','F2','G2']
        set_correspondenceanalysis_delayed_mat(self)


def _test():

    c1 = dok_matrix((4, 4), dtype=np.float32)
    c1[0,0]=20
    c1[1,1]=10
    c1[2,2]=33
    c1[3,3]=2
    c1[0,2]=7
    c1[3,0]=5
    c1[1,0]=4
    
    #print(.todense())
    
    from Orange.widgets.unsupervised.owcorrespondence import correspondence
    c1dense=c1.todense()
    #print(c1dense)
    ca=correspondence(c1dense)
    print(ca.D)
    print(ca.U)
    
    cad=CA(3)

    cad.fit(c1)
    print(cad.D)
    print(cad.U)
    print(cad.Vt)
    print(cad.F)

    print(sdot(cad.F, scipy.sparse.diags(np.ones(3))))

    print(cad.eigenvalues_)
    
if __name__ == '__main__':

    #_test()
    # mat = scipy.sparse.load_npz('tmp.npz')
    # dim=min(10,min(mat.shape))
    # cad=SparseCorrespondenceAnalysis(mat,dim)
    # #print(cad.D)
    if len(sys.argv)==2:
        X = scipy.sparse.load_npz('tmp.npz')
        dim=min(10,min(X.shape))
        if sys.argv[1] == 'delay':
            print('delayed sparse CA')
            ca=CA()
            ca.fit(X)
        elif  sys.argv[1] == 'orange':
            print('existing method(Orange lib)')
            from Orange.widgets.unsupervised.owcorrespondence import correspondence
            X = scipy.sparse.load_npz('tmp.npz')
            ca=correspondence(X.todense())
