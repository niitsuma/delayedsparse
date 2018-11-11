

# Author: Hirotaka Niitsuma
# @2018 Hirotaka Niirtsuma
#
# You can use this code olny for self evaluation.
# Cannot use this code for commercial and academical use.
#
# pantent pending
#  https://patentscope2.wipo.int/search/ja/detail.jsf?docId=JP225380312
#  Japan patent office patent number 2017-007741



from scipy.sparse import csc_matrix,issparse

import numpy as np
import scipy



def safe_sparse_dot_org(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly

    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.
    """
    if issparse(a) or issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b) #fast_dot(a, b)

def safe_sparse_dot(a, b, dense_output=False):
    #print a.shape,b.shape#,a,b
    if isdelayedsparse(a):
        return a.dot(b)
    elif isdelayedsparse(b):
        return b.rdot(a)
    elif isinstance(a, np.ndarray) and isinstance(b, np.matrix) and  len(a.shape)==1 and  b.shape[0]==1 :
        return np.matrix(a).T*b
    else:
        return safe_sparse_dot_org(a, b, dense_output)
   

 
def _sdot(a, b):
    return safe_sparse_dot(a, b)

def _default_dot_func(x):
    return x

# def _default_getitem_func(i,j):
#     return 0

def isdense(x):
    return isinstance(x, np.ndarray)
def isscalarlike(x):
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return np.isscalar(x) or (isdense(x) and x.ndim == 0)


class delayedspmatrix(object):
    def __init__(self,
                 dot_closure=_default_dot_func,#lambda x:x,
                 rdot_closure=_default_dot_func,#lambda x:x,
                 shape=None,
                 transposed=False,
                 getitem=None,
                 ):
        self.dot_closure=dot_closure
        self.rdot_closure=rdot_closure
        self.transposed=transposed
        self.shape=shape
        self.getitem=getitem
        if self.getitem is not None:
            d=self.getitem((0,0))
            if type(d) in [np.float32,np.float64,np.float,float]:
                self.dtype=np.float
            
    def dot(self, other):
        if  self.transposed:
            return self.rdot_closure(other.T).T
        else:
            return self.dot_closure(other)

    def rdot(self, other):
        if  self.transposed:
            return self.dot_closure(other.T).T
        else:
            return self.rdot_closure(other)

    def multiply(self, other):
        if self.getitem is None:
            raise Exception("undef getitem in delayedspmatrix")
        else:
            tmp=other.copy()
            if hasattr(self, 'dtype'):
                if self.dtype in [np.float32,np.float64,np.float,float]:
                    if hasattr(tmp, 'dtype'):
                        if tmp.dtype not in [np.float32,np.float64,np.float,float]:
                            tmp=tmp*1.0
            rows,cols = tmp.nonzero()
            if  self.transposed:
                for row,col in zip(rows,cols):
                    tmp[row,col] *=self.getitem((col,row))
                return tmp
            else:
                for row,col in zip(rows,cols):
                    tmp[row,col] *=self.getitem((row,col))
                return tmp
    def force(self):
        return self.dot(scipy.sparse.eye(self.shape[1]))
        
    def todense(self,order=None, out=None):
        #return (self.dot(scipy.sparse.eye(self.shape[1]))).todense(order, out)
        return self.force().todense(order, out)

    def toarray(self,order=None, out=None):
        return (self.todense()).toarray(order, out)

    def sum(self,axis=None):
        if axis is None:
            return (self.dot(np.ones((self.shape[1],1)))).sum()
        elif axis==0 : 
            return self.rdot(np.ones((1,self.shape[1])))
        else:
            return self.dot(np.ones((self.shape[1],1)))
        
        
    def __mul__(self, other):
        return self.dot_closure(other)

    def __rmul__(self, other):
        return self.rdot_closure(other)
    
    
    def __getitem__(self, key):
        if self.getitem is None:
            evec=csc_matrix((1,self.shape[0]))
            evec[0,key]=1
            return self.rdot_closure(evec)
        else:
            return self.getitem(key)
            
    def __getattr__(self, attr):
        if attr == 'T':
            return delayedspmatrix(self.dot_closure,self.rdot_closure,(self.shape[1],self.shape[0]),not(self.transposed),self.getitem)

    def __getstate__(self):
        return self.__dict__

    # def __get__(self):
    #     return _sdot(self, scipy.sparse.diags(np.ones(self.shape[1])))

    
def isdelayedspmatrix(x):
    return isinstance(x, delayedspmatrix)


isdelayedsparse = isdelayedspmatrix

def delayedspmatrix_t(A):
    return delayedspmatrix(lambda x: _sdot(A.T,x),
                           lambda x: _sdot(x,A.T),
                               (A.shape[1],A.shape[0]))




def _test():
    # d1=delayedspmatrix()
    # print  d1.dot(2)
    # print d1 * 3
    # print 4 * d1

    from scipy.sparse import csr_matrix
    B = csr_matrix([[1, 2, 1], [1, 2, 3], [4, 3, 5]])
    b=delayedspmatrix(lambda x:B*x,lambda x:x*B,(3,3))

    A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    print( (B*A).todense()-(b*A).todense())
    print( (A*B).todense()-(b.rdot(A)).todense() )

    print( B.sum(axis=0))
    print( B.T.sum(axis=1))
    print( b.sum(axis=0))
    print( (b.T).sum(axis=1))
    print( B.sum(axis=1))
    print( B.T.sum(axis=0))
    print( b.sum(axis=1))
    print( (b.T).sum(axis=0))

    


    print(B[0])
    print(B.toarray()[0])
    print(b[0])
    print(b.todense())

    #print(b[0].toarray())
    #print(b[0].toarray()[0])

    #print(_sdot(b[0], b[1]))
    # print ((B.T)*A).todense()-((b.T).dot(A)).todense()
    # print (A*(B.T)).todense()-((b.T).rdot(A)).todense()


    
if __name__ == '__main__':
    _test()

    

