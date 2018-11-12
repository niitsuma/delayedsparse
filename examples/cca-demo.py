import sys
import scipy
from delayedsparse import CCA


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
      
