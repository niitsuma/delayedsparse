import sys

from delayedsparse  import PCA

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
