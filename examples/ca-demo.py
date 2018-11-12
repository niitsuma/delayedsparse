
import sys
import scipy
from delayedsparse import CA


if __name__ == '__main__':

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
