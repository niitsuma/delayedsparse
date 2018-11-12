
import delayedsparse
import scipy.sparse

X=scipy.sparse.rand(1000, 100, density=0.3, format="csr")
Y=scipy.sparse.rand(1000, 100, density=0.3, format="csr")

cca=delayedsparse.CCA(n_components=3)
cca.fit(X,Y)
Xmaped,Ymapped=cca.transform(X,Y)

