
import delayedsparse
import scipy.sparse

X=scipy.sparse.rand(1000, 300, density=0.3, format="csr")

pca=delayedsparse.PCA(n_components=3)
pca.fit(X)
Xmaped=pca.transform(X)

