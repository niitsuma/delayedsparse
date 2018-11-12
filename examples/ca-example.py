
import delayedsparse
import scipy.sparse

X=scipy.sparse.rand(400, 400, density=0.3, format="csr")

ca=delayedsparse.CA(n_components=3)
ca.fit(X)
print(ca.F*1)

