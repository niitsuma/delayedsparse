
## Delayed Sparse Matrix

Efficient sparse matrix implementation for various "Principal Component Analysis".
And demo usages of the efficient implementation for 

* Correspondence Analysis(CA) 
* Principal Component Analysis (PCA)
* Canonical Correlation Analysis (CCA)


To compare with existing methods, you can execute demo.sh.
```sh
>>> git clone https://github.com/niitsuma/delayedsparse
>>> cd delayedsparse
>>> bash  demo.sh
```

This library is effective when the input matrix size ls large.
But, in order to demonstrations, the demo programs use only a small matrix.
You can test more large matrix by setting SIZE variable in demo-*.sh


When the input matrix size is large, 
the program of this library will finish within in few minutes, 
but the existing methods take hours.



You can find more general description about CA and PCA in
https://github.com/MaxHalford/prince


## Installation

**Via PyPI**

```sh
>>> pip install delayedsparse
```

**Via GitHub for the latest development version**

```sh
>>> pip install git+https://github.com/niitsuma/delayedsparse 
```

## Usage

### Principal component analysis (PCA)

```python
>>> import delayedsparse
>>> import scipy.sparse
>>> X=scipy.sparse.rand(1000, 300, density=0.3, format="csr")
>>> pca=delayedsparse.PCA(n_components=3)
>>> pca.fit(X)
>>> Xmaped=pca.transform(X)
```

### Correspondence Analysis (CA)

```python
>>> import delayedsparse
>>> import scipy.sparse
>>> X=scipy.sparse.rand(400, 400, density=0.3, format="csr")
>>> ca=delayedsparse.CA(n_components=3)
>>> ca.fit(X)
>>> print(ca.F*1)
```

### Canonical Correlation Analysis (CCA)

```python
>>> import delayedsparse
>>> import scipy.sparse
>>> X=scipy.sparse.rand(1000, 100, density=0.3, format="csr")
>>> Y=scipy.sparse.rand(1000, 100, density=0.3, format="csr")
>>> cca=delayedsparse.CCA(n_components=3)
>>> cca.fit(X,Y)
>>> Xmaped,Ymapped=cca.transform(X,Y)
```

## Requirements

```sh
>>> pip3 install sklearn
```

In order to execute demo.sh, you need install /usr/bin/time and orange library

```sh
>>> apt-get install time
>>> pip3 install orange
```


## License

@2018 Hirotaka Niirtsuma.


You can use these codes olny for self evaluation.
Cannot use these codes for commercial and academical use.

* pantent pending
  * https://patentscope2.wipo.int/search/ja/detail.jsf?docId=JP225380312
  * Japan patent office:patent number 2017-007741



## Author
Hirotaka Niitsuma.


@2018 Hirotaka Niirtsuma.

