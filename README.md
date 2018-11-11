
## Delayed Sparse Matrix

Efficient sparse matrix implementation for various "Principal Component Analysis".
And demo usages of the efficient implementation for Correspondence Analysis(CA) and principal component analysis (PCA).

To compare with existing methods, you can execute demo.sh.
```sh
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

Author: Hirotaka Niitsuma.

@2018 Hirotaka Niirtsuma.

You can use these codes olny for self evaluation.
Cannot use these codes for commercial and academical use.

-  pantent pending
 - https://patentscope2.wipo.int/search/ja/detail.jsf?docId=JP225380312
 - Japan patent office:patent number 2017-007741



