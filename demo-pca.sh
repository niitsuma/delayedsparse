#!/bin/bash

pip3 install -U delayedsparse --user 

echo "PCA"
for SIZE in 1000 4000 ;do
    echo "======================";

    echo "matrix size=" $SIZE;
    python3 examples/gen-mat.py $SIZE  

    echo "------delayed sparse";
    
    /usr/bin/time -f "%M KB %E s" python3 examples/pca-demo.py delay

    echo "------ sklearn ";
    /usr/bin/time -f "%M KB %E s" python3 examples/pca-demo.py sklearn

    echo "------ randomized svd";
    /usr/bin/time -f "%M KB %E s" python3 examples/pca-demo.py randmomized


done
