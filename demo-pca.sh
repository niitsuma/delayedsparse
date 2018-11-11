#!/bin/bash

echo "PCA"
for SIZE in 1000 4000 ;do
    echo "======================";

    echo "matrix size=" $SIZE;
    python3 gen-mat.py $SIZE  

    echo "------";
    /usr/bin/time -f "%M KB %E s" python3 delayedsparse/pca.py delay

    echo "------";
    /usr/bin/time -f "%M KB %E s" python3 delayedsparse/pca.py sklearn

    echo "------";
    /usr/bin/time -f "%M KB %E s" python3 delayedsparse/pca.py randmomized


done
