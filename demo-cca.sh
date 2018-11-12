#!/bin/bash

echo "CCA"
for SIZE in 300 ;do
    echo "======================";

    echo "size=" $SIZE
    python3 gen-mat.py 2000 $SIZE
    mv tmp.npz tmp2.npz
    python3 gen-mat.py 2000 $SIZE 

    echo "------";
    /usr/bin/time -f "%M KB %E s" python3 delayedsparse/cca.py delay 

    echo "------";
    /usr/bin/time -f "%M KB %E s" python3 delayedsparse/cca.py sklearn



done
