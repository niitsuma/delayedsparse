#!/bin/bash

echo "CCA"
for SIZE in 300 ;do
    echo "======================";

    echo "size=" $SIZE
    python3 examples/gen-mat.py 2000 $SIZE
    mv tmp.npz tmp2.npz
    python3 examples/gen-mat.py 2000 $SIZE 

    echo "------";
    /usr/bin/time -f "%M KB %E s" python3 examples/cca-demo.py delay 

    echo "------";
    /usr/bin/time -f "%M KB %E s" python3 examples/cca-demo.py sklearn



done
