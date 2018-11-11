#!/bin/bash

echo "CA"
for SIZE in 500 1000 2000 ;do
    echo "======================";

    echo "matrix size=" $SIZE;
    python3 gen-mat.py $SIZE

    echo "------";
    /usr/bin/time -f "%M KB %E s" python3 delayedsparse/ca.py delay

    echo "------";
    /usr/bin/time -f "%M KB %E s" python3 delayedsparse/ca.py orange
done
