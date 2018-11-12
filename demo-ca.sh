#!/bin/bash

echo "CA"
for SIZE in 500 1000 2000 ;do
    echo "======================";

    echo "matrix size=" $SIZE;
    python3 examples/gen-mat.py $SIZE

    echo "------delayed sparse";
    /usr/bin/time -f "%M KB %E s" python3 examples/ca-demo.py delay

    echo "------orange lib";
    /usr/bin/time -f "%M KB %E s" python3 examples/ca-demo.py orange
done
