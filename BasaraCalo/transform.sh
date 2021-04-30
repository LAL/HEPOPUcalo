#!/bin/bash
# Bash script for transforming h5 files to arrays
MIXED_DATA=~/data/hep-rousseau/refactor_data/mixed_900

for k in {0..300}
do
  python TransformArraySparseFloat.py $MIXED_DATA/mixed_data_$k.h5 900
done
