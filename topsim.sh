#!/bin/bash

# USAGE:
# In a conda environment that has python=3.6 and EGG=v1.0,
# run `bash topsim.sh natt nval cvoc clen id1 id2 ...`.

partition="big"
natt=$1; shift
nval=$1; shift
cvoc=$1; shift
clen=$1; shift

for id in "$@"; do
    sbatch -p big -o ./log/out_%j-topsim.log \
    --wrap "python codes/topsim.py --id=$id --n_attributes=$natt --n_values=$nval --vocab_size=$cvoc --max_len=$clen"
done
