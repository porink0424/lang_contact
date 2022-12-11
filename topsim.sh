#!/bin/bash

# USAGE:
# In a conda environment that has python=3.6 and EGG=v1.0,
# run `bash topsim.sh id natt nval cvoc clen`.

partition="big"
id=$1
natt=$2
nval=$3
cvoc=$4
clen=$5

sbatch -p big -o ./log/out_%j-topsim.log \
--wrap "python codes/topsim.py --id=$id --n_attributes=$natt --n_values=$nval --vocab_size=$cvoc --max_len=$clen"
