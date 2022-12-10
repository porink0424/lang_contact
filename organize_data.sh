#!/bin/bash
export PATH=/home/app/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/app/cuda/lib64:$LD_LIBRARY_PATH

# USAGE:
# In a conda environment that has python=3.6 and EGG=v1.0,
# run `bash organize_data.sh $natt $nval $cvoc $clen id1 id2 ...`. (You can add any number of ids)

partition="p"
natt=$1; shift
nval=$1; shift
cvoc=$1; shift
clen=$1; shift

for id in "$@"; do
    sbatch -p $partition -o ./log/out_%j-organize_data.log --gres=gpu:1 \
    --wrap "python codes/organize_data.py result/$id--$natt-$nval-$cvoc-$clen.txt"
done