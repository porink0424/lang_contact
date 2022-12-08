#!/bin/bash
export PATH=/home/app/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/app/cuda/lib64:$LD_LIBRARY_PATH

# `bash prganize_data.sh $natt $nval $cvoc $clen 実験id 実験id ...`で実行（実験idは何個でも指定できる）
# organize_data.pyのみを走らせる

natt=$1; shift
nval=$1; shift
cvoc=$1; shift
clen=$1; shift

for id in "$@"; do
    sbatch -p p -o ./log/out_%j-organize_data.log --gres=gpu:1 \
    --wrap "python codes/organize_data.py result/$id--$natt-$nval-$cvoc-$clen.txt"
done