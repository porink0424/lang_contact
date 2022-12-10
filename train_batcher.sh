#!/bin/bash
export PATH=/home/app/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/app/cuda/lib64:$LD_LIBRARY_PATH

# USAGE:
# In a conda environment that has python=3.6 and EGG=v1.0,
# run `bash train_batcher.sh`.

# variables to change
start=1
end=3
partition="v"
comment="test" # NOTE: Do not include blank.
natt=2
nval=25
cvoc=100
clen=8
epoch=1000
early_stopping_thr=0.99
sender_entropy_coeff=0.5

# variables to keep unchanged basically
batch_size=5120
data_scaler=60
sender_hidden=500
receiver_hidden=500
sender_cell=gru
receiver_cell=gru
lr=0.001
receiver_emb=30
sender_emb=5

gres=""
if [ $partition = "p" ] || [ $partition = "v" ]; then
    gres="--gres=gpu:1"
fi

# Continuously batch while changing random seeds
for random_seed_iter in $(seq $start $end); do
    id=$(date +%Y%m%d%H%M%S)
    sbatch $gres -p $partition -o ./log/out_%j-job.log --wrap "python codes/train.py --id=$id --comment=$comment --n_attributes=$natt --n_values=$nval --vocab_size=$cvoc --max_len=$clen --batch_size=$batch_size --data_scaler=$data_scaler --n_epochs=$epoch --sender_hidden=$sender_hidden --receiver_hidden=$receiver_hidden --sender_entropy_coeff=$sender_entropy_coeff --random_seed=$random_seed_iter --sender_cell=$sender_cell --receiver_cell=$receiver_cell --lr=$lr --receiver_emb=$receiver_emb --sender_emb=$sender_emb --early_stopping_thr=$early_stopping_thr > result/$id--$natt-$nval-$cvoc-$clen.txt"
    ids[$random_seed_iter]=$id
    sleep 1
done

# Display all id
echo "List of id:"
echo ${ids[@]}