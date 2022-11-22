#!/bin/bash
#SBATCH --gres=gpu:1
export PATH=/home/app/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/app/cuda/lib64:$LD_LIBRARY_PATH

id=$1
comment=$2
natt=$3
nval=$4
cvoc=$5
clen=$6
batch_size=$7
data_scaler=$8
epoch=$9
sender_hidden=${10}
receiver_hidden=${11}
sender_entropy_coeff=${12}
random_seed=${13}
sender_cell=${14}
receiver_cell=${15}
lr=${16}
receiver_emb=${17}
sender_emb=${18}
early_stopping_thr=${19}

# training
python codes/train.py \
--id=$id \
--comment=$comment \
--n_attributes=$natt --n_values=$nval --vocab_size=$cvoc --max_len=$clen \
--batch_size=$batch_size --data_scaler=$data_scaler --n_epochs=$epoch \
--sender_hidden=$sender_hidden --receiver_hidden=$receiver_hidden \
--sender_entropy_coeff=$sender_entropy_coeff --random_seed=$random_seed \
--sender_cell=$sender_cell --receiver_cell=$receiver_cell \
--lr=$lr --receiver_emb=$receiver_emb --sender_emb=$sender_emb \
--early_stopping_thr=$early_stopping_thr \
> result/$id--$natt-$nval-$cvoc-$clen.txt

# organize data
python codes/organize_data.py \
result/$id--$natt-$nval-$cvoc-$clen.txt
