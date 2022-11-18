#!/bin/bash
#SBATCH -p big
# NOTE: egg36なるconda環境で実行する

# 画像ファイル名はi_att, i_val, c_voc, c_len

natt=4
nval=4
cvoc=5
clen=4

python codes/train.py \
--n_attributes=$natt --n_values=$nval --vocab_size=$cvoc --max_len=$clen \
--batch_size=5120 --data_scaler=60 --n_epochs=3000 \
--sender_hidden=500 --receiver_hidden=500 --sender_entropy_coeff=0.5 \
--sender_cell=gru --receiver_cell=gru --random_seed=1 \
--lr=0.001 --receiver_emb=30 --sender_emb=5 \
> result/$natt-$nval-$cvoc-$clen.txt
