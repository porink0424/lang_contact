#!/bin/bash
#SBATCH -p big
# NOTE: egg36なるconda環境で、`sbatch train.sh`で実行

# 画像ファイル名はi_att, i_val, c_voc, c_len

comment="テストコメント" # 結果ファイルにコメントとして残される
natt=4
nval=4
cvoc=5
clen=4
epoch=2
early_stopping_thr=0.90


# training
id=$(date +%Y%m%d%H%M%S)
python codes/train.py \
--id=$id \
--comment=$comment \
--n_attributes=$natt --n_values=$nval --vocab_size=$cvoc --max_len=$clen \
--batch_size=5120 --data_scaler=60 --n_epochs=$epoch \
--sender_hidden=500 --receiver_hidden=500 --sender_entropy_coeff=0.5 \
--sender_cell=gru --receiver_cell=gru --random_seed=1 \
--lr=0.001 --receiver_emb=30 --sender_emb=5 --early_stopping_thr=$early_stopping_thr \
> result/$id--$natt-$nval-$cvoc-$clen.txt

# organize data
python codes/visualize.py \
result/$id--$natt-$nval-$cvoc-$clen.txt
