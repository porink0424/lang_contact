#!/bin/bash
# NOTE: egg36なるconda環境で、`sh batcher.sh`で実行

# 自分で決める変数
partition="big"
comment="batcherテスト" # 結果ファイルにコメントとして残される、NOTE: 空白をコメントに入れるとバグる
natt=2
nval=4
cvoc=4
clen=4
epoch=1
early_stopping_thr=1.0001
sender_entropy_coeff=0.5

# 基本的に変えない変数
batch_size=5120
data_scaler=60
sender_hidden=500
receiver_hidden=500
sender_cell=gru
receiver_cell=gru
random_seed=1
lr=0.001
receiver_emb=30
sender_emb=5

# random seedを変えながら連続でバッチする
start=1
end=2
for i in $(seq $start $end); do
    id=$(date +%Y%m%d%H%M%S)
    sbatch -p $partition -o ./log/out_%j.log job.sh \
    $id $comment $natt $nval $cvoc $clen \
    $batch_size $data_scaler $epoch \
    $sender_hidden $receiver_hidden \
    $sender_entropy_coeff $random_seed \
    $sender_cell $receiver_cell \
    $lr $receiver_emb $sender_emb \
    $early_stopping_thr
    sleep 1
done

# # 1jobだけバッチする
# id=$(date +%Y%m%d%H%M%S)
# sbatch -p $partition job.sh \
# $id $comment $natt $nval $cvoc $clen \
# $batch_size $data_scaler $epoch \
# $sender_hidden $receiver_hidden \
# $sender_entropy_coeff $random_seed \
# $sender_cell $receiver_cell \
# $lr $receiver_emb $sender_emb \
# $early_stopping_thr
