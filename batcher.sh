#!/bin/bash
# NOTE: egg36なるconda環境で、`sh batcher.sh`で実行

# 自分で決める変数
mode=1 # 1: random_seedをいろいろ変えながら実行、0: 1つだけ実行
start=10 # mode=1のときのみ使用
end=10 # mode=1のときのみ使用
random_seed=1 # mode=0のときのみ使用
partition="p"
comment="test" # 結果ファイルにコメントとして残される、NOTE: 空白をコメントに入れるとバグる
natt=4
nval=4
cvoc=5
clen=4
epoch=5
early_stopping_thr=0.9
sender_entropy_coeff=0.5

# 基本的に変えない変数
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

if [ $mode -eq 1 ]; then
    # random seedを変えながら連続でバッチする
    for random_seed_iter in $(seq $start $end); do
        id=$(date +%Y%m%d%H%M%S)
        sbatch $gres -p $partition -o ./log/out_%j-job.log job.sh \
            $id $comment $natt $nval $cvoc $clen \
            $batch_size $data_scaler $epoch \
            $sender_hidden $receiver_hidden \
            $sender_entropy_coeff $random_seed_iter \
            $sender_cell $receiver_cell \
            $lr $receiver_emb $sender_emb \
            $early_stopping_thr
        sleep 1
    done
else
    # 1つだけバッチする
    id=$(date +%Y%m%d%H%M%S)
    sbatch $gres -p $partition -o ./log/out_%j-job.log job.sh \
        $id $comment $natt $nval $cvoc $clen \
        $batch_size $data_scaler $epoch \
        $sender_hidden $receiver_hidden \
        $sender_entropy_coeff $random_seed \
        $sender_cell $receiver_cell \
        $lr $receiver_emb $sender_emb \
        $early_stopping_thr
fi