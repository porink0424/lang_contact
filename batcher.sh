#!/bin/bash
# NOTE: egg36なるconda環境で、`bash batcher.sh`で実行

# 自分で決める変数
start=3
end=5
partition="big"
comment="test" # 結果ファイルにコメントとして残される、NOTE: 空白をコメントに入れるとバグる
natt=4
nval=4
cvoc=5
clen=4
epoch=6
early_stopping_thr=0.99
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
    ids[$random_seed_iter]=$id
    sleep 1
done

# 全てのidを出力する
echo "List of id:"
echo ${ids[@]}