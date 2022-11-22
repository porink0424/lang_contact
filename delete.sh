#!/bin/bash
# `sh delete.sh 実験id 実験id ...`で実行（実験idは何個でも指定できる）
# 結果ファイルを全て燃やす

for input; do
    rm -r model/$input
    rm result/$input--*
    rm result_graph/$input--*
    rm result_md/$input--*
done
