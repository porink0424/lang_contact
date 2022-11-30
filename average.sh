#!/bin/bash
# NOTE: `bash average.sh id1 id2 id3 ...`の形で実行

srun -p big python codes/average.py --ids $@