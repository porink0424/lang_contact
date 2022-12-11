#!/bin/bash
# USAGE:
# In a conda environment that has python=3.6 and EGG=v1.0,
# run `bash other_topsim.sh id1 id2 id3 ...`.

# srun -p big python codes/other_topsim.py --ids $@
python codes/other_topsim.py --ids $@