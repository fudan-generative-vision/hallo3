#!/bin/bash
if [ $# -ne 3 ]; then
    echo "Usage: $0 <rank> <total_nodes> <master>"
    exit 1
fi

RANK=$1
TOTAL_NODES=$2
ADDR=$3

echo "RUN on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, RANK $RANK, MASTER_ADDR $ADDR"

run_cmd="torchrun --master_port 29311 --nnodes $TOTAL_NODES --node-rank $RANK --master-addr $ADDR --nproc_per_node=8 hallo3/train_video.py --base configs/cogvideox_5b_i2v_s1.yaml configs/sft_s1.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on $(hostname)"
