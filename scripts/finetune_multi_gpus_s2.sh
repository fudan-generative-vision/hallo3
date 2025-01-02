#! /bin/bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
echo "RUN on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

run_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=8 hallo3/train_video.py --base configs/cogvideox_5b_i2v_s2.yaml sat/configs/sft_s2.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"