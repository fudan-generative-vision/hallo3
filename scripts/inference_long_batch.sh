#! /bin/bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python hallo3/sample_video.py --base ./configs/cogvideox_5b_i2v_s2.yaml ./configs/inference.yaml --seed $RANDOM --input-file $1 --output-dir $2"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
