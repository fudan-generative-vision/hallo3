#! /bin/bash
root_dir="$1"
p="$2"
r="$3"
name="$4"
video_dir="${root_dir}/videos"

python hallo3/data_preprocess.py -i "$video_dir" -p "$p" -r "$r"

python hallo3/extract_meta_info.py -r "$root_dir" -n "$name"
