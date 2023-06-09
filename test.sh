#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
set -x

JOB_NAME=$1
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ceph=${ceph:-"false"}
test_file=${test_file:-"anno/unseen_2mix.txt"}
mix_number=${mix_number:-2}
batchSize=${batchSize:-1}
nThreads=${nThreads:-4}
audio_root=${audio_root:-"../mouth"}
mouth_root=${mouth_root:-"../mouth"}
mouthroi_format=${mouthroi_format:-"h5"}
mp4_root=${mp4_root:-"../mp4"}
weights_lipnet=${weights_lipnet:-"checkpoints/lipreading_best.pth"}
weights_facenet=${weights_facenet:-"checkpoints/facial_best.pth"}
weights_unet=${weights_unet:-"checkpoints/unet_best.pth"}
weights_FRNet=${weights_FRNet:-"checkpoints/refine_best.pth"}

FRNet_layers=${FRNet_layers:-2}



PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python test.py \
    --name ${JOB_NAME} \
    --ceph ${ceph} \
    --test_file ${test_file} \
    --mix_number ${mix_number} \
    --batchSize ${batchSize} \
    --nThreads ${nThreads} \
    --audio_root ${audio_root} \
    --mouth_root ${mouth_root} \
    --mouthroi_format ${mouthroi_format} \
    --mp4_root ${mp4_root} \
    --num_frames 64 \
    --audio_length 2.55 \
    --hop_size 160 \
    --window_size 400 \
    --n_fft 512 \
    --weights_lipnet ${weights_lipnet} \
    --weights_facenet ${weights_facenet} \
    --weights_unet ${weights_unet} \
    --weights_FRNet ${weights_FRNet} \
    --lipnet_config_path configs/lrw_snv1x_tcn2x.json \
    --unet_output_nc 2 \
    --visual_feature_type both \
    --face_feature_dim 128 \
    --visual_pool maxpool \
    --compression_type none \
    --mask_clip_threshold 5 \
    --number_of_face_frames 1 \
    --FRNet_layers ${FRNet_layers}
