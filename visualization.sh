#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 30000))
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
set -x

PARTITION=$1
JOB_NAME=$2
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
ceph=${ceph:-"true"}
test_file=${test_file:-"s3://chy/voxceleb2/unseen_test/unseen_2mix.txt"}
mix_number=${mix_number:-2}
batchSize=${batchSize:-1}
nThreads=${nThreads:-4}
audio_root=${audio_root:-"s3://chy/voxceleb2/mouth_roi_hdf5/"}
mouth_root=${mouth_root:-"s3://chy/voxceleb2/mouth_roi_hdf5/"}
mouthroi_format=${mouthroi_format:-"h5"}
mp4_root=${mp4_root:-"s3://chy/voxceleb2/mp4"}
output_dir_root=${output_dir_root:-"/mnt/petrelfs/chenghaoyue/projects/VisualVoice/output"}
weights_lipnet=${weights_lipnet:-"/mnt/petrelfs/chenghaoyue/projects/VisualVoice/checkpoints/vox_multi_sisnr_refine5_two_layers_r0.5_2gpus_batch8/lipreading_best.pth"}
weights_facenet=${weights_facenet:-"/mnt/petrelfs/chenghaoyue/projects/VisualVoice/checkpoints/vox_multi_sisnr_refine5_two_layers_r0.5_2gpus_batch8/facial_best.pth"}
weights_unet=${weights_unet:-"/mnt/petrelfs/chenghaoyue/projects/VisualVoice/checkpoints/vox_multi_sisnr_refine5_two_layers_r0.5_2gpus_batch8/unet_best.pth"}
weights_FRNet=${weights_FRNet:-"/mnt/petrelfs/chenghaoyue/projects/VisualVoice/checkpoints/vox_multi_sisnr_refine5_two_layers_r0.5_2gpus_batch8/refine_best.pth"}

FRNet_layers=${FRNet_layers:-2}

PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=auto \
    ${SRUN_ARGS} \
    python -u visualization.py \
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
    --output_dir_root ${output_dir_root} \
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
