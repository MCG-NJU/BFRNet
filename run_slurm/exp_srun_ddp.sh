#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 30000))
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
set -x

PARTITION=$1
JOB_NAME=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
port=${MASTER_PORT:-"29789"}
resume=${resume:-"true"}

sampler_type=${sampler_type:-"normal"}
curriculum_sample=${curriculum_sample:-1 2 3}

mouthroi_format=${mouthroi_format:-"h5"}
mp4_root=${mp4_root:-"s3://chy/voxceleb2/mp4"}
audio_root=${audio_root:-"s3://chy/voxceleb2/mouth_roi_hdf5"}
mouth_root=${mouth_root:-"s3://chy/voxceleb2/mouth_roi_hdf5"}
train_file=${train_file:-"s3://chy/voxceleb2/formal/train_clips.txt"}
val_file=${val_file:-"s3://chy/voxceleb2/formal/val_clips.txt"}
checkpoints_dir=${checkpoints_dir:-"/mnt/lustre/chenghaoyue/projects/BFRNet/checkpoints"}
batchSize=${batchSize:-8}
nThreads=${nThreads:-4}
seed=${seed:-0}

display_freq=${display_freq:-10}
save_latest_freq=${save_latest_freq:-50}
validation_on=${validation_on:-"true"}
validation_freq=${validation_freq:-100}

weights_facenet=${weights_facenet:-"a"}
weights_unet=${weights_unet:-"a"}
weights_FRNet=${weights_FRNet:-"a"}
weights_lipnet=${weights_lipnet:-"a"}

sisnr_loss_weight=${sisnr_loss_weight:-1}
lamda=${lamda:-0.5}

FRNet_layers=${FRNet_layers:-2}

epochs=${epochs:-19}
lr_steps=${lr_steps:-12 15}
lr_lipnet=${lr_lipnet:-1e-4}
lr_facenet=${lr_facenet:-1e-4}
lr_unet=${lr_unet:-1e-4}
lr_FRNet=${lr_FRNet:-1e-4}
decay_factor=${decay_factor:-0.1}

unet_input_nc=${unet_input_nc:-2}
unet_output_nc=${unet_output_nc:-2}

reliable_face=${reliable_face:-"true"}
face_feature_dim=${face_feature_dim:-128}
visual_feature_type=${visual_feature_type:-"both"}
mask_clip_threshold=${mask_clip_threshold:-5}
compression_type=${compression_type:-"none"}
optimizer=${optimizer:-"adam"}
normalization=${normalization:-"true"}
audio_normalization=${audio_normalization:-"true"}
num_frames=${num_frames:-64}
audio_length=${audio_length:-2.55}
hop_size=${hop_size:-160}
window_size=${window_size:-400}
n_fft=${n_fft:-512}
tensorboard=${tensorboard:-"true"}


cd ../

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=spot \
    ${SRUN_ARGS} \
    python -u train.py \
    --port ${port} \
    --name ${JOB_NAME} \
    --resume ${resume} \
    --mp4_root ${mp4_root} \
    --audio_root ${audio_root} \
    --mouth_root ${mouth_root} \
    --mouthroi_format ${mouthroi_format} \
    --sampler_type ${sampler_type} \
    --curriculum_sample ${curriculum_sample} \
    --train_file ${train_file} \
    --val_file ${val_file} \
    --checkpoints_dir ${checkpoints_dir} \
    --batchSize ${batchSize} \
    --nThreads ${nThreads} \
    --seed ${seed} \
    --display_freq ${display_freq} \
    --save_latest_freq ${save_latest_freq} \
    --validation_on ${validation_on} \
    --validation_freq ${validation_freq} \
    --sisnr_loss_weight ${sisnr_loss_weight} \
    --lamda ${lamda} \
    --FRNet_layers ${FRNet_layers} \
    --epochs ${epochs} \
    --lr_steps ${lr_steps} \
    --lr_lipnet ${lr_lipnet} \
    --lr_facenet ${lr_facenet} \
    --lr_unet ${lr_unet} \
    --lr_FRNet ${lr_FRNet} \
    --weights_facenet ${weights_facenet} \
    --weights_unet ${weights_unet} \
    --weights_FRNet ${weights_FRNet} \
    --weights_lipnet ${weights_lipnet} \
    --decay_factor ${decay_factor} \
    --unet_input_nc ${unet_input_nc} \
    --unet_output_nc ${unet_output_nc} \
    --face_feature_dim ${face_feature_dim} \
    --visual_feature_type ${visual_feature_type} \
    --mask_clip_threshold ${mask_clip_threshold} \
    --compression_type ${compression_type} \
    --optimizer ${optimizer} \
    --normalization ${normalization} \
    --audio_normalization ${audio_normalization} \
    --num_frames ${num_frames} \
    --audio_length ${audio_length} \
    --hop_size ${hop_size} \
    --window_size ${window_size} \
    --n_fft ${n_fft} \
    --tensorboard ${tensorboard} \
    --weighted_mask_loss \
    --visual_pool maxpool \
    --audio_pool maxpool \
    --lipnet_config_path configs/lrw_snv1x_tcn2x.json \
    |& tee logs.txt
