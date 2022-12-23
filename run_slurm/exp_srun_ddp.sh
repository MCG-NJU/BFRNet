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

audio_augmentation=${audio_augmentation:-"false"}
noise_file=${noise_file:-"s3://chy/noise/filelist.txt"}
noise_root=${noise_root:-"s3://chy/noise/audio"}

display_freq=${display_freq:-10}
save_latest_freq=${save_latest_freq:-50}
validation_on=${validation_on:-"true"}
validation_freq=${validation_freq:-200}

weights_facenet=${weights_facenet:-"a"}
weights_unet=${weights_unet:-"a"}
weights_refine=${weights_refine:-"a"}
weights_lipnet=${weights_lipnet:-"a"}
weights_vocal=${weights_vocal:-"a"}

use_mixandseparate_loss=${use_mixandseparate_loss:-"false"}
use_sisnr_loss=${use_sisnr_loss:-"true"}
use_contrast_loss=${use_contrast_loss:-"false"}

mixandseparate_loss_weight=${mixandseparate_loss_weight:-1}
sisnr_loss_weight=${sisnr_loss_weight:-1}
after_refine_ratio=${after_refine_ratio:-0.5}

refine_num_layers=${refine_num_layers:-2}
residual_last=${residual_last:-"true"}
refine_kernel_size=${refine_kernel_size:-1}

mask_loss_type=${mask_loss_type:-"L2"}
contrast_loss_type=${contrast_loss_type:-"TripletLossCosine"}
contrast_loss_weight=${contrast_loss_weight:-1e-2}
contrast_margin=${contrast_margin:-0.5}
contrast_temp=${contrast_temp:-0.2}

epochs=${epochs:-19}
lr_steps=${lr_steps:-12 15}
lr_lipnet=${lr_lipnet:-1e-4}
lr_facenet=${lr_facenet:-1e-4}
lr_unet=${lr_unet:-1e-4}
lr_refine=${lr_refine:-1e-4}
lr_vocal=${lr_vocal:-1e-4}
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
lipreading_extract_feature=${lipreading_extract_feature:-"true"}


cd ../

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=auto \
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
    --audio_augmentation ${audio_augmentation} \
    --noise_file ${noise_file} \
    --noise_root ${noise_root} \
    --checkpoints_dir ${checkpoints_dir} \
    --batchSize ${batchSize} \
    --nThreads ${nThreads} \
    --seed ${seed} \
    --display_freq ${display_freq} \
    --save_latest_freq ${save_latest_freq} \
    --validation_on ${validation_on} \
    --validation_freq ${validation_freq} \
    --use_mixandseparate_loss ${use_mixandseparate_loss} \
    --use_sisnr_loss ${use_sisnr_loss} \
    --use_contrast_loss ${use_contrast_loss} \
    --mask_loss_type ${mask_loss_type} \
    --mixandseparate_loss_weight ${mixandseparate_loss_weight} \
    --sisnr_loss_weight ${sisnr_loss_weight} \
    --after_refine_ratio ${after_refine_ratio} \
    --refine_num_layers ${refine_num_layers} \
    --residual_last ${residual_last} \
    --refine_kernel_size ${refine_kernel_size} \
    --contrast_loss_type ${contrast_loss_type} \
    --contrast_loss_weight ${contrast_loss_weight} \
    --contrast_margin ${contrast_margin} \
    --contrast_temp ${contrast_temp} \
    --epochs ${epochs} \
    --lr_steps ${lr_steps} \
    --lr_lipnet ${lr_lipnet} \
    --lr_facenet ${lr_facenet} \
    --lr_unet ${lr_unet} \
    --lr_refine ${lr_refine} \
    --lr_vocal ${lr_vocal} \
    --weights_facenet ${weights_facenet} \
    --weights_unet ${weights_unet} \
    --weights_refine ${weights_refine} \
    --weights_lipnet ${weights_lipnet} \
    --weights_vocal ${weights_vocal} \
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
    --lipreading_extract_feature ${lipreading_extract_feature} \
    --weighted_mask_loss \
    --visual_pool maxpool \
    --audio_pool maxpool \
    --lipnet_config_path configs/lrw_snv1x_tcn2x.json \
    |& tee logs.txt
