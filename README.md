## Filter-Recovery Network for Multi-Speaker Audio-Visual Speech Separation
This repository contains the code for [BFRNet](https://openreview.net/pdf?id=fiB2RjmgwQ6).

[Filter-Recovery Network for Multi-Speaker Audio-Visual Speech Separation](https://openreview.net/pdf?id=fiB2RjmgwQ6)\
[Haoyue Cheng](https://scholar.google.com/citations?user=hg0h5YEAAAAJ&hl=en&oi=ao), 
[Zhaoyang Liu](https://scholar.google.com/citations?user=btgwZosAAAAJ&hl=en&oi=ao), 
[Wayne Wu](https://scholar.google.com/citations?user=uWfZKz4AAAAJ&hl=en&oi=ao) 
and [Limin Wang](https://scholar.google.com/citations?user=HEuN8PcAAAAJ&hl=en&oi=ao)

### Dataset Preparation
1. Download the VoxCeleb2 test mixture lists from the following link:
```markdown
https://pan.xunlei.com/s/VNXTbMyuZOijYSvNAFJPmVOvA1?pwd=wxtt#
```
2. Create directory "voxceleb2" in the main directory BFRNet, and move the mixture files to directory "voxceleb2".

```markdown
# Directory structure of the VoxCeleb2 dataset:
#    ├── VoxCeleb2                          
#    │       └── [mp4]               (contain the face tracks)
#    │                └── [train]
#    │                           └── [spk_id]
#    │                                       └── [video_id]
#    │                                                     └── [clip_id]
#    │                                                                  └── .mp4 files
#    │                └── [val]
#    │       └── [mouth]             (contain the audio files and mouth roi files)
#    │                └── [train]
#    │                           └── [spk_id]
#    │                                       └── [video_id]
#    │                                                     └── [clip_id]
#    │                                                                  └── .h5 files, .wav files
#    │                └── [val]
```

```markdown
# Directory structure of the lrs2/lrs3 dataset:
#    ├── lrs2/lrs3                          
#    │       └── [main]               (contain the face tracks, audio files, and mouth roi files)
#    │                 └── [video_id]
#    │                               └── .wav files, .npz files, .mp4 files
```

2. Please contact with chenghaoyue98@gmail.com to download datasets.

### Train the model
1. Train the model with slurm:
```shell
GPUS=[GPUS] GPUS_PER_NODE=[GPUS_PER_NODE] bash train_slurm.sh [PARTITION] [JOB_NAME]
```

2. torch.distributed training:
```shell
NNODES=[NNODES] GPUS_PER_NODE=[GPUS_PER_NODE] bash train_dist.sh [JOB_NAME]
```

### Evaluate the model
1. Download the pre-trained networks from the following link:
```shell
https://drive.google.com/drive/folders/1J0qxFMb7NVbsXQwM4HiOJ1u7MI0pUquO
```
2. Create directory "checkpoints" in the main directory BFRNet, and move the models to directory "checkpoints".
3. Evaluate the models on VoxCeleb2 unseen_2mix test set:
```shell
mix_number=2 test_file="anno/unseen_2mix.txt" bash test.sh inference_unseen_2mix
```