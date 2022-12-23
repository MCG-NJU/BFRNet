#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/LICENSE

# Ack: Code taken from Pingchuan Ma: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks

""" Crop Mouth ROIs from videos for lipreading"""

import os
import os.path as osp
import cv2
import glob
import argparse
import numpy as np
from collections import deque

from utils import *
from transform import *
from mmcv import ProgressBar


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing')
    # -- utils
    parser.add_argument('--video_root', default=None, help='raw video directory')

    parser.add_argument('--tracked_video', default=None, help='raw video directory')
    parser.add_argument('--landmark', default=None, help='landmark directory')
    parser.add_argument('--filename_input', default='./lrw500_detected_face.csv', help='list of detected video and its subject ID')
    parser.add_argument('--mouthroi', default=None, help='the directory of saving mouth ROIs')
    # -- mean face utils
    parser.add_argument('--mean_face', default='./utils/20words_mean_face.npy', help='mean face pathname')
    # -- mouthROIs utils
    parser.add_argument('--crop_width', default=96, type=int, help='the width of mouth ROIs')
    parser.add_argument('--crop_height', default=96, type=int, help='the height of mouth ROIs')
    parser.add_argument('--start_idx', default=48, type=int, help='the start of landmark index')
    parser.add_argument('--stop_idx', default=68, type=int, help='the end of landmark index')
    parser.add_argument('--window_margin', default=12, type=int, help='window margin for smoothed_landmarks')
    # -- convert to gray scale
    parser.add_argument('--convert_gray', default=False, action='store_true', help='convert2grayscale')
    # -- test set only
    parser.add_argument('--testset_only', default=False, action='store_true', help='process testing set only')

    args = parser.parse_args()
    return args


args = load_args()
os.makedirs(args.mouthroi, exist_ok=True)
# -- mean face utils
STD_SIZE = (256, 256)
mean_face_landmarks = np.load(args.mean_face)
stablePntsIDs = [33, 36, 39, 42, 45]


def crop_patch(video_pathname, landmarks):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    frame_idx = 0
    frame_gen = read_video(video_pathname)
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == args.window_margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img( smoothed_landmarks[stablePntsIDs, :],
                                           mean_face_landmarks[stablePntsIDs, :],
                                           cur_frame,
                                           STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append( cut_patch( trans_frame,
                                        trans_landmarks[args.start_idx:args.stop_idx],
                                        args.crop_height//2,
                                        args.crop_width//2,))
        if frame_idx == len(landmarks)-1:
            #deal with corner case with video too short
            if len(landmarks) < args.window_margin:
                smoothed_landmarks = np.mean(q_landmarks, axis=0)
                cur_landmarks = q_landmarks.popleft()
                cur_frame = q_frame.popleft()

                # -- affine transformation
                trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :],
                                            mean_face_landmarks[stablePntsIDs, :],
                                            cur_frame,
                                            STD_SIZE)
                trans_landmarks = trans(cur_landmarks)
                # -- crop mouth patch
                sequence.append(cut_patch( trans_frame,
                                trans_landmarks[args.start_idx:args.stop_idx],
                                args.crop_height//2,
                                args.crop_width//2,))

            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform( trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append( cut_patch( trans_frame,
                                            trans_landmarks[args.start_idx:args.stop_idx],
                                            args.crop_height//2,
                                            args.crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None


def landmarks_interpolate(landmarks):
    
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


videos = os.listdir(args.video_root)
pb = ProgressBar(len(videos))
pb.start()
for video_input in videos:
    if not osp.exists(osp.join(args.tracked_video, video_input)) or \
        not osp.exists(osp.join(args.filename_input, osp.splitext(video_input)[0] + '.csv')) or \
        not osp.exists(osp.join(args.landmark, osp.splitext(video_input)[0] + '.npz')):
        continue

    lines = open(osp.join(args.filename_input, osp.splitext(video_input)[0] + '.csv')).read().splitlines()
    lines = list(filter(lambda x: 'test' in x, lines)) if args.testset_only else lines

    for filename_idx, line in enumerate(lines):

        filename, person_id = line.split(',')
        print('idx: {} \tProcessing.\t{}'.format(filename_idx, filename))

        landmarks_pathname = osp.join(args.landmark, osp.splitext(video_input)[0] + '.npz')
        dst_pathname = osp.join(args.mouthroi, osp.splitext(video_input)[0] + '.npz')

        if os.path.exists(dst_pathname):
            continue

        multi_sub_landmarks = np.load(landmarks_pathname, allow_pickle=True)['data']
        landmarks = [None] * len(multi_sub_landmarks)
        for frame_idx in range(len(landmarks)):
            try:
                #landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)]['facial_landmarks'] #original for LRW
                landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)] #VOXCELEB2
            except (IndexError, TypeError):
                continue

        # -- pre-process landmarks: interpolate frames not being detected.
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        if not preprocessed_landmarks:
            continue

        # -- crop
        sequence = crop_patch(osp.join(args.video_root, video_input), preprocessed_landmarks)
        assert sequence is not None, "cannot crop from {}.".format(filename)

        # -- save
        data = convert_bgr2gray(sequence) if args.convert_gray else sequence[...,::-1]
        save2npz(dst_pathname, data=data)

    pb.update()


print('Done.')
