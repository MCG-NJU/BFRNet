import os
import io
import os.path as osp
import argparse
import face_alignment
from facenet_pytorch import MTCNN
import torch
import mmcv
from mmcv import ProgressBar
from PIL import Image
from collections import deque
from petrel_client.client import Client

from utils import *
from transform import *


STD_SIZE = (256, 256)
stablePntsIDs = [33, 36, 39, 42, 45]


def face2head(boxes, scale=1.5):
    new_boxes = []
    for box in boxes:
        width = box[2] - box[0]
        height = box[3] - box[1]
        width_center = (box[2] + box[0]) / 2
        height_center = (box[3] + box[1]) / 2
        square_width = int(max(width, height) * scale)
        new_box = [width_center - square_width / 2, height_center - square_width / 2, width_center + square_width / 2,
                   height_center + square_width / 2]
        new_boxes.append(new_box)
    return new_boxes


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def crop_patch(args, mean_face_landmarks, video, landmarks):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    # frame_idx = 0
    # frame_gen = read_video(video_pathname)
    for frame_idx in range(len(video)):
    # while True:
    #     try:
    #         frame = frame_gen.__next__() ## -- BGR
    #     except StopIteration:
    #         break
        frame = video[frame_idx]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', type=str, required=True)  # 所有文件根目录
    parser.add_argument('--save_root', type=str, required=True)  # 所有文件根目录
    parser.add_argument('--anno_file', type=str, required=True)  # anno file: 5957531/00008 5957531/00008

    parser.add_argument('--detect_every_N_frame', type=int, default=8)
    parser.add_argument('--scalar_face_detection', type=float, default=1.5)

    parser.add_argument('--mean_face', required=True, help='mean face pathname')
    parser.add_argument('--convert_gray', default=False, action='store_true', help='convert2grayscale')

    parser.add_argument('--crop_width', default=96, type=int, help='the width of mouth ROIs')
    parser.add_argument('--crop_height', default=96, type=int, help='the height of mouth ROIs')
    parser.add_argument('--start_idx', default=48, type=int, help='the start of landmark index')
    parser.add_argument('--stop_idx', default=68, type=int, help='the end of landmark index')
    parser.add_argument('--window_margin', default=12, type=int, help='window margin for smoothed_landmarks')

    parser.add_argument('--fail_file', type=str, required=True)
    args = parser.parse_args()
    mean_face_landmarks = np.load(args.mean_face)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    videos = []
    client = Client()
    with io.BytesIO(client.get(args.anno_file)) as p:
        for line in p.readlines():
            two_clips = line.decode("utf-8").strip().split(" ")
            videos.append(osp.join(args.video_root, two_clips[0]))
            videos.append(osp.join(args.video_root, two_clips[1]))

    mtcnn = MTCNN(keep_all=True, device=device)

    fail_video_list = []

    pb = ProgressBar(len(videos))
    pb.start()
    for video_input in videos:
        landmarks = []
        faces = []
        boxes = []

        video = mmcv.VideoReader(client.generate_presigned_url(video_input + '.mp4'))
        frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        try:
            for i, frame in enumerate(frames):
                # Detect faces
                if i % args.detect_every_N_frame == 0:
                    box, _ = mtcnn.detect(frame)
                    box = box[:1]
                    box = face2head(box, args.scalar_face_detection)
                else:
                    box = [boxes[-1]]
                box = box[0]
                # Crop faces and save landmarks
                face = frame.crop((box[0], box[1], box[2], box[3])).resize((224, 224))
                landmark = fa.get_landmarks(np.array(face))
                faces.append(face)
                landmarks.append(landmark)
                boxes.append(box)

            # interpolate landmark
            multi_sub_landmarks = landmarks
            landmarks = [None] * len(multi_sub_landmarks)
            for frame_idx in range(len(landmarks)):
                landmarks[frame_idx] = multi_sub_landmarks[frame_idx][0]
            preprocessed_landmarks = landmarks_interpolate(landmarks)
            assert preprocessed_landmarks is not None

            # face video
            face_video = list()
            for face in faces:
                face_video.append(cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR))
            face_video = np.array(face_video)

            # crop mouth
            sequence = crop_patch(args, mean_face_landmarks, face_video, preprocessed_landmarks)
            assert sequence is not None

            # -- save
            data = convert_bgr2gray(sequence) if args.convert_gray else sequence[..., ::-1]
            save_path = os.path.join(args.save_root, video_input[len(args.video_root) + 1:]) + ".npz"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(save_path, data=data)

            # dim = data[0].shape
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # video_tracked = cv2.VideoWriter(video_input + '_tracked.mp4', fourcc, 25.0, dim, isColor=False)
            # for i in range(len(data)):
            #     video_tracked.write(data[i])
            # video_tracked.release()
        except Exception as e:
            print(e)
            fail_video_list.append(osp.join(osp.basename(osp.dirname(video_input)), osp.basename(video_input)))

        pb.update()

    with open(args.fail_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fail_video_list))


if __name__ == '__main__':
    main()
