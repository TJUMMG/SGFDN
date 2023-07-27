import os
import json
from torch.utils.data import Dataset
from pathlib import Path
from .video_transforms import make_video_transforms, prepare
import time
import numpy as np
import random
import cv2


class VideoModulatedSTGrounding(Dataset):
    def __init__(
        self,
        vid_folder,
        ann_file,
        transforms,
        is_train=False,
        video_max_len=200,
        video_max_len_train=100,
        fps=5,
        tmp_crop=False,
        tmp_loc=True,
        stride=0,
    ):
        """
        :param vid_folder: path to the folder containing a folder "video"
        :param ann_file: path to the json annotation file
        :param transforms: video data transforms to be applied on the videos and boxes
        :param is_train: whether training or not
        :param video_max_len: maximum number of frames to be extracted from a video
        :param video_max_len_train: maximum number of frames to be extracted from a video at training time
        :param fps: number of frames per second
        :param tmp_crop: whether to use temporal cropping preserving the annotated moment
        :param tmp_loc: whether to use temporal localization annotations
        :param stride: temporal stride k
        """
        self.vid_folder = vid_folder
        print("loading annotations into memory...")
        tic = time.time()
        self.annotations = json.load(open(ann_file, "r"))
        print("Done (t={:0.2f}s)".format(time.time() - tic))
        self._transforms = transforms
        self.is_train = is_train
        self.video_max_len = video_max_len
        self.video_max_len_train = video_max_len_train
        self.fps = fps
        self.tmp_crop = tmp_crop
        self.tmp_loc = tmp_loc
        self.vid2imgids = (
            {}
        )  # map video_id to [list of frames to be forwarded, list of frames in the annotated moment]
        self.stride = stride
        ignore_vids = ['1_aMYcLyh9OhU.mkv'] # some videos will cause training error
        popsed_indice = []
        for i_vid, video in enumerate(self.annotations["videos"]):
            if video['video_path'] in ignore_vids:
                popsed_indice.append(i_vid)
                continue
            video_fps = video["fps"]  # used for extraction
            sampling_rate = fps / video_fps
            assert sampling_rate <= 1  # downsampling at fps
            start_frame = (
                video["start_frame"] if self.tmp_loc else video["tube_start_frame"]
            )
            end_frame = video["end_frame"] if self.tmp_loc else video["tube_end_frame"]
            frame_ids = [start_frame]
            for frame_id in range(start_frame, end_frame):
                if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
                    frame_ids.append(frame_id)

            if len(frame_ids) > video_max_len:  # subsample at video_max_len
                frame_ids = [
                    frame_ids[(j * len(frame_ids)) // video_max_len]
                    for j in range(video_max_len)
                ]

            inter_frames = set(
                [
                    frame_id
                    for frame_id in frame_ids
                    if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]
                ]
            )  # frames in the annotated moment
            # drop the samples without selected frames inside the temporal gt due to the sampling operation
            if len(inter_frames) == 0:
                popsed_indice.append(i_vid)
                continue
            self.vid2imgids[video["video_id"]] = [frame_ids, inter_frames]
        popsed_indice.reverse() # pop the error video from the end to the start
        for popsed_index in popsed_indice:
            self.annotations.pop(popsed_index)

    def __len__(self) -> int:
        return len(self.annotations["videos"])

    def __getitem__(self, idx):
        """
        :param idx: int
        :return:
        images: a CTHW video tensor
        targets: list of frame-level target, one per frame, dictionary with keys image_id, boxes, orig_sizes
        tmp_target: video-level target, dictionary with keys video_id, qtype, inter_idx, frames_id, caption
        """
        video = self.annotations["videos"][idx]
        caption = video["caption"]
        video_id = video["video_id"]
        video_original_id = video["original_video_id"]
        clip_start = video["start_frame"]  # included
        clip_end = video["end_frame"]  # excluded
        frame_ids, inter_frames = self.vid2imgids[video_id]
        trajectory = self.annotations["trajectories"][video_original_id][
            str(video["target_id"])
        ]

        if os.path.exists(os.path.join(self.vid_folder, "video", video["video_path"])):
            vid_path = os.path.join(self.vid_folder, "video", video["video_path"])
        else:
            vid_path = os.path.join(self.vid_folder, "video", video["video_path"].split('/')[-1])

        w = video["width"]
        h = video["height"]
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ids[0])
        idx = frame_ids[0]
        images_list = []
        while idx <= frame_ids[-1]:
            ret = cap.grab()
            if not ret:
                break
            if idx in frame_ids:
                ret, frame = cap.retrieve()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images_list.append(frame)
            idx += 1
        cap.release()
        images_list = np.stack(images_list)
        assert len(images_list) == len(frame_ids), (video["video_path"], len(images_list), len(frame_ids))

        # prepare frame-level targets
        targets_list = []
        inter_idx = []  # list of indexes of the frames in the annotated moment
        for i_img, img_id in enumerate(frame_ids):
            if img_id in inter_frames:
                bbox = trajectory[
                    str(img_id)
                ]  # dictionary with bbox [left, top, width, height] key
                anns = [bbox]
                inter_idx.append(i_img)
            else:
                anns = []
            target = prepare(w, h, anns)
            target["image_id"] = f"{video_id}_{img_id}"
            targets_list.append(target)

        # video spatial transform
        if self._transforms is not None:
            images, targets = self._transforms(images_list, targets_list)
        else:
            images, targets = images_list, targets_list

        if (
            inter_idx
        ):  # number of boxes should be the number of frames in annotated moment
            assert (
                len([x for x in targets if len(x["boxes"])])
                == inter_idx[-1] - inter_idx[0] + 1
            ), (len([x for x in targets if len(x["boxes"])]), inter_idx)

        # temporal crop
        if self.tmp_crop:
            p = random.random()
            if p > 0.5:  # random crop
                # list possible start indexes
                if inter_idx:
                    starts_list = [i for i in range(len(frame_ids)) if i < inter_idx[0]]
                else:
                    starts_list = [i for i in range(len(frame_ids))]

                # sample a new start index
                if starts_list:
                    new_start_idx = random.choice(starts_list)
                else:
                    new_start_idx = 0

                # list possible end indexes
                if inter_idx:
                    ends_list = [i for i in range(len(frame_ids)) if i > inter_idx[-1]]
                else:
                    ends_list = [i for i in range(len(frame_ids)) if i > new_start_idx]

                # sample a new end index
                if ends_list:
                    new_end_idx = random.choice(ends_list)
                else:
                    new_end_idx = len(frame_ids) - 1

                # update everything
                prev_start_frame = frame_ids[0]
                prev_end_frame = frame_ids[-1]
                frame_ids = [
                    x
                    for i, x in enumerate(frame_ids)
                    if new_start_idx <= i <= new_end_idx
                ]
                images = images[:, new_start_idx : new_end_idx + 1]  # CTHW
                targets = [
                    x
                    for i, x in enumerate(targets)
                    if new_start_idx <= i <= new_end_idx
                ]
                clip_start += frame_ids[0] - prev_start_frame
                clip_end += frame_ids[-1] - prev_end_frame
                if inter_idx:
                    inter_idx = [x - new_start_idx for x in inter_idx]

        if (
            self.is_train and len(frame_ids) > self.video_max_len_train
        ):  # densely sample video_max_len_train frames
            if inter_idx:
                starts_list = [
                    i
                    for i in range(len(frame_ids))
                    if inter_idx[0] - self.video_max_len_train < i <= inter_idx[-1]
                ]
            else:
                starts_list = [i for i in range(len(frame_ids))]

            # sample a new start index
            if starts_list:
                new_start_idx = random.choice(starts_list)
            else:
                new_start_idx = 0

            # select the end index
            new_end_idx = min(
                new_start_idx + self.video_max_len_train - 1, len(frame_ids) - 1
            )

            # update everything
            prev_start_frame = frame_ids[0]
            prev_end_frame = frame_ids[-1]
            frame_ids = [
                x for i, x in enumerate(frame_ids) if new_start_idx <= i <= new_end_idx
            ]
            images = images[:, new_start_idx : new_end_idx + 1]  # CTHW
            targets = [
                x for i, x in enumerate(targets) if new_start_idx <= i <= new_end_idx
            ]
            clip_start += frame_ids[0] - prev_start_frame
            clip_end += frame_ids[-1] - prev_end_frame
            if inter_idx:
                inter_idx = [
                    x - new_start_idx
                    for x in inter_idx
                    if new_start_idx <= x <= new_end_idx
                ]

        # video level annotations
        tmp_target = {
            "video_id": video_id,
            "qtype": video["qtype"],
            "inter_idx": [inter_idx[0], inter_idx[-1]]
            if inter_idx
            else [
                -100,
                -100,
            ],  # start and end (included) indexes for the annotated moment
            "frames_id": frame_ids,
            "caption": caption,
        }
        if self.stride:
            return images[:, :: self.stride], targets, tmp_target, images
        return images, targets, tmp_target


def build(image_set, args):
    vid_dir = Path(args.vidstg_vid_path)

    if image_set == "test":
        ann_file = Path(args.vidstg_ann_path) / f"test.json"
    elif image_set == "val":
        ann_file = Path(args.vidstg_ann_path) / f"val.json"
    else:
        ann_file = (
            Path(args.vidstg_ann_path) / f"train.json"
        )

    dataset = VideoModulatedSTGrounding(
        vid_dir,
        ann_file,
        transforms=make_video_transforms(
            image_set, cautious=True, resolution=args.resolution
        ),
        is_train=image_set == "train",
        video_max_len=args.video_max_len,
        video_max_len_train=args.video_max_len_train,
        fps=args.fps,
        tmp_crop=args.tmp_crop and image_set == "train",
        tmp_loc=args.sted,
        stride=args.stride,
    )
    return dataset
