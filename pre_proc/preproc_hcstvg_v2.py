import json
import os
from tqdm import tqdm

# load config
with open("config/hcstvg_v2.json", "r") as f:
    cfg = json.load(f)
video_path = os.path.join(cfg['hcstvg_vid_path'], "video")
anno_save_path = cfg['hcstvg_ann_path']
anno_ori_path = cfg['hcstvg_ori_ann_path']
if not os.path.exists(anno_save_path):
    os.makedirs(anno_save_path)
# get video to path mapping
dirs = os.listdir(video_path)
vid2path = {}
# for dir in dirs:
#     files = os.listdir(os.path.join(video_path, dir))
#     for file in files:
#         assert os.path.exists(os.path.join(video_path, dir, file))
#         vid2path[file[:-4]] = os.path.join(dir, file)
for file in dirs:
    assert os.path.exists(os.path.join(video_path, file))
    vid2path[file[:-4]] = file

# preproc annotations
files = ["train_v2.json", "val_v2.json"]
for file in files:
    videos = []
    annotations = json.load(open(os.path.join(anno_ori_path, file), "r"))
    for video, annot in tqdm(annotations.items()):
        out = {
            "original_video_id": video[:-4],
            "frame_count": annot["img_num"],
            "width": annot["img_size"][1],
            "height": annot["img_size"][0],
            "tube_start_frame": annot["st_frame"],  # starts with 1
            # excluded
            "tube_end_frame": annot["st_frame"] + len(annot["bbox"]),
            "tube_start_time": annot["st_time"],
            "tube_end_time": annot["ed_time"],
            "video_path": vid2path[video[:-4]],
            "caption": annot["English"],
            "video_id": len(videos),
            "trajectory": annot["bbox"],
        }
        videos.append(out)

    json.dump(videos, open(os.path.join(
        anno_save_path, file.split('_')[0] + ".json"), "w"))
