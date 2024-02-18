from typing import List
from mmpose.apis import MMPoseInferencer
from dataclasses import dataclass
from mmpose.utils import register_all_modules
import cv2
import numpy as np
import os
import time
import json
from shapesimilarity import shape_similarity


def main():
    # grab the jsons from the pose-core folder
    json_path1 = r"D:\AxxessHack\pose-core\pro_jumping_jacks_cut_coordinates.json"
    json_path2 = r"D:\AxxessHack\pose-core\cut_video_eric_jump_coordinates.json"

    # load the jsons
    pose_output1 = None
    pose_output2 = None
    with open(json_path1) as f:
        pose_output1 = json.load(f)
    with open(json_path2) as f:
        pose_output2 = json.load(f)

    # compare the jsons using numpy and shapesimilarity
    left_hand1 = pose_output1["left_hand"]
    # since the coordinates are in the form [[x1, y1], [x2, y2], ...]
    left_hand1_conv = []
    for coord in left_hand1:
        left_hand1_conv.append([coord[0], coord[1]])
    left_hand1_conv = np.array(left_hand1_conv)

    left_hand2 = pose_output2["left_hand"]
    left_hand2_conv = []
    for coord in left_hand2:
        left_hand2_conv.append([coord[0], coord[1]])
    left_hand2_conv = np.array(left_hand2_conv)

    # for whichever coordinates are shorter, pad them with points in-between the existing points
    # until they are the same length
    if len(left_hand1_conv) < len(left_hand2_conv):
        left_hand1_conv = np.pad(
            left_hand1_conv,
            ((0, len(left_hand2_conv) - len(left_hand1_conv)), (0, 0)),
            "linear_ramp",
        )
    elif len(left_hand2_conv) < len(left_hand1_conv):
        left_hand2_conv = np.pad(
            left_hand2_conv,
            ((0, len(left_hand1_conv) - len(left_hand2_conv)), (0, 0)),
            "linear_ramp",
        )

    print(
        "Similarity between left hand coordinates:",
        shape_similarity(left_hand1_conv, left_hand2_conv),
    )


if __name__ == "__main__":
    main()
