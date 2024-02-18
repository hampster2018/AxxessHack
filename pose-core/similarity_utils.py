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


def similarity_score_keypoints(json1, json2):
    total_similarity = 0
    total_count = 0
    for key in json1:

        if key in ["left_foot", "right_foot"]:
            continue

        # convert the coordinates to numpy arrays
        coord1 = np.array(json1[key])
        coord2 = np.array(json2[key])

        # pad the coordinates
        if len(coord1) < len(coord2):
            coord1 = np.pad(
                coord1,
                ((0, len(coord2) - len(coord1)), (0, 0)),
                "linear_ramp",
            )
        elif len(coord2) < len(coord1):
            coord2 = np.pad(
                coord2,
                ((0, len(coord1) - len(coord2)), (0, 0)),
                "linear_ramp",
            )

        print(f"similarity of {key} is {shape_similarity(coord1, coord2)}")

        # add the similarity to the total
        total_similarity += shape_similarity(coord1, coord2)
        total_count += 1

    return total_similarity / total_count


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

    # calculate the similarity
    similarity = similarity_score_keypoints(pose_output1, pose_output2)
    print(similarity)


if __name__ == "__main__":
    main()
