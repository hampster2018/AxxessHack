from typing import List
from mmpose.apis import MMPoseInferencer
from dataclasses import dataclass
from mmpose.utils import register_all_modules
import cv2
import numpy as np
import os
import time
import json


@dataclass
class PoseCoordinate:
    x: int
    y: int

    def __repr__(self):
        return f"[{self.x}, {self.y}]"


@dataclass
class PoseOutput:
    coordinates: List[PoseCoordinate]

    def __repr__(self):
        return f"[{self.coordinates}]"


def prepare_inference() -> MMPoseInferencer:
    register_all_modules()
    inferencer = MMPoseInferencer(
        pose2d=r"D:\AxxessHack\pose-core\td-hm_alexnet_8xb64-210e_coco-256x192.py",
        pose2d_weights=r"D:\AxxessHack\pose-core\alexnet_coco_256x192-a7b1fd15_20200727.pth",
    )
    return inferencer


def get_video_hands_and_feet_coordinates(
    inferencer: MMPoseInferencer, video_path: str
) -> any:
    result_generator = inferencer(video_path)

    output = []

    names = {
        9: "left_hand",
        10: "right_hand",
        15: "right_foot",
        16: "left_foot",
    }

    for result in result_generator:
        frame_output = {}
        for index, keypoint in enumerate(result["predictions"][0][0]["keypoints"]):
            x, y = keypoint[0:2]
            if index in names:
                frame_output[names[index]] = PoseCoordinate(x=int(x), y=int(y))
        output.append(frame_output)

    return output


def get_frame_hands_and_feet_coordinates(inferencer: MMPoseInferencer, frame) -> any:
    result_generator = inferencer(frame)

    output = {}

    names = {
        9: "left_hand",
        10: "right_hand",
        15: "right_foot",
        16: "left_foot",
    }

    result = next(result_generator)

    for index, keypoint in enumerate(result["predictions"][0][0]["keypoints"]):
        x, y = keypoint[0:2]
        if index in names:
            output[names[index]] = PoseCoordinate(x=int(x), y=int(y))

    return output


def main():
    inferencer = prepare_inference()

    # get video frames and then draw them with cv2 imshow
    video_name = "pro_jumping_jacks_cut"
    video_path = r"D:\AxxessHack\pose-core\\" + f"{video_name}.mp4"
    cap = cv2.VideoCapture(video_path)
    left_hand_coordinates = []
    right_hand_coordinates = []
    left_foot_coordinates = []
    right_foot_coordinates = []
    video_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_output = get_frame_hands_and_feet_coordinates(inferencer, frame)

        # draw the points
        for key, value in frame_output.items():
            cv2.circle(frame, (int(value.x), int(value.y)), 5, (0, 255, 0), -1)
            # write the text
            cv2.putText(
                frame,
                key,
                (int(value.x), int(value.y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            if key == "left_hand":
                left_hand_coordinates.append(value)
            elif key == "right_hand":
                right_hand_coordinates.append(value)
            elif key == "left_foot":
                left_foot_coordinates.append(value)
            elif key == "right_foot":
                right_foot_coordinates.append(value)

        # draw a line between the previous points up to the current point
        if left_hand_coordinates:
            for i in range(1, len(left_hand_coordinates)):
                cv2.line(
                    frame,
                    (left_hand_coordinates[i - 1].x, left_hand_coordinates[i - 1].y),
                    (left_hand_coordinates[i].x, left_hand_coordinates[i].y),
                    (255, 255, 255),
                    2,
                )

        if right_hand_coordinates:
            for i in range(1, len(right_hand_coordinates)):
                cv2.line(
                    frame,
                    (right_hand_coordinates[i - 1].x, right_hand_coordinates[i - 1].y),
                    (right_hand_coordinates[i].x, right_hand_coordinates[i].y),
                    (255, 255, 255),
                    2,
                )

        if left_foot_coordinates:
            for i in range(1, len(left_foot_coordinates)):
                cv2.line(
                    frame,
                    (left_foot_coordinates[i - 1].x, left_foot_coordinates[i - 1].y),
                    (left_foot_coordinates[i].x, left_foot_coordinates[i].y),
                    (255, 255, 255),
                    2,
                )

        if right_foot_coordinates:
            for i in range(1, len(right_foot_coordinates)):
                cv2.line(
                    frame,
                    (right_foot_coordinates[i - 1].x, right_foot_coordinates[i - 1].y),
                    (right_foot_coordinates[i].x, right_foot_coordinates[i].y),
                    (255, 255, 255),
                    2,
                )

        cv2.imshow("frame", frame)

        # append the frame to the video_frames list
        video_frames.append(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # write the frames to a new video
    out = cv2.VideoWriter(
        r"D:\AxxessHack\pose-core\\" + f"{video_name}_output.mp4",
        -1,
        30,
        (video_frames[0].shape[1], video_frames[0].shape[0]),
    )

    for frame in video_frames:
        out.write(frame)

    out.release()

    print("Done!")

    # convert the coordinates to lists of list of integers
    left_hand_coordinates = [[coord.x, coord.y] for coord in left_hand_coordinates]
    right_hand_coordinates = [[coord.x, coord.y] for coord in right_hand_coordinates]
    left_foot_coordinates = [[coord.x, coord.y] for coord in left_foot_coordinates]
    right_foot_coordinates = [[coord.x, coord.y] for coord in right_foot_coordinates]

    # save the coordinates to a json file
    with open(f"D:\\AxxessHack\\pose-core\\{video_name}_coordinates.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "left_hand": left_hand_coordinates,
                    "right_hand": right_hand_coordinates,
                    "left_foot": left_foot_coordinates,
                    "right_foot": right_foot_coordinates,
                }
            )
        )


if __name__ == "__main__":
    main()
