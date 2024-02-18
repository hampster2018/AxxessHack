from typing import List
from mmpose.apis import MMPoseInferencer
from dataclasses import dataclass
from mmpose.utils import register_all_modules


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
) -> List[PoseOutput]:
    result_generator = inferencer(video_path)

    results: List[PoseOutput] = []

    for result in result_generator:
        current_pose = []
        for index, keypoint in enumerate(result["predictions"][0][0]["keypoints"]):
            x, y = keypoint[0:2]
            current_pose.append(PoseCoordinate(x=x, y=y))
        results.append(PoseOutput(coordinates=current_pose))

    return results
