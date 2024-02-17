from mmpose.apis import MMPoseInferencer
from mmpose.utils import register_all_modules

register_all_modules()

img_path = "IMG_3072.MOV"

inferencer = MMPoseInferencer(
    pose2d="td-hm_alexnet_8xb64-210e_coco-256x192.py",
    pose2d_weights="alexnet_coco_256x192-a7b1fd15_20200727.pth",
)

result_generator = inferencer(img_path, show=True)
result = next(result_generator)
while result is not None:
    result = next(result_generator)
