# AxxessHack
Pose Estimation Python

# Installing

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (for CUDA enabled)

or

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (for NO CUDA)


`pip install -U openmim`

`mim install mmcv-full`

`pip install mmpose`

`mim download mmpose --config td-hm_alexnet_8xb64-210e_coco-256x192 --dest .`

and then run `sample_inf.py` to make sure everything works.