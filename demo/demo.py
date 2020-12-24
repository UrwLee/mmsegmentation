from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os

config_file = '/home/dmx/dmx/code/mmsegmentation/my_config/fcn_r50-d8_512x1024_40k_cocolip.py'
checkpoint_file = '/home/dmx/dmx/disk/ubuntu_file/checkpoints/FCN/mmseg/fcn_r50-d8_512x1024_20k_cocolip_base_v1/latest.pth'
dir = '/home/dmx/dmx/disk/ubuntu_file/data/Human_Seg_CoCo_Lip/val_image'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
ims = os.listdir(dir)
for im in ims:
    # test a single image and show the results
    img = mmcv.imread(os.path.join(dir,im))  # or img = mmcv.imread(img), which will only load it once
    result = inference_segmentor(model, img)
    # visualize the results in a new window
    model.show_result(img, result, show=True)
    # # or save the visualization results to image files
    # model.show_result(img, result, out_file='result.jpg')

# # test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_segmentor(model, frame)
#     model.show_result(frame, result, wait_time=1)