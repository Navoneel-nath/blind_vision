import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
model.eval()

raw_img = cv2.imread(r"C:\Users\navon\Documents\capstone\dataset\left.png")
depth = model.infer_image(raw_img) # HxW raw depth map
