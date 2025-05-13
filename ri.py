import torch
from torchvision.io import read_image
img = read_image('example-image.png')
print('Image shape:', img.shape)
print('Number of channels:', img.shape[0])
print('Image data type:', img.dtype)
print(img[:, 100:102, 100:102])