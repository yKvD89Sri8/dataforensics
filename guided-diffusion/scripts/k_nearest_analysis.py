import os
import numpy as np
from PIL import Image
from imageio import imread, imsave
import torch
import torchvision.datasets as datasets
from tqdm import tqdm

query_file = ""
ds_file = ""

generate = np.load('tmp/openai-2022-11-21-22-02-56-116053/samples_4x64x64x3.npz', mmap_mode='r')['arr_0']
# generate = np.load('/home/donghao/GAN-Leaks/gan_models/pggan/results/pggan_default/003-pgan-cifar10_train-preset-v2-1gpu/generated.npz', mmap_mode='r')['img_r01']
for i, img in enumerate(tqdm(generate)):
    im = Image.fromarray((img).astype(np.uint8))
    # im = Image.fromarray((img*255).astype(np.uint8))
    im.save(f"tmp/synthetic_cifar10_diffusion_15k/{i:05d}.png")
