import os
import cv2
import torch
import random
from PIL import Image
from io import BytesIO
from functools import partial
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

import artist.data as data
import artist.ops as ops
from tools.annotator.sketch import pidinet_bsd, sketch_simplification_gan


def random_resize(img, size):
    img = [TF.resize(u, size, interpolation=random.choice([
        InterpolationMode.BILINEAR,
        InterpolationMode.BICUBIC,
        InterpolationMode.LANCZOS])) for u in img]
    return img


def gen_sketch(image_path, gpu=0, misc_size=384):

    sketch_mean = [0.485, 0.456, 0.406]
    sketch_std = [0.229, 0.224, 0.225]

    pidinet = pidinet_bsd(pretrained=True, vanilla_cnn=True).eval().requires_grad_(False).to(gpu)
    cleaner = sketch_simplification_gan(pretrained=True).eval().requires_grad_(False).to(gpu)
    pidi_mean = torch.tensor(sketch_mean).view(1, -1, 1, 1).to(gpu)
    pidi_std = torch.tensor(sketch_std).view(1, -1, 1, 1).to(gpu)   

    misc_transforms = data.Compose([
        T.Lambda(partial(random_resize, size=misc_size)),
        data.CenterCropV2(misc_size),
        data.ToTensor()])
    
    image = Image.open(open(image_path, mode='rb')).convert('RGB')
    image = misc_transforms([image]) # 
    image = image.to(gpu)

    sketch = pidinet(image.sub(pidi_mean).div_(pidi_std))
    sketch = 1.0 - cleaner(1.0 - sketch)
    sketch = sketch.cpu()

    sketch = sketch[0][0]
    sketch = (sketch.numpy()*255).astype('uint8')
    file_name = os.path.basename(image_path)
    save_pth = 'source/inputs/' + file_name.replace('.', '_sketch.')
    cv2.imwrite(save_pth, sketch)


gen_sketch(image_path='demo_video/sunflower.png')
