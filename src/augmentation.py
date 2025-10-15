import albumentations as A
import os
import numpy as np
from PIL import Image

def augment_tiles(tiles_dir, output_dir, num_aug=3):
    os.makedirs(output_dir, exist_ok=True)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.3),
        A.ElasticTransform(p=0.2)
    ])
    idx = 0
    for fname in os.listdir(tiles_dir):
        if fname.endswith('.png'):
            img = np.array(Image.open(os.path.join(tiles_dir, fname)))
            for i in range(num_aug):
                aug_img = transform(image=img)["image"]
                out_name = f"{os.path.splitext(fname)[0]}_aug{i}.png"
                Image.fromarray(aug_img).save(os.path.join(output_dir, out_name))
                idx += 1# Augmentation module code here
