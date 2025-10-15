import numpy as np
from PIL import Image
import os
import re

def stitch_masks(masks_dir, output_path, wsi_shape, tile_size=1024, overlap=0):
    stitched_mask = np.zeros(wsi_shape, dtype=np.uint8)
    for fname in os.listdir(masks_dir):
        if fname.endswith('_mask.png'):
            match = re.match(r'tile_\d+_(\d+)_(\d+)_mask.png', fname)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                mask = np.array(Image.open(os.path.join(masks_dir, fname)).convert('L'))
                stitched_mask[y:y+tile_size, x:x+tile_size] = np.maximum(
                    stitched_mask[y:y+tile_size, x:x+tile_size], mask)
    Image.fromarray(stitched_mask).save(output_path)# Stitching module code here
