import openslide
import os
from PIL import Image

def tile_wsi(wsi_path, output_dir, tile_size=1024, overlap=0):
    slide = openslide.OpenSlide(wsi_path)
    width, height = slide.dimensions
    os.makedirs(output_dir, exist_ok=True)
    idx = 0
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            tile = slide.read_region((x, y), 0, (tile_size, tile_size))
            tile = tile.convert("RGB")
            tile.save(os.path.join(output_dir, f"tile_{idx}_{x}_{y}.png"))
            idx += 1
    slide.close()# Tiling module code here
