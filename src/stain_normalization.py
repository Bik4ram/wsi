import staintools
import os

def normalize_tiles(tiles_dir, target_image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    target = staintools.read_image(target_image_path)
    normalizer = staintools.StainNormalizer(method='macenko')
    normalizer.fit(target)
    for fname in os.listdir(tiles_dir):
        if fname.endswith('.png'):
            tile_path = os.path.join(tiles_dir, fname)
            tile = staintools.read_image(tile_path)
            norm_tile = normalizer.transform(tile)
            out_path = os.path.join(output_dir, fname)
            staintools.write_image(norm_tile, out_path)# Stain normalization module code here
