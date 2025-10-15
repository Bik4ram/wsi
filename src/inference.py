import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob
import os
from unet_model import UNet

class InferenceDataset(Dataset):
    def __init__(self, images_dir):
        self.images = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))
        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)/255
        return image, self.images[idx]

def run_inference(images_dir, checkpoint_path, output_dir, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    dataset = InferenceDataset(images_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.eval()
    with torch.no_grad():
        for img, fname in loader:
            img = img.to(device)
            mask = model(img)
            mask = (mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            out_name = os.path.basename(fname[0]).replace('.png', '_mask.png')
            Image.fromarray(mask).save(os.path.join(output_dir, out_name))# Inference module code here
