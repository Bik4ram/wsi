import torch
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import numpy as np
from unet_model import UNet
from utils import dice_score, save_checkpoint
import os

class TilesDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted(glob.glob(os.path.join(images_dir, '*.png')))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, '*.png')))
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))
        mask = np.array(Image.open(self.masks[idx]).convert('L'))
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)/255
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)/255
        return image, mask

def train(images_dir, masks_dir, epochs=30, batch_size=8, lr=1e-4, device='cuda'):
    dataset = TilesDataset(images_dir, masks_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_dice += dice_score(preds, masks).item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}, Dice: {epoch_dice/len(loader):.4f}")
        save_checkpoint(model, optimizer, epoch, f"checkpoint_epoch{epoch+1}.pt")
    return model# Training module code here
