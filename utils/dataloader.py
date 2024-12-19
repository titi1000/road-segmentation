from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import functional as TF
import random
import torchvision.transforms as T


class RouteSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask

class SegmentationTransform:
    def __init__(self):
        self.resize = T.Resize((384, 384))
    
    def __call__(self, image, mask):
        image = self.resize(image)
        mask = self.resize(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        angle = random.randint(0, 3) * 90
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask


class TestDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.image_files = sorted(os.listdir(images_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image