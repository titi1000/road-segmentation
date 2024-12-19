import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform

def plot_loss(training_losses, validation_losses):
    plt.plot(training_losses, "b")
    plt.plot(validation_losses, "r")
    plt.savefig("plot.png")

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Dice Loss for binary segmentation tasks.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice_coef = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_coef
    
class RandomShadow(ImageOnlyTransform):
    def __init__(self, shadow_color=(0, 0, 0), intensity=0.5, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.shadow_color = shadow_color
        self.intensity = intensity

    def apply(self, img, **params):
        h, w = img.shape[:2]
        shadow_mask = np.zeros((h, w), dtype=np.uint8)

        num_vertices = np.random.randint(4, 8)
        points = np.random.randint(0, min(h, w), size=(num_vertices, 2))

        cv2.fillPoly(shadow_mask, [points], color=255)

        if len(img.shape) == 3:
            shadow_mask = cv2.merge([shadow_mask] * 3)

        shadow = (np.array(self.shadow_color, dtype=np.uint8) * self.intensity).astype(np.uint8)
        shadowed_img = cv2.addWeighted(img, 1, shadow_mask, self.intensity, 0)

        return np.where(shadow_mask > 0, shadow, img)