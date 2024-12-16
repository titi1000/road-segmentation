import matplotlib.pyplot as plt
import torch.nn as nn

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