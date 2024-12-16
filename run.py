import torch
import os
from torch.utils.data import random_split

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloader import RouteSegmentationDataset, TestDataset, SegmentationTransform

import albumentations as A

from skimage.transform import resize

from utils.utils import plot_loss, DiceLoss
from utils.resnet_unet import ResNetUnet

import matplotlib.image as mpimg
import torchvision.transforms as T

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
N_CLASSES = 1
N_CHANNELS = 3
BEST_CHECKPOINTS_PATH = "checkpoints1/best_checkpoints.pt"

image_dir = "training/images"
mask_dir = "training/groundtruth"
image_dir_test = "test_set_images"

# Define augmentations
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomResizedCrop(size=(256,256), scale=(0.25,0.5), ratio=(0.75, 1.3333333333333333), interpolation=1, mask_interpolation=0, p=0.5, always_apply=None),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
])

# Augment and save images
image_filenames = os.listdir(image_dir)
mask_filenames = os.listdir(mask_dir)

for i, (image_file, mask_file) in enumerate(zip(image_filenames, mask_filenames)):
    # Load image and mask
    image = np.array(Image.open(os.path.join(image_dir, image_file)))
    mask = np.array(Image.open(os.path.join(mask_dir, mask_file)))

    # Apply augmentations
    for j in range(35):
        augmented = augmentations(image=image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']
        
        # Save augmented data
        Image.fromarray(augmented_image).save(os.path.join(image_dir, f"aug_image_{100+i}_{j}.png"))
        Image.fromarray(augmented_mask).save(os.path.join(mask_dir, f"aug_mask_{100+i}_{j}.png"))

print("Augmented data saved!")

# Create datasets with consistent transformations
transform_training = SegmentationTransform()
transform_test = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

dataset = RouteSegmentationDataset(image_dir, mask_dir, transform=transform_training)
test_dataset = TestDataset(image_dir_test, transform=transform_test)

# Create DataLoaders
train_size = int(0.85 * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ResNetUnet(N_CHANNELS, N_CLASSES).to(device)

### TRAINING

loss_fn = DiceLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# Training loop 
training_losses = []
validation_losses = []

for epoch in range(EPOCHS):
    model.train()
    avg_loss = 0
    for batch, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        
    scheduler.step()

    avg_loss /= len(train_loader)
    training_losses.append(avg_loss)
    if epoch % 5 == 0:
        print(f"Training loss for epoch {epoch}: {avg_loss:>7f}")

    avg_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in validation_loader:
            X, y = X.to(device), y.to(device)
        
            pred = model(X)

            loss = loss_fn(pred, y)
            avg_loss += loss.item()

        avg_loss /= len(validation_loader)
        validation_losses.append(avg_loss)
        
        print(f"Validation loss for epoch {epoch}: {avg_loss:>7f}")

        if len(validation_losses) == 0 or avg_loss <= min(validation_losses): # save the best epoch checkpoints
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()  
                }, BEST_CHECKPOINTS_PATH)

    plot_loss(training_losses, validation_losses)

### TEST

# load test set images for later use
test_images = []
for i in range(50):
    mask_path = os.path.join(image_dir_test, sorted(os.listdir(image_dir_test), key=lambda x: int(x.split('_')[1].split('.')[0]))[i])
    test_images.append(np.array(Image.open(mask_path)))

# Set model to evaluation mode
model.eval()

# Place to store predictions
all_predictions = []

with torch.no_grad():
    for image in test_dataset:
        image = image.unsqueeze(0).to(device)

        output = model(image)
        output = output.cpu().detach().numpy()

        threshold = 0.25
        predicted_labels = (output > threshold).astype(np.float32)

        all_predictions.append(predicted_labels)

all_predictions = np.concatenate(all_predictions, axis=0)
final_all_predictions = []

for i in range(50):
    final_all_predictions.append(resize(all_predictions[i][0], test_images[i].shape, order=0))

# create predictions on test set
os.makedirs("test_set_preds", exist_ok=True)

for i in range(len(all_predictions)):
    mask = final_all_predictions[i]
    mask_uint8 = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_uint8).save(f"test_set_preds/pred_{i + 1}.png")

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename, img_number):
    """Reads a single image and outputs the strings that should go into the submission file"""
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        i = 1
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, i))
            i+=1


submission_filename = "submissions.csv"
image_filenames = []
for i in range(1, 51):
    image_filename = f"test_set_preds/pred_{i}.png"
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)