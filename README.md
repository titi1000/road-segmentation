# Road Segmentation Using UNet and ResNet-UNet Architectures: A Comparative Study

### Authors: 
Quentin Chappuis, Louis Martins and Kelu Huang

### Dataset

The utilized dataset consists of 100 satellite images of roads along with their corresponding ground truth masks.

### How to run?
1. Install PyTorch from [here](https://pytorch.org/)
2. Install the requirements as follows :
```
pip install -r requirements.txt
```
3. Extract the data from [here](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files?unique_download_uri=403803&challenge_id=68) and place them at the root of the repository. The training data should be in a folder named `training` and the test data in a folder named `test_set_images`
4. Run the `run.py` file
5. Get your predictions on the test set in the file `submissions.csv` placed at the root of the repository

### Results

The model's performance on the test set achieved an F1 score of `0.829` and an accuracy of `0.908`. This was accomplished using a ResNet-UNet architecture trained over 50 iterations with a learning rate of 1×10−4. For this project we used the Binary Cross-Entropy (BCE) loss.