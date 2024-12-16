# Road Segmentation with ResNet-UNet

### Authors: 
Quentin Chappuis, Louis Martins and Kelu Huang

### Dataset

The dataset utilized consists of 100 satellite images of roads along with their corresponding ground truth masks.

### How to run?
1. Install PyTorch from [here](https://pytorch.org/)
2. Install the requirements as follows :
```
pip install -r requirements.txt
```
3. Extract the data from [here](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files?unique_download_uri=403803&challenge_id=68) and place them at the root of the repository
4. Run the `run.py` file
5. Get your predictions on the test set in the file `submissions.csv` placed at the root of the repository

### Results

The predicted result on the test set of this model resulted in a F1 score of `0.434` and an accuracy score of `0.870`. For this we used regularized logistic regression over 1000 iterations, with a Learning Rate of 1e-1 and a regularizer of 1e-7.

### Tree

```tree

```