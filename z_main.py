### import modules from torch/torchvision library ###
import os
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim


from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision import datasets


### import modules from other files ###
from z_architecture import WildLifeModel
from z_optimizing import BoundingBoxLoss, CrossEntropyLoss, create_optimizer

## inputs variables ##
num_epochs = 10
num_classes = 2


### downloading dataset #####
Inaturalist = datasets.INaturalist(
    root="./data",
    version="2021_valid",
    transform=ToTensor(),
    download=False,
)

#### setting up data loader #####
train_data_loader = torch.utils.data.DataLoader(
    Inaturalist, batch_size=10, shuffle=True
)

# # len(train_data_loader))
# print(type(train_data_loader))


# iter_data_loader = iter(train_data_loader)
(
    images,
    labels,
) = train_data_loader  # returns an iterator. -> 1st batch of 64 images and labels
print(images.shape)
print(labels.shape)


# #### creating model #####
# device = (
#         "cuda"
#         # check if cuda enabled gpu accessible
#           if torch.cuda.is_available()
#         # set mps
#           else "mps"
#         # check if mps is available
#           if torch.backends.mps.is_available()
#         # set cpu
#           else "cpu"
# )


# model = WildLifeModel(num_classes=num_classes).to(device)
# print(model)

# ##### optimizing model parameters: loss & optimizer ######

# # Define the loss function and optimizer
# bounding_box_loss_fn = BoundingBoxLoss()
# classification_loss_fn = CrossEntropyLoss()
# optimizer = create_optimizer(model)

# # Define the input and target data
# x_train = torch.Tensor(...)  # Your input data
# y_train_bounding_box = torch.Tensor(...)  # Your bounding box labels
# y_train_classification = torch.Tensor(...)  # Your classification labels


# ##### training loop ######

# for epoch in range(num_epochs):
#     optimizer.zero_grad()

#     # Forward pass
#     y_pred_bounding_box, y_pred_classification = model(x_train)

#     # Compute losses
#     bounding_box_loss = bounding_box_loss_fn(y_train_bounding_box, y_pred_bounding_box)
#     classification_loss = classification_loss_fn(y_train_classification, y_pred_classification)

#     # Total loss
#     total_loss = bounding_box_loss + classification_loss

#     # Backward pass
#     total_loss.backward()
#     optimizer.step()


##### test loop ######
##### saving model ######
##### loading model ######
##### making predictions ######
##### saving predictions ######
##### loading predictions ######
##### visualizing predictions ######
##### saving visualizations ######
##### loading visualizations ######
