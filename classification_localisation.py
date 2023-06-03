####### classification & localisation #######

##### import modules ########

## Import modules
import torch
import torch.optim as optim
import torch.nn as nn
import re
import json

import torchvision
import torchvision.transforms as transforms

# from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

from tqdm import tqdm  # progress bar

# Import from project
from create_data_loader import dataloader

# from optimizing import create_optimizer, BoundingBoxLoss, CrossEntropyLoss


######### Definitions
classes = 2  # no humans nor vehicles in the dataset
epochs = 10
#########


# ###### model architecture

# # 1. classification head
# # fully connected layer
# # relu

# # 2. regression head
# # fully connected layer
# # relu

back_bone = torchvision.models.mobilenet_v3_small(pretrained=True)


class class_local(nn.Module):
    """classification with localisation neural network

    Args:
        nn.Module: initialises the class_local class with nn.Module as parent class
    """

    def __init__(self, pretrained_model):
        super(class_local, self).__init__()
        self.pretrained_model = pretrained_model

        self.classification_head = nn.Sequential(
            nn.Linear(in_features=1000, out_features=classes, bias=True),
            nn.ReLU(),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(in_features=1000, out_features=4, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        # x = self.pretrained_model.features(
        #     x
        # )
        x = self.pretrained_model(x)
        # .features removes the final layer of the backbone
        y_class = self.classification_head(x)
        y_reg = self.regression_head(x)

        return y_class, y_reg


#### model and device

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = class_local(back_bone)  # .to(device)
model.pretrained_model.requires_grad_ = False

#### loss function & optimizer

# classification loss -> cross entropy loss between the predicted logits and the true labels (softmax)
criterion_class = nn.CrossEntropyLoss()

# regression loss -> mean squared error loss between the predicted values and the true regression targets
criterion_reg = nn.MSELoss()

# optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.1
)  # only update parameters t
# that have requires_grad = True

##### input description

# # X train input
# # Y1 classification head (truth)
# # Y2 regression head (truth)

#### description of the data
# 2 classes: animals (category 1) and background (category 2)

# for animals the bounding box is given by the values inside bbox
# for background the bounding box is set to the image size "bbox": [0.5,0.5,1,1]
# 0.5,0.5 -> center of the image
# 1st value x coordinate from the top left corner
# 2nd value y coordinate from the top left corner
# 1,1 -> width and height of the image


##### training
# def train(model, train_loader, optimizer, criterion_class,cirterion_reg):


# goal retrieve the first detection from the list of detections
# problem for data["landmarks"]["detections"]:  string inside of list contains dictionaries

# dataloader_iterator = iter(dataloader)
train_loss = 0.0

model.train()
for batch_idx, data in tqdm(enumerate(dataloader)):
    ######## loading inputs and truths ########
    # image
    X = data["image"]

    # classification & localisation truth

    # go inside of outer list
    key_detection = data["landmarks"]["detections"][0]

    # create a parser to find the first dictionary element (string)
    pattern = r"\{[^{}]+\}"
    match = re.search(pattern, key_detection)

    if match != None:
        ind_start = match.start()
        ind_end = match.end()

        # convert string to dictionary
        key_detection = json.loads(
            key_detection[ind_start:ind_end].replace("'", '"')
        )  # json loader can not parse single quotes

        # Y_class = torch.tensor(key_detection["conf"])
        Y_class = torch.tensor([[1.0, 0.0]])
        Y_reg = torch.tensor(key_detection["bbox"]).reshape(1, 4)

    else:
        # no detection -> background
        Y_class = torch.tensor([[0.0, 1.0]])
        Y_reg = torch.tensor([0.5, 0.5, 1, 1]).reshape(1, 4)

    # print(Y_class, Y_reg)
    ##########################################

    ##### forward pass

    optimizer.zero_grad()

    output_class, output_reg = model(X)
    # print("output_class: ", output_class)
    # print("output_reg: ", output_reg)

    # calculate loss
    loss_class = criterion_class(output_class, Y_class)
    loss_reg = criterion_reg(output_reg, Y_reg)

    # total training loss
    loss = loss_class + loss_reg

    # backpropagation
    loss.backward()

    # gradients
    optimizer.step()
    train_loss = train_loss + 1 / (batch_idx + 1) * (loss - train_loss)
    print("train_loss: ", train_loss)
    # save the model

model_scripted = torch.jit.script(model)  # Export to TorchScript
model_scripted.save("model_scripted.pt")  # Save
