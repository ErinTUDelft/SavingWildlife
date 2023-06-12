##############################################################################################################
# Train classification & localization
##############################################################################################################

#######  #######

##### import modules ########

# TODO remove sofmax

## Import modules
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import re
import json
from PIL import Image
from tqdm import tqdm  # progress bar

# Import from project
from create_data_loader import train_loader
from network import class_local

######### Definitions
classes = 2  # no humans nor vehicles in the dataset
epochs = 10
#########


# #### Create model


back_bone = torchvision.models.mobilenet_v3_small(pretrained=True)
model = class_local(back_bone)
model.back1.requires_grad_ = False


# #### loss function & optimizer

# classification loss
criterion_class = (
    nn.CrossEntropyLoss()
)  # cross entropy loss between the predicted logits and the true labels (softmax)

# regression loss
criterion_reg = (
    nn.MSELoss()
)  # mean squared error loss between the predicted values and the true regression targets

# optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001
)  # only update parameters that have requires_grad = True


def training(model, dataloader, optimizer, criterion_class, criterion_reg):
    train_loss = 0.0
    correct_pred = 0.0

    model.train()
    for batch_idx, data in enumerate(dataloader):
        # if batch_idx % 100 == 0:
        #     print("batch_idx ", batch_idx)

        X = data["image"]

        #########################################
        # getting truths

        key_detection = data["landmarks"]["detections"][0]  # getting
        pattern = r"\{[^{}]+\}"
        match = re.search(pattern, key_detection)

        if match != None:
            ind_start = match.start()
            ind_end = match.end()

            # convert string to dictionary
            key_detection = json.loads(
                key_detection[ind_start:ind_end].replace("'", '"')
            )  # json loader can not parse single quotes

            Y_class = torch.tensor([[1.0, 0.0]])
            Y_reg = torch.tensor(key_detection["bbox"]).reshape(1, 4)
            category = "animal"
            label = 1

        else:
            # no detection -> background
            Y_class = torch.tensor([[0.0, 1.0]])
            Y_reg = torch.tensor([0.5, 0.5, 1, 1]).reshape(1, 4)
            category = "background"
            label = 0
        ##########################################

        # forward pass
        optimizer.zero_grad()

        output_class, output_reg = model(X)

        print("the output class ", output_class[0], "the truth ", category)
        # print(output_class)
        ### checking pictures

        if output_class[0][0] > output_class[0][1]:
            # if animal is closer to the truth than background
            loss = criterion_class(output_class, Y_class) + criterion_reg(
                output_reg, Y_reg
            )

        else:
            # background is closer to the truth than animal -> update only classification head
            model.regression_head.requires_grad_ = False
            loss = criterion_class(output_class, Y_class)

        # backpropagation
        loss.backward()

        # gradients
        optimizer.step()
        train_loss += loss.item()
        avg_loss = train_loss / (batch_idx + 1)

        # evaluate the model
        # print(" random", output_class[0][0].item())
        correct_pred += ((torch.argmax(output_class[0]), 1) == label).sum().item()
        eval_acc = correct_pred / len(dataloader.dataset)

        print(eval_acc)
        # print(f"average_loss: {avg_loss}")

        # if batch_idx % 100 == 0:
        #     print(f"average_loss: {avg_loss}")

        # save the model

    torch.save(model.state_dict(), "saved_model.pt")  # Export to TorchScript

    return


training(model, train_loader, optimizer, criterion_class, criterion_reg)


##########################################################################################

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


#########################################################################################
