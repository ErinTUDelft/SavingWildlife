####### classification & localisation #######

##### import modules ########

"""
Changed the following: (Erin)
- set grad back2 to false
- changed 0.5 to output_class[0][1]
- added util fuctions
- added epoch loop
"""

"""
ToDO:
-The regression loss is much lower than the classification loss; maybe we need to scale it? (multiply by 5 or so)
-Clean up the dataset
-Data augmentation? (rotation, flip, noise, etc.) There are some easy torch functions for this
-Add model to network.py and import it here for cleanlyness
"""

"""
Right now things seem to be actually be working really well! The average correct goes up to a decently high number. 
Final challenge is to get the bounding boxes to work right. I have not yet trained the model for the full training set and 
number of epochs. Give it a try if you want! (But I'm expecting we need to ask Korneel to clean the dataset first)
"""

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
from utils import classification_info, regression_info


######### Definitions
classes = 2  # no humans nor vehicles in the dataset
epochs = 2
#########


class class_local(nn.Module):
    """classification with localisation neural network

    Args:
        nn.Module: initialises the class_local class with nn.Module as parent class
    """

    def __init__(self, back_bone):
        super(class_local, self).__init__()
        self.back1 = nn.Sequential(
            *nn.ModuleList(back_bone.children())[:-1]
        )  # out_features 1024
        self.back2 = back_bone.classifier[0]

        # classification head
        self.classification_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=classes, bias=True),
            # nn.Linear(in_features=1024, out_features=256, bias=True),
            # nn.ReLU(),
            # nn.Linear(in_features=256, out_features=128, bias=True),
            # nn.ReLU(),
            # nn.Linear(in_features=128, out_features=classes, bias=True),
            nn.Sigmoid(),
        )
        # localisation head
        self.regression_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=4, bias=True),
            nn.Sigmoid(),
        )
        #     nn.Linear(in_features=1024, out_features=256, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=128, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128, out_features=4, bias=True),
        #     nn.Sigmoid(),
        # )

    def forward(self, x):
        x = self.back1(x)
        x = x.view(x.size(0), -1)
        x = self.back2(x)    
        y_class = self.classification_head(x) 
        y_reg = self.regression_head(x)
        return y_class, y_reg


back_bone = torchvision.models.mobilenet_v3_small(pretrained=True)
model = class_local(back_bone)
model.back1.requires_grad_ = False
model.back2.requires_grad_ = False # I think this should have no grad, maybe later for fine tuning
print('model', model)

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

    model.train()
    total = 0
    correct = 0

    for epoch in range(epochs):
        for batch_idx, data in enumerate(dataloader):
            # if batch_idx % 100 == 0:
            #     print("batch_idx ", batch_idx)
            animal = False

            X = data["image"]

            #########################################
            # getting truths

            key_detection = data["landmarks"]["detections"][0]  # getting
            pattern = r"\{[^{}]+\}"
            match = re.search(pattern, key_detection)

            if match != None:
                animal = True
                ind_start = match.start()
                ind_end = match.end()

                # convert string to dictionary
                key_detection = json.loads(
                    key_detection[ind_start:ind_end].replace("'", '"')
                )  # json loader can not parse single quotes

                Y_class = torch.tensor([[1.0, 0.0]])
                Y_reg = torch.tensor(key_detection["bbox"]).reshape(1, 4)

            else:
                # no detection -> backgroundlater 
                animal = False
                Y_class = torch.tensor([[0.0, 1.0]])
                #Y_reg = torch.tensor([0.5, 0.5, 1, 1]).reshape(1, 4)
                Y_reg = [] # to really make sure it does not accidentally train on this
        

            # forward pass
            optimizer.zero_grad()
            output_class, output_reg = model(X)
 
            """
            With the changes I made this if statement is not necessary anymore, but I kept it in just in case
            """
            if output_class[0][0] > output_class[0][1]:
                # if animal is closer to the truth than background
                classification_loss = criterion_class(output_class, Y_class) 
            else:
                # background is closer to the truth than animal -> update only classification head
                # model.regression_head.requires_grad_ = False # I'm putting this below
                # maybe back2 should be grad false too 
                classification_loss = criterion_class(output_class, Y_class)

            """
            With this new approach we only train the bounding box when it is in the ground truth, 
            I *think* that is better than training on predicted classifications (also prevents the neural network
            from just predicting "background" the whole time in order to reduce its loss)
            """
            if animal==True: # There is an animal (I know the ==True can be ommited, but like to add it for clarity)
                regression_loss = criterion_reg(output_reg, Y_reg)
                loss = classification_loss + regression_loss
                model.regression_head.requires_grad_ = True

            else:
                loss = classification_loss
                model.regression_head.requires_grad_ = False

            # backpropagation
            loss.backward()
            # gradients
            optimizer.step()
            train_loss += loss.item()
            avg_loss = train_loss / (batch_idx + 1)
            print(f"average_loss: {avg_loss}")

            total += 1
            """
            I added some more info which can be commented out if not needed
            """
            # (comment if not needed)
            correct, average_correct = classification_info(total, output_class, Y_class, correct)
            print('Classification loss', classification_loss)
            info = regression_info(output_reg, Y_reg, criterion_reg, animal)

    torch.save(model.state_dict(), "saved_model3.pt")  # Export to TorchScript
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
