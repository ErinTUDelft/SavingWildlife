##########################################################
# TESTING
##########################################################


# import from libraries
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import re
import json
import torch.nn as nn

# import from project
from network import class_local
from create_data_loader import test_loader


back_bone = torchvision.models.mobilenet_v3_small(pretrained=True)
model_class_local = class_local(back_bone)
model_class_local.load_state_dict(torch.load("saved_model.pt"))


def testing(test_loader, model):
    test_loss = 0.0
    data_test_images = test_loader

    # setup model to evaluate

    model.eval()

    # classification loss
    criterion_class = (
        nn.CrossEntropyLoss()
    )  # cross entropy loss between the predicted logits and the true labels (softmax)

    # regression loss
    criterion_reg = nn.MSELoss()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_test_images):
            # image
            # create input for model
            X = data["image"]

            # truths for class and bounding box

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

            else:
                # no detection -> background
                Y_class = torch.tensor([[0.0, 1.0]])
                Y_reg = torch.tensor([0.5, 0.5, 1, 1]).reshape(1, 4)

            class_pred, box_pred = model_class_local(X)
            # calculate loss

            # if background is closer to the truth than animal -> update only classification head
            if class_pred[0][0] < 0.5:
                loss = criterion_class(class_pred, Y_class)

            # if animal is closer to the truth than background
            else:
                loss = criterion_class(class_pred, Y_class) + criterion_reg(
                    box_pred, Y_reg
                )

            # average test loss
            test_loss += loss.item()
            avg_test_loss = test_loss / (batch_idx + 1)

            print("batch idx: ", batch_idx)
            print("average classification loss: ", avg_test_loss)

    return


# evaluate model
testing(test_loader, model_class_local)
