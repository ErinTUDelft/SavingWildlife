# import from libraries
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
import json
import torch.nn as nn

# import from project
from train_class_local import class_local
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
            distance_background = class_pred - torch.tensor([[0.0, 1.0]])
            distance_animal = class_pred - torch.tensor([[1.0, 0.0]])

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


## test an image
#### draw bounding box
# image_path = "./kenya_ims/8a2f5c20-21bc-11ea-a13a-137349068a90.jpg"
image_path = "./kenya_ims/8a0a30c6-21bc-11ea-a13a-137349068a90.jpg"
# image_path = "./kenya_ims/8a1dc35c-21bc-11ea-a13a-137349068a90.jpg"
# image_path = "./kenya_ims/8a0b9fba-21bc-11ea-a13a-137349068a90.jpg"
test_image = Image.open(image_path)
transform = transforms.Compose(
    [
        # you can add other transformations in this list
        # transforms.Grayscale(),
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
    ]
)


# test_image = transform(test_image).reshape(1, 3, 240, 240)

# back_bone = torchvision.models.mobilenet_v3_small(pretrained=True)
# model_class_local = class_local(back_bone)
# model_class_local.load_state_dict(torch.load("saved_model2.pt"))
# model_class_local.eval()

# class_pred, box_pred = model_class_local(test_image)
# print("class_pred: ", class_pred)
# print("box_pred: ", box_pred)


# # regenerate image from tensor

# image = cv2.imread(image_path)
# image = cv2.resize(image, (240, 240))

# x_left = float(box_pred[0][0])
# y_top = float(box_pred[0][1])
# x_width = float(box_pred[0][2])
# y_heigth = float(box_pred[0][3])

# x_min = abs(x_left - x_width / 2) * 240
# y_min = abs(y_top - y_heigth / 2) * 240
# x_max = abs(x_left + x_width / 2) * 240
# y_max = abs(y_top + y_heigth / 2) * 240


# print("x_min: ", x_min)
# print("y_min: ", y_min)
# print("x_max: ", x_max)
# print("y_max: ", y_max)

# bounding_box = (round(x_min), round(y_min), round(x_max), round(y_max))
# image = cv2.rectangle(
#     image,
#     (bounding_box[0], bounding_box[1]),
#     (bounding_box[2], bounding_box[3]),
#     (0, 255, 0),
#     2,
# )
# cv2.imshow("image with bounding box", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
