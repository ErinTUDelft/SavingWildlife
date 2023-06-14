####################################################################################
# VALIDATION
####################################################################################

from PIL import Image
from torchvision import transforms
import torchvision
import torch
import cv2

from network import class_local


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


test_image = transform(test_image).reshape(1, 3, 240, 240)

back_bone = torchvision.models.mobilenet_v3_small(pretrained=True)
model_class_local = class_local(back_bone)
model_class_local.load_state_dict(torch.load("saved_model3.pt"))
model_class_local.eval()

class_pred, box_pred = model_class_local(test_image)
print("class_pred: ", class_pred)
print("box_pred: ", box_pred)


# regenerate image from tensor

image = cv2.imread(image_path)
image = cv2.resize(image, (240, 240))

x_left = float(box_pred[0][0])
y_top = float(box_pred[0][1])
x_width = float(box_pred[0][2])
y_heigth = float(box_pred[0][3])

x_min = abs(x_left - x_width / 2) * 240
y_min = abs(y_top - y_heigth / 2) * 240
x_max = abs(x_left + x_width / 2) * 240
y_max = abs(y_top + y_heigth / 2) * 240


print("x_min: ", x_min)
print("y_min: ", y_min)
print("x_max: ", x_max)
print("y_max: ", y_max)

bounding_box = (round(x_min), round(y_min), round(x_max), round(y_max))
image = cv2.rectangle(
    image,
    (bounding_box[0], bounding_box[1]),
    (bounding_box[2], bounding_box[3]),
    (0, 255, 0),
    2,
)
cv2.imshow("image with bounding box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
