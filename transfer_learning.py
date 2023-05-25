import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms


# load images from kenya_ims
path_to_image = "./kenya_ims/8a0a6ef6-21bc-11ea-a13a-137349068a90.jpg/"

# adapt image dataset to fit model
transforms = torch.nn.Sequential(
    transforms.Grayscale(),
    transforms.Resize((240, 240)),
)

# load model
mobileNetV2 = mobilenet_v2(pretrained=True)
print(mobileNetV2)


# transfer learning class
# class TransferLearning(nn.Module):

#     def __init__(self,pretrained_model):
#         super(TransferLearning, self).__init__()
#         self.pretrained_model = pretrained_model
#         self.fc = nn.Linear(1000, 14)

#     def forward(self, x):
#         x = self.pretrained_model(x)
#         x = self.fc(x)
#         return x


# TransferLearning = TransferLearning(mobileNetV2)
# TransferLearning.eval()
