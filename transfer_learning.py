import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms


# load images from kenya_ims

# adapt image dataset to fit model
transforms = torch.nn.Sequential(
    transforms.Grayscale(),
    transforms.Resize((240, 240)),
)

# load model
mobileNetV2 = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)


# transfer learning class
class TransferLearning(nn.Module):
    def __init__(self, pretrained_model):
        super(TransferLearning, self).__init__()
        self.pretrained_model = pretrained_model  # output features 1000

    def forward(self, x):
        x = self.pretrained_model(x)

        return x


TransferLearning = TransferLearning(mobileNetV2)
# print(TransferLearning)

TransferLearning.eval()
