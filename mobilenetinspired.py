import torch
import torchvision.transforms as transforms
import torch.nn as nn

# https://cms.tinyml.org/wp-content/uploads/summit2022/tinyML_Talks_Shawn_Hymel_220405.pdf slide 44
# set up transform to use in MobileNetV2
transforms = torch.nn.Sequential(
    transforms.Grayscale(),
    transforms.Resize((240,240)),
)

# set up bottleneck residual blocks
class Bottleneck(nn.Module):


    def __init__(self, inplanes, width, planes, stride):
        super(Bottleneck,self).__init__()

        self.conv1 = nn.Conv2d(inplanes,width, kernel_size = 1)
        self.norm1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width,width, kernel_size = 3, stride=stride)
        self.norm2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes, kernel_size = 1)
        self.norm3 = nn.BatchNorm2d(width)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        out += identity
        out = self.relu(out)

        return out


class LightNet(nn.Module):
    '''
    Lightweight network for object detection on raspberry pi zero
    '''
    def __init__(self):
        super(LightNet, self).__init__()

        self.conv1 = nn.Conv2d(1,3,3,stride=2, padding = 0)
        self.conv2 = nn.Conv2d(3,5,3,stride=2, padding = 0)
        self.conv3 = nn.Conv2d(5,5,3,stride=2, padding = 0)
        self.conv4 = nn.Conv2d(5,8,3,stride=2, padding = 0)

        self.bottle1 = Bottleneck(8,2,8,2)
        self.bottle2 = Bottleneck(8,2,8,2)
        self.bottle3 = Bottleneck(8,2,8,2)
        self.bottle4 = Bottleneck(8,2,8,2)
        self.bottle5 = Bottleneck(8,2,8,2)
        self.bottle6 = Bottleneck(8,2,8,2)
        self.bottle7 = Bottleneck(8,2,5,2)

        self.conv5 = nn.Conv2d(5,1,3,stride=2, padding = 0)

        self.fc = nn.Linear(400,100)
        self.sm = nn.Softmax()

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        
        out = self.bottle1(out)
        out = self.bottle2(out)
        out = self.bottle3(out)
        out = self.bottle4(out)
        out = self.bottle5(out)
        out = self.bottle6(out)
        out = self.bottle7(out)

        out = self.conv5(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.sm(out)
        

        return out
    





