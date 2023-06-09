import torch.nn as nn


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
            nn.Linear(in_features=1024, out_features=2, bias=True),
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
