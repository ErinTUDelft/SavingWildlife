import torch
import torch.nn as nn
import torch.optim as optim


# Bounding box regression loss function
class BoundingBoxLoss(nn.Module):
    def __init__(self):
        super(BoundingBoxLoss, self).__init__()

    def forward(self, y_true, y_pred):
        # Extract the bounding box coordinates
        true_x, true_y, true_width, true_height = torch.split(y_true, 1, dim=1)
        pred_x, pred_y, pred_width, pred_height = torch.split(y_pred, 1, dim=1)

        # Calculate the MSE loss
        loss = (
            nn.MSELoss()(true_x, pred_x)
            + nn.MSELoss()(true_y, pred_y)
            + nn.MSELoss()(true_width, pred_width)
            + nn.MSELoss()(true_height, pred_height)
        )

        return loss



# Cross-entropy loss function
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y_true, y_pred):
        loss = nn.CrossEntropyLoss()(y_pred, y_true)
        return loss


# Create the optimizer
def create_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return optimizer
