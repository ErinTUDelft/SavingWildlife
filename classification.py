## ToDO:
"""
- fix labels
- findng out required data type (rgb or greyscale)
- need to normalize the input
"""

## Future Improvements
"""
- splitting the dataset 
- finetuning on top layers 
- using bounding box loss
- adding data augmentation (flipping, noise)
- cropping the watermark
- adding regularization
"""

## Questions
"""
- is the watermark a problem?
- is using the bounding bos for classification smart?   
- is the last linear layer indeed the right thing to do?
- how to fix the crossenthropy loss?
"""
## Import modules
import torch
import torch.optim as optim
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

# from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

from tqdm import tqdm  # progress bar

# Import from project
from create_data_loader import dataloader

# from optimizing import create_optimizer, BoundingBoxLoss, CrossEntropyLoss

######### Definitions
classes = 2  # no humans nor vehicles in the dataset
epochs = 10
#########

######### Create the model
# Load the model
model = torchvision.models.mobilenet_v3_small(
    pretrained=True
)  # pretrained will become deprecated in the future
# modify the last layer
model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=classes, bias=True)
# freeze all layers
for param in model.parameters():
    param.requires_grad = False
# unfreeze the last layer
for param in model.classifier[3].parameters():
    param.requires_grad = True
#########

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dataloader from the training and test set
train_loader = dataloader
test_loader = []

loss = criterion(torch.tensor([1.0]), torch.tensor([1.0]))
print('loss58', loss)

######### Training


# Y_pred_good = torch.tensor[2.1, 1.0, 2.1]
# print(model)
# labels = torch.tensor([1,0,1,0])
# Y = torch.tensor([2,0,1])


# Y = torch.tensor([2,0,1,1])
# print('Y', Y)
# print('Y.size', Y.size())
# Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [0.1, 1.0, 2.1], [0.1, 1.0, 2.1], [0.1, 1.0, 2.1]])
# #Y_pred_good = torch.tensor([[2.1, 1.0, 2.1], [1.1, 1.0, 2.1], [2.1, 1.0, 2.1]])
# #[0.1, 1.0, 2.1]]
# print(Y_pred_good)
# print('shape', Y_pred_good.size())

# l1 = criterion(Y_pred_good, Y)

# print('l1', l1)


def train(model, criterion, optimizer, train_loader):
    """
    Trains network for one epoch in batches.
    Args:
        train_loader: Data loader for training set.
        model: MobileNetV3
        optimizer: Adam
        criterion: CrossEntropyLoss
    """
    avg_loss = 0
    correct = 0
    total = 0

    # for batch, data in enumerate(train_loader):
    # print(train_loader)
    model.train()
    for data in train_loader:
        # inputs, labels = data
        inputs = data["image"]
        labels = data["landmarks"]["max_detection_conf"]

        print("labels", labels)

        if labels == ["0.0"]:
            # labels = torch.tensor([[1, 0],[1, 0] ,[1, 0] ,[1, 0]]) # target of cross-enropy loss should be class index
            labels = torch.tensor([1, 0])
        else:
            # labels = torch.tensor([[1, 0],[1, 0] ,[1, 0] ,[1, 0]])
            labels = torch.tensor([1, 0, 1, 0])

            # labels is random tensor with batch size 4

        print("labels", labels)

        print("shape", labels.size())
        # unsqueeze the labels
        labels1 = labels  # .unsqueeze(0)
        print("labels", labels1)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        print("outputs", outputs)
        print(outputs.size())

        loss = criterion(outputs, labels1)
        loss.backward()
        optimizer.step()

        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        total = 2
        # correct += (predicted == labels).sum().item()
        correct = 1

    return avg_loss / len(train_loader), 100 * correct / total


######### Testing
def test(test_loader, model, criterion):
    """
    Evaluates network in batches
    Args:
        test_loader: Data loader for test set.
        model: MobileNetV3
        criterion: CrossEntropyLoss
    """
    avg_loss = 0
    correct = 0
    total = 0

    model.eval()
    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]

            inputs = data["image"]
            labels = data["landmarks"]["max_detection_conf"]

            # forward pass
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            labels2 = np.array([1, 2])
            # keep track of loss and accuracy
            avg_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels2.size(0)

            correct += (predicted == labels).sum().item()

    return avg_loss / len(test_loader), 100 * correct / total


print(train_loader)

# Train the network
for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    # Train on data
    train_loss, train_acc = train(model, criterion, optimizer, train_loader)

    # Test on data
    # test_loss, test_acc = test(test_loader, model, criterion)

    # Print training results
    print(
        "Epoch: {}, Train Loss: {}, Train Acc: {}".format(epoch, train_loss, train_acc)
    )
    # Print results
    # print('Epoch: {}, Train Loss: {}, Train Acc: {}, Test Loss: {}, Test Acc: {}'.format(epoch, train_loss, train_acc, test_loss, test_acc))
