## ToDO:
"""
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
#from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

from tqdm import tqdm # progress bar

# Import from project
from create_dataloader import dataloader_train, dataloader_test


#from optimizing import create_optimizer, BoundingBoxLoss, CrossEntropyLoss

######### Definitions
classes = 2 # no humans nor vehicles in the dataset
epochs = 10
#########

######### Create the model
# Load the model
model = torchvision.models.mobilenet_v3_small(pretrained=True) #pretrained will become deprecated in the future
# modify the last layer
model.classifier[3] = torch.nn.Linear(in_features=1024, out_features= classes, bias=True)
# freeze all layers 
for param in model.parameters():
    param.requires_grad = False
# unfreeze the last layer
for param in model.classifier[3].parameters():
    param.requires_grad = True
#########

# loss function
criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dataloader from the training and test set 
train_loader = dataloader_train
test_loader = dataloader_test


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


    model.train() # apparently good practice to do
    for data in train_loader:
        #inputs, labels = data
        inputs = data['image']

        #print('image', inputs)
        labels = data['landmarks']['max_detection_conf']

        converted_labels = []

        for label in labels:

            #print('labels', label)
            #print(label.shape())


            if label == '0.0': 
                #labels = torch.tensor([[1, 0],[1, 0] ,[1, 0] ,[1, 0]]) # target of cross-enropy loss should be class index
                labels = torch.tensor([0])
                converted_labels.append(0)
            else:
                #labels = torch.tensor([[1, 0],[1, 0] ,[1, 0] ,[1, 0]]) 
                labels = torch.tensor([1])
                converted_labels.append(1)

        

        labels= torch.LongTensor(converted_labels)
    
        # zero the parameter gradients (VERY important; otherwise gradients accumulate)
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print('new block')
        # print('labels', labels)
        # print('outputs129:', outputs)
        # print('loss130', loss)
        # print('outputs.data', outputs.data)

        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)


        # print('predicted', predicted)
        # print('groundtru', labels)
        total += labels.size(0)
       
        correct += (predicted == labels).sum().item()

        if total%320 == 0:
            print(total)
        
        if total > 15000:
            return avg_loss/len(train_loader), 100 * correct / total

        
        

    return avg_loss/len(train_loader), 100 * correct / total
        
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
            
            inputs = data['image']
            labels = data['landmarks']['max_detection_conf']

            converted_labels = []

            for label in labels:
                if label == '0.0': 
                    #labels = torch.tensor([[1, 0],[1, 0] ,[1, 0] ,[1, 0]]) # target of cross-enropy loss should be class index
                    labels = torch.tensor([0])
                    converted_labels.append(0)
                else:
                    #labels = torch.tensor([[1, 0],[1, 0] ,[1, 0] ,[1, 0]]) 
                    labels = torch.tensor([1])
                    converted_labels.append(1)

            labels= torch.LongTensor(converted_labels)

            # forward pass
            outputs = model(inputs)
           
            loss = criterion(outputs, labels)
     

            # keep track of loss and accuracy
            avg_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()

            if total%160 == 0:
                print(total)

            if total > 3000:
                return avg_loss/len(test_loader), 100 * correct / total


    return avg_loss/len(test_loader), 100 * correct / total

print(train_loader)

# Train the network
j = 0
for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    print('j', j)
    j +=1

    # Train on data
    train_loss, train_acc = train(model, criterion, optimizer, train_loader)
    # Test on data
    test_loss, test_acc = test(test_loader, model, criterion)
    # Print results
    print('Epoch: {}, Train Loss: {}, Train Acc: {}, Test Loss: {}, Test Acc: {}'.format(epoch, train_loss, train_acc, test_loss, test_acc))