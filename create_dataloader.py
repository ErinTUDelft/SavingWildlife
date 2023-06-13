import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from torchvision import transforms

# use transform to make samples consistent
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Grayscale(),
    transforms.Resize((240,240)),
    transforms.ToTensor()
    
])

# create class to pass to dataloader
class KenyaDataset(Dataset):
    '''Dataset with wildlife images from the iWildLife competition.
    '''

    def __init__(self, csv_file, root_dir, transform = None) -> None:
        super(Dataset, self).__init__()
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        print(self.labels.iloc[idx, 1])
        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 1])
        
        image = Image.open(img_name)
        landmarks = self.labels.iloc[idx, 2:].to_dict()
 

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'landmarks': landmarks}
        return sample
    


# kenya_dataset = KenyaDataset('filtered_kenya_labels.csv', 'kenya_ims', transform=transform)
# print(kenya_dataset.info())

transformed_dataset = KenyaDataset(csv_file='filtered_kenya_labels.csv',
                                           root_dir='kenya_ims',
                                           transform=transform
                                           )

# create the dataloader
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True)
