from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

import json
import pandas as pd
import os
# from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# data = json.load(open('iwildcam2022_mdv4_detections.json'))
# bounding_frame = pd.DataFrame(data['images'])

# print(df.head())
# print(bounding_frame.info())

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
        landmarks = self.labels.iloc[idx, 2:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

kenya_dataset = KenyaDataset('kenya_labels.csv', 'kenya_ims')
print(len(kenya_dataset))
sample = kenya_dataset[2]['image']
plt.imshow(sample)
plt.show()

transformed_dataset = KenyaDataset(csv_file='kenya_labels.csv',
                                           root_dir='kenya_ims',
                                           transform=None
                                           )
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)