from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from torchvision import transforms

transform = transforms.Compose([
    # you can add other transformations in this list
    #transforms.Grayscale(),
    transforms.Resize((240,240)),
    transforms.ToTensor()
    
])

import json
import pandas as pd
import os
# from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
        # print(self.labels.info())
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #print(self.labels.iloc[idx, 1])
        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 1])
        
        image = Image.open(img_name)
        landmarks = self.labels.iloc[idx, 2:].to_dict()
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        

        if self.transform:
            # print(sample['image'])
            image = self.transform(image)

        sample = {'image': image, 'landmarks': landmarks}
        return sample
    


kenya_dataset = KenyaDataset('/home/erin/Code/Q4/WildlifeData/kenya_labels.csv', '/home/erin/Code/Q4/WildlifeData/kenya_ims', transform=transform)
# print(len(kenya_dataset))
sample = kenya_dataset[3]['image']
# print(kenya_dataset[3]['landmarks'].value())
# plt.imshow(sample)
# plt.show()

transformed_dataset = KenyaDataset(csv_file='/home/erin/Code/Q4/WildlifeData/kenya_labels.csv',
                                           root_dir='/home/erin/Code/Q4/WildlifeData/kenya_ims',
                                           transform=transform
                                           )

dataloader = DataLoader(transformed_dataset, batch_size=1,
                        shuffle=True)
print(next(iter(dataloader)))


# for data in dataloader:
#     images = data['image']
#     labels = data['landmarks']['max_detection_conf']
#     #print(labels)
#     if labels == ['0.0']:
#         print(labels)
#     print(data.keys())
#     print(images)
#     print(labels)
# 



