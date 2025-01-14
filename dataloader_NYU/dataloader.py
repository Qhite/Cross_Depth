import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from dataloader_NYU.data_transform import *

num_workers = 8

# Dataset

class NYUDv2_dataset(Dataset):
    def __init__(self, datalist_file, transform=None):
        super(NYUDv2_dataset, self).__init__()
        self.transform = transform
        self.datalist_file = f"./datasets/data/nyu2_{datalist_file}.csv"
        self.frame = pd.read_csv(self.datalist_file, header=None)
        
    def __getitem__(self, idx):
        image_name=self.frame.iloc[idx, 0]
        depth_name=self.frame.iloc[idx, 1]

        image=Image.open("./datasets/"+image_name)
        depth=Image.open("./datasets/"+depth_name)
        sample={'image': image, 'depth': depth}

        if self.transform:
            sample=self.transform(sample)
        return sample

    def __len__(self):
        return len(self.frame)

# DataLoader

def getTrain_Data(batch_size, lidar_size):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]), 
        'eigvec': torch.Tensor([ [-0.5675,  0.7192,  0.4009], 
                                [-0.5808, -0.0045, -0.8140], 
                                [-0.5836, -0.6948,  0.4203] ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    
    pre_process = transforms.Compose([
        Resize(),
        RandomHorizontalFlip(),
        CenterCrop(),
        ToTensor(),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,),
        Normalize(__imagenet_stats["mean"], __imagenet_stats["std"]),
    ])

    transform_train = NYUDv2_dataset(datalist_file="train", transform=pre_process)
    
    dataloader_train = DataLoader(dataset=transform_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        
    return dataloader_train

def getTest_Data(batch_size, lidar_size):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    
    pre_process = transforms.Compose([
        Resize(),
        CenterCrop(),
        ToTensor(is_test=True),
        Normalize(__imagenet_stats["mean"], __imagenet_stats["std"]),
    ])

    transform_test = NYUDv2_dataset(datalist_file="test", transform=pre_process)
    
    dataloader_test = DataLoader(dataset=transform_test, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return dataloader_test