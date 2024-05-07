import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from .data_transform import *
from .velodyne_preprocess import open_lidar

num_workers = 8

class KITTI_dataset(Dataset):
    def __init__(self, datalist_file, transform=None, FOV=120, channel=16):
        super(KITTI_dataset, self).__init__()
        self.FOV = FOV
        self.channel = channel
        self.transform = transform
        self.datalist_file = f"./datasets/kitti/kitti_{datalist_file}.csv"
        self.frame = pd.read_csv(self.datalist_file, header=None)
    
    def __getitem__(self, idx):
        image_name=self.frame.iloc[idx, 0]
        depth_name=self.frame.iloc[idx, 1]
        lidar_name=self.frame.iloc[idx, 2]

        image=Image.open("./datasets/kitti"+image_name)#.resize(default_size)
        depth=Image.open("./datasets/kitti"+depth_name)#.resize(default_size)
        lidar=open_lidar("./datasets/kitti"+lidar_name, self.FOV, self.channel)
        sample={'image': image, 'depth': depth}

        if self.transform:
            sample=self.transform(sample)

        sample["lidar"] = lidar

        return sample
    
    def __len__(self):
        return len(self.frame)

def getTrain_Data(batch_size, FOV, channel):
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
        ToTensor(),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,),
        Normalize(__imagenet_stats["mean"], __imagenet_stats["std"])
    ])

    transform_train = KITTI_dataset(datalist_file="train", transform=pre_process, FOV=FOV, channel=channel)
    
    dataloader_train = DataLoader(dataset=transform_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        
    return dataloader_train

def getTest_Data(batch_size, FOV, channel, crop=True, upsample=False):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]), 
        'eigvec': torch.Tensor([ [-0.5675,  0.7192,  0.4009], 
                                [-0.5808, -0.0045, -0.8140], 
                                [-0.5836, -0.6948,  0.4203] ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    
    pre_process = transforms.Compose([
        Resize(crop=crop, upsample=upsample),
        ToTensor(),
        Normalize(__imagenet_stats["mean"], __imagenet_stats["std"]),
    ])

    transform_train = KITTI_dataset(datalist_file="test", transform=pre_process, FOV=FOV, channel=channel)
    
    dataloader_train = DataLoader(dataset=transform_train, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
    return dataloader_train
