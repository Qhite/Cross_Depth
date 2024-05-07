import torch
import torch.nn.functional as F
import os
import argparse
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm
import math

def get_config():
    parser = argparse.ArgumentParser(description="Get Config yaml File")
    parser.add_argument("--config", required=True)
    return parser.parse_args()

def to_device(data_dict, device):
    for k in data_dict.keys():
        data_dict[k] = data_dict[k].to(device=device)

def update_lr(optim, epoch, cfg):
    if epoch in cfg.train.decay_epoch:
        for param_group in optim.param_groups:
            param_group['lr'] *= cfg.train.lr_decay

def cal_metric(predict, target, m_cfg):
    if predict.shape[-2:] != target.shape[-2:]:
        predict = F.interpolate(predict, target.shape[-2:], mode="nearest")
    mask = torch.gt(target, m_cfg["depth_range"][0])
    p = predict[mask]
    t = target[mask]
    
    diff = torch.abs(p - t)
    ratio = torch.max(p / t, t / p)

    delta1 = torch.sum( ratio < 1.25 ) / p.size(0) # Threshold Accuarcy 1.25
    delta2 = torch.sum( ratio < 1.25**2 ) / p.size(0) # Threshold Accuarcy 1.25^2
    delta3 = torch.sum( ratio < 1.25**3 ) / p.size(0) # Threshold Accuarcy 1.25^3
    RMS = torch.sqrt(torch.pow(diff, 2).mean()) # Root Mean Square Error
    Log = (torch.abs( torch.log10(p+1e-3) - torch.log10(t+1e-3) )).mean() # Averager log10 Error
    Rel = (diff / t).mean() # Relative Error
    SqRel = torch.sqrt(Rel) # Squared Relative Error

    return torch.tensor([delta1, delta2, delta3, RMS, Log, Rel, SqRel])

def show_metric(metrics):
    print(f"Delta_1: {metrics[0]:.3f} | Delta_2: {metrics[1]:.3f} | Delta_3: {metrics[2]:.3f} | RMS: {metrics[3]:.3f}| Log: {metrics[4]:.3f} | Rel: {metrics[5]:.3f} | SqRel: {metrics[6]:.3f}")

def save_model(args, model):
    name = str(int(time()))
    file_n = f"./log/{name}"
    os.system(f"mkdir -p {file_n}")
    os.system(f"cp {args.config} ./log/{name}")
    torch.save(model.state_dict(), f"./log/{name}/{name}.pth.tar")

def lin_interp(shape, xyd):
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

@torch.no_grad()
def visualization(data_loader, model, path, device, data_set, lidar):
    os.system(f"mkdir -p {path}/img")
    os.system(f"mkdir -p {path}/depth")
    os.system(f"mkdir -p {path}/predict")

    if data_set == "kitti":
        datalist_file = "./datasets/kitti/kitti_test.csv"
        d_range = [0.001, 100]
    elif data_set == "nyu":
        datalist_file = "./datasets/data/nyu2_test.csv"
        d_range = [0.001, 10]

    frame = pd.read_csv(datalist_file, header=None)
    psu_lidar = pesudo_lidar(lidar)

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Output Images"):
        batch = psu_lidar(batch)
        to_device(batch, device)
        p, c = model(batch)
        d = batch["depth"]
        image_name=frame.iloc[i, 0]

        if data_set == "kitti":
            os.system(f"cp ./datasets/kitti{image_name} {path}/img/{i}_img.png")
        elif data_set == "nyu":
            os.system(f"cp ./datasets/{image_name} {path}/img/{i}_img.png")
        
        # if d.shape[-2] != p.shape[-2]:
        #     p = F.interpolate(p, d.shape[-2:], mode="nearest")
        
        plt.imsave(f"{path}/predict/{i}_pr.png", p.cpu().squeeze(), cmap="magma_r", vmin=d_range[0], vmax=d_range[1])
        plt.imsave(f"{path}/depth/{i}_gt.png", d.cpu().squeeze(), cmap="magma_r", vmin=d_range[0], vmax=d_range[1])
        
class pesudo_lidar(object):
    def __init__(self, l_size=[]):
        self.beam = l_size[0]
        self.points = l_size[1]
        self.focal_l = 518.8579/2
        self.im_size = [228, 304]

        if self.beam == 1:
            self.beam_angle = [5]
            self.ch_idx = [0,258]
        elif self.beam == 4:
            self.beam_angle = [-15, -5, 5, 15]
            self.ch_idx = [0,284,266,258,280]
        elif self.beam == 8:
            self.beam_angle = [-15, -10, -5, -1.67, 1.67, 5, 10, 15]
            self.ch_idx = [0,284,271,266,280,268,258,262,280]
        
        self.mask = self.masking()
    
    def __call__(self, d_dict):
        depth = d_dict["depth"]
        batch_size = depth.size(0)
        lidar = torch.zeros([batch_size, sum(self.ch_idx)])

        for i in range(batch_size):
            lidar[i] = depth[i, self.mask.unsqueeze(0)]
        
        lidar = self.by_channel(lidar)

        d_dict["lidar"] = lidar

        return d_dict

    def masking(self):
        H, W = self.im_size[0], self.im_size[1]
        mask = torch.zeros(self.im_size, dtype=torch.bool)
        margin = 0.18
        
        for angle in self.beam_angle:
            for u in range(W):
                for v in range(H):
                    x = (u-W//2)/self.focal_l
                    y = (v-H//2)/self.focal_l
                    if (math.atan(y/math.sqrt(x**2+1)) < math.pi/180*angle) * (math.atan(y/math.sqrt(x**2+1)) > math.pi/180*(angle-margin)):
                        mask[v,u] = True
        return mask
    
    def by_channel(self, pl):
        batch_size = pl.size(0)
        idx = torch.cumsum(torch.tensor(self.ch_idx), 0).tolist()
        pl_ch = torch.zeros([batch_size, self.beam, self.points])

        for i in range(batch_size):
            pl_b = pl[i]
            pl_c = torch.ones([self.beam, self.points])
            for j in range(self.beam):
                p = pl_b[idx[j]:idx[j+1]]
                step = len(p) // self.points
                pad = len(p) % self.points
                pl_c[j] = p[ pad//2 : -pad//2 : step ]

            pl_ch[i] = pl_c
        
        return pl_ch