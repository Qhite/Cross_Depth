import torch
import torch.nn.functional as F
import numpy as np

def get_points(f_path):
    raw = np.fromfile(f_path, dtype=np.float32).reshape(-1,4)
    refl_mask = raw[:,3] > 0
    points = raw[refl_mask]
    return torch.tensor(points)

def ahead_point_masking(points, range):
    head = torch.tensor([1.0, 0.0])
    angle = torch.rad2deg(torch.atan2(points[:,1], points[:,0]))
    mask1 = angle > -range/2
    mask2 = angle < range/2
    mask = torch.logical_and(mask1, mask2)
    angle = angle_step(angle, range)

    return torch.cat((points[mask], angle[mask].unsqueeze(1)), dim=1)

def angle_step(angles, FOV):
    vmin = -FOV/2
    angles = vmin + torch.round( (angles - vmin) / 0.16 ) * 0.16
    return angles

def split_channel(points, FOV):
    dangle = points[:-1, 4] - points[1:, 4]
    ch_idx = torch.nonzero( torch.gt(dangle, FOV//2) ).squeeze()
    return torch.tensor_split(points, ch_idx+1)[1:-1]

def to_tensor(point_channels, FOV):
    step = int(FOV/0.16)
    buff = torch.tensor([])
    for p in point_channels:
        p = torch.flip(p, [0]) # CCW LiDAR Raw -> CW arrange
        p = p.transpose(1, 0)
        p = F.interpolate(p.unsqueeze(0), (step)).squeeze()
        p = p.transpose(1, 0)
        dist = torch.norm(p[:,:3], dim=1)
        buff = torch.cat( (buff, dist.unsqueeze(0)) )
        
    return buff

def open_lidar(bin_path, FOV, set_channel):
    points = get_points(bin_path)
    points = ahead_point_masking(points, FOV)
    points_ch = split_channel(points, FOV)
    points = to_tensor(points_ch, FOV)
    stride = points.size(0) // set_channel
    pad = points.size(0) % set_channel
    
    if pad == 0:
        if stride == 0:
            print(points.size(0), set_channel, stride, pad)
            exit()
        return points[ : : stride]
    else:
        return points[ pad//2 : -pad//2 : stride]
