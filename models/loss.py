import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.loss import chamfer_distance

# Loss functions used in AdaBins paper

class Losses(nn.Module):
    def __init__(self, m_cfg, cfg):
        super(Losses, self).__init__()
        self.SILL = SILogLoss(m_cfg)
        self.BCL1 = BinsChamferLoss(m_cfg)
        self.BCL2 = BinsChamferLoss(m_cfg)
        self.alpha = cfg.train.alpha
        self.beta1 = 0.1
        self.beta2 = cfg.train.beta
    
    def forward(self, output, centers, d_dict):
        target = d_dict["depth"]

        if output.shape[-2:] != target.shape[-2:]:
            output = F.interpolate(output, target.shape[-2:], mode="nearest")

        SIL_loss = self.SILL(output, target)
        BC1_loss = self.BCL1(centers, target)
        BC2_loss = self.BCL2(centers, d_dict["lidar"])

        loss = self.alpha * SIL_loss + self.beta1 * BC1_loss + self.beta2 * BC2_loss

        return loss

class SILogLoss(nn.Module):  
    def __init__(self, m_cfg=None, lamb=0.85):
        super(SILogLoss, self).__init__()
        self.name = 'SILogLoss'
        self.lamb = lamb
        self.d_min = m_cfg["depth_range"][0]

    def forward(self, output, target):
        mask_pr = output.ge(self.d_min)
        mask_gt = target.ge(self.d_min)
        mask = torch.logical_and(mask_gt, mask_pr)

        masked_output = (output*mask).flatten(1)
        masked_target = (target*mask).flatten(1)

        g = torch.log(masked_output + 1e-3) - torch.log(masked_target + 1e-3)

        Dg = torch.var(g) + (1 - self.lamb) * torch.pow(torch.mean(g), 2)
        losses = torch.sqrt(Dg)

        return losses

class BinsChamferLoss(nn.Module):
    def __init__(self, m_cfg):
        super(BinsChamferLoss, self).__init__()
        self.name = "ChamferLoss"
        self.d_min = m_cfg["depth_range"][0]
    
    def forward(self, bin_centers, target):
        if len(bin_centers.shape) == 1:
            bin_centers = bin_centers.unsqueeze(0).unsqueeze(2)
        else:
            bin_centers = bin_centers.unsqueeze(2)

        target_points = target.flatten(1).float()

        mask = target_points.ge(self.d_min)
        target_points = [p[m] for p, m in zip(target_points, mask)]

        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target.device)

        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)

        loss, _ = chamfer_distance(x=bin_centers, y=target_points, y_lengths=target_lengths)

        return loss