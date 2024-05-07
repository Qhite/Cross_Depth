import torch.nn as nn
import torch.nn.functional as F
import models

class DepthNet(nn.Module):
    def __init__(self, cfg):
        super(DepthNet, self).__init__()
        self.bin_size = cfg["bin_size"]
        self.depth_range = cfg["depth_range"]

        # Build Model
        self.model, self.info = models.get_model(cfg["backbone"])
        self.encoder = self.model.get_submodule("features")
        self.Atten = models.Attention_block(cfg["feat_size"], cfg["lidar_info"], self.info[1][0])
        self.decoder = models.Decoder(self.info[1])

        # Output
        self.bin_out = nn.Sequential(
            nn.Linear(128, self.bin_size),
            nn.ReLU()
        )
        self.Conv_out = nn.Sequential(
            nn.Conv2d(16, self.bin_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
        )
    
    def get_features(self, x):
        features = []
        for name, module in self.encoder.named_children():
            x = module(x)
            if int(name) in self.info[0]:
                features.append(x)
                if int(name) == self.info[0][-1]:
                    break
        return x, features

    def forward(self, data):
        x, y = data["image"], data["lidar"]

        # Encoder
        _, f = self.get_features(x)

        # Attention block
        f[-1], bins = self.Atten({"feat":f[-1], "lidar":y})
        
        # Get Bins
        depth_scale = (self.depth_range[1] - self.depth_range[0])
        bins = self.bin_out(bins)
        bins = bins / bins.sum(axis=1, keepdim=True)
        bins = depth_scale * bins + self.depth_range[0]

        # Bin Centers
        bin_width = F.pad(bins, (1,0), mode="constant", value=1e-3)
        bin_edge = bin_width.cumsum(dim=1)
        centers = 0.5 * (bin_edge[:, :-1]+bin_edge[:, 1:])
        centers = centers.unsqueeze(2).unsqueeze(2)

        # Decoder
        out = self.decoder(f)
        out = self.Conv_out(out)
        out = F.softmax(out, dim=1) # Depth bin-probability map

        # Depth map
        predict = (out * centers).sum(axis=1, keepdim=True)

        return predict, centers.squeeze()
