import torch
import torch.nn as nn
import dataloader_KITTI as dataloaderK
import models
import tools
from tqdm import tqdm
from flops_profiler.profiler import get_model_profile

class model_banch(nn.Module):
    def __init__(self, model):
        super(model_banch, self).__init__()
        self.model = model
    
    def forward(self, x):
        data = {"image":torch.rand([1,3,215,1136]).to(cfg.device), 
                "lidar":torch.rand([1]+cfg.lidar).to(cfg.device)}
        out = self.model(data)

args = tools.get_config()
cfg, m_cfg = tools.get_attr(args.config)

TrDL = dataloaderK.getTrain_Data(batch_size=3, FOV=120, channel=16)
TeDL = dataloaderK.getTest_Data(batch_size=1, FOV=120, channel=16)

model = models.DepthNet(m_cfg).to(cfg["device"])
model.eval()

tools.show_cfg(cfg, model)

with torch.no_grad():
    model_b = model_banch(model)
    flops, macs, params = get_model_profile(model=model_b, input_shape=(1,1), print_profile=False)
    print(f"FLOPs: {flops/1000000000:.2f}G")
    print(f" MACs: {macs/1000000000:.2f}G")

data = {"image":torch.rand([1,3,215,1136]).to(cfg.device), 
        "lidar":torch.rand([1]+cfg.lidar).to(cfg.device)}

o,c = model(data)
# print(o.size())

for b in tqdm(TrDL, total=len(TrDL)):
    # print(b["image"].size())
    # print(b["depth"].size())
    pass
