import os
import yaml
from yacs.config import CfgNode as CN

def get_attr(file_name):
    with open(file_name) as f:
        fyaml = yaml.load(f, Loader=yaml.FullLoader)
    
    cfg = CN()
    cfg.device = fyaml["device"]
    cfg.dataset = fyaml["dataset"]
    cfg.lidar = fyaml["lidar_size"]

    cfg.model = CN()
    cfg.model.backbone = fyaml["model"]["backbone"]
    cfg.model.bin_size = fyaml["model"]["bin_size"]

    cfg.train = CN()
    cfg.train.tag = fyaml["train"]["tag"]
    cfg.train.batch_size = fyaml["train"]["batch_size"]
    cfg.train.epochs = fyaml["train"]["epochs"]
    cfg.train.lr = fyaml["train"]["lr"]
    cfg.train.lr_decay = fyaml["train"]["lr_decay"]
    cfg.train.decay_epoch = fyaml["train"]["decay_epoch"]
    cfg.train.weight_decay = fyaml["train"]["weight_decay"]
    cfg.train.alpha = fyaml["train"]["alpha"]
    cfg.train.beta = fyaml["train"]["beta"]

    if cfg.dataset == "kitti":
        model_cfg = {"bin_size": cfg.model.bin_size,
                     "depth_range": [0.001, 100],
                     "backbone": cfg.model.backbone,
                     "feat_size": [7, 36],
                     "lidar_info": cfg.lidar}
    
    elif cfg.dataset == "nyu":
        model_cfg = {"bin_size": cfg.model.bin_size,
                     "depth_range": [0.001, 10],
                     "backbone": cfg.model.backbone,
                     "feat_size": [8, 10],
                     "lidar_info": cfg.lidar}

    return cfg, model_cfg

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_cfg(cfg, model, is_test=False):
    os.system('clear')
    print("*Model info")
    print(f" -Backbone: {cfg.model.backbone}")
    print(f" -Bin size: {cfg.model.bin_size}")
    print(f" -Model Params: {count_parameters(model)/1000000:.2f}M")
    print(f" -LiDAR Channel: {cfg.lidar[0]}")
    print("=========================")
    if not is_test:
        print("*Train info")
        print(f" -Epochs: {cfg.train.epochs}")
        print(f" -Batch Size: {cfg.train.batch_size}")
        print(f" -Adam Optimzier")
        print(f"   -Learning Rate: {cfg.train.lr}")
        print(f"   -LR Decay: {cfg.train.lr_decay}")
        print(f"   -Weight Decay: {cfg.train.weight_decay}")
        print("=========================")