import torch
import models
import dataloader_KITTI as dataloader
import tools
import os
from tqdm import tqdm

# Get Trained File Path
args = tools.get_config()
log_file = args.config
ld = os.listdir(log_file)
config_yaml = [f for f in ld if f.endswith(".yaml")][0]
weight = [f for f in ld if f.endswith(".pth.tar")][0]

# Load Config File
cfg, m_cfg = tools.get_attr(f"{log_file}/{config_yaml}") # Load Test Model

# Model Load & Initialization
model = models.DepthNet(m_cfg).to(device=cfg.device)
model.load_state_dict(torch.load(f"{log_file}/{weight}"))
model.eval()

# Dataload
test_loader = dataloader.getTest_Data(batch_size=1, FOV=120, channel=cfg.lidar[0], crop=False, upsample=True)

tools.show_cfg(cfg, model, True)

# Loss Funtions
Loss = models.Losses(m_cfg, cfg).to(device=cfg.device)

if __name__ == "__main__":
    m_sum = torch.zeros(7)
    loss_sum = torch.zeros(1)

    for i, batch in enumerate(tqdm(test_loader, total=len(test_loader), desc="Loss & Errors")):
        with torch.no_grad():
            tools.to_device(batch, cfg.device)
            p, c = model(batch)
            t = batch["depth"]

            metrics = tools.cal_metric(p, t, m_cfg)
            m_sum += metrics

            loss = Loss(p, c.unsqueeze(0), batch)
            loss_sum += loss.cpu()

    avg_loss = (loss_sum/len(test_loader))
    avg_errors = (m_sum/len(test_loader)).tolist()
    print(f"Loss: {avg_loss}")
    tools.show_metric(avg_errors)

    path = f"/root/output/{cfg.train.tag}-kitti-eval"
    tools.visualization(test_loader, model, path, cfg.device, cfg.dataset)
    os.system(f"cp -r {log_file} {path}")
