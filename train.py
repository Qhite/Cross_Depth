import torch
import models
import dataloader_KITTI as dataloader
import tools
from tqdm import tqdm
import pandas as pd
import wandb

torch.manual_seed(13)
torch.cuda.manual_seed(13)

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True

# Get Config File Path
args = tools.get_config()
config_yaml = args.config

# Load Config File
cfg, m_cfg = tools.get_attr(config_yaml) # Load Train config

# wandb init
wandb.init(
    project="kitti",
    config={
        "batch_size": cfg.train.batch_size,
        "learning_rate": cfg.train.lr,
        "epochs": cfg.train.epochs,
        "weight_decay": cfg.train.weight_decay,
    }
)
wandb.run.name = cfg.train.tag
wandb.run.save()

# Model Load & Initialization
model = models.DepthNet(m_cfg).to(device=cfg.device)

# Dataload
test_loader = dataloader.getTest_Data(batch_size=1, FOV=120, channel=cfg.lidar[0])
train_loader = dataloader.getTrain_Data(batch_size=cfg.train.batch_size, FOV=120, channel=cfg.lidar[0])

# Optimizer
enc = list(map(id, model.encoder.parameters()))
params = filter(lambda p: id(p) not in enc, model.parameters())

optimizer = torch.optim.Adam([{"params": model.encoder.parameters(), "lr": cfg.train.lr * 0.1},
                             {"params": params}],
                             lr=cfg.train.lr,
                             weight_decay=cfg.train.weight_decay
                             )

# Loss Funtions
Loss = models.Losses(m_cfg, cfg).to(device=cfg.device)

# Show Configs
tools.show_cfg(cfg, model)

Epochs = cfg.train.epochs

#Train
def train(model):
    for epoch in range(Epochs):
        loss_sum = torch.zeros(1)

        model.train()
        tools.update_lr(optimizer, epoch, cfg)

        train_tqdm = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, batch in train_tqdm:
            tools.to_device(batch, cfg.device)

            optimizer.zero_grad()

            output, centers = model(batch)

            loss = Loss(output, centers, batch)
            loss.backward()

            optimizer.step()

            loss_sum += loss.clone().detach().cpu()

            train_tqdm.set_description(f"Epoch {epoch+1:2d}/{Epochs:2d} | Loss {float(loss_sum/i):.3f}")

        model.eval()
        avg_l, avg_e = valiate(epoch, model)
        wandb.log({
            'train_loss': float(loss_sum/i),
            'validate_loss': float(avg_l),
            'delta1': float(avg_e[0]),
            'delta2': float(avg_e[1]),
            'delta3': float(avg_e[2]),
            'rms': float(avg_e[3]),
            'log_rms': float(avg_e[4]),
            'rel': float(avg_e[5]),
            'sqrel': float(avg_e[6]),})
        
    tools.save_model(args, model)

@torch.no_grad()
def valiate(epoch, model):
    val_tqdm = tqdm(enumerate(test_loader), total=len(test_loader))
    m_sum = torch.zeros(7)
    loss_sum = torch.zeros(1)

    for i, batch in val_tqdm:
        tools.to_device(batch, cfg.device)

        predict, centers = model(batch)
        t = batch["depth"].clone()
        p, c = predict.clone(), centers.clone()

        metrics = tools.cal_metric(p, t, m_cfg)
        m_sum += metrics

        loss = Loss(p, c.unsqueeze(0), batch)
        loss_sum += loss.cpu()

        val_tqdm.set_description(f"Delta_1 {float(m_sum[0]/i):.3f} | RMS {float(m_sum[3]/i):.3f} | REL {float(m_sum[5]/i):.3f} | loss {float(loss_sum/i):.3f}")
    
    avg_loss = (loss_sum/len(test_loader))
    avg_errors = (m_sum/len(test_loader)).tolist()

    tools.show_metric(avg_errors)
    print("==" * 50)

    return avg_loss, avg_errors

if __name__ == "__main__":
    train(model)