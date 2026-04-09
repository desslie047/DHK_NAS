import json
import os

import torch
from dhknas.data import load_data
from dhknas.models import load_model
from dhknas.args_model import get_args
from dhknas.trainer import load_trainer

# args
args = get_args()

kaa_config_path = os.path.join(args.dhconfig, "kaa_config.json")
if os.path.exists(kaa_config_path):
    with open(kaa_config_path, 'r') as f:
        kaa_config = json.load(f)
    args.use_kaa = kaa_config.get("use_kaa", False)
    args.kan_layers = kaa_config.get("kan_layers", 2)
    args.grid_size = kaa_config.get("grid_size", 1)
    args.spline_order = kaa_config.get("spline_order", 1)
else:
    args.use_kaa = False

# dataset
dataset, args = load_data(args)

# model
model = load_model(args, dataset)

# device
model = model.to(args.device)
dataset.to(args.device)

# train
trainer, criterion = load_trainer(args)
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=args.lr, weight_decay=args.wd
)
train_dict = trainer(
    model,
    optimizer,
    criterion,
    dataset,
    args,
    args.max_epochs,
    args.patience,
    disable_progress=False,
    writer=None,
    grad_clip=args.grad_clip,
    device=args.device,
)
print(f"Final Test: {train_dict['test_auc']:.4f}")

# close
from dhknas.trainer import log_train

log_train(args.log_dir, args, train_dict, None)
