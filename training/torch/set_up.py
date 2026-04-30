import torch.nn as nn
import torch.optim as optim
from models.dl_models import EarlyStopper
import torch.optim.lr_scheduler as lr_scheduler


def set_up_training_components(model, lr, class_weight, patience=5, min_delta=0.001):
    if class_weight.dim() > 0:
        class_weight = class_weight[0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight, reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.01)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    return criterion, optimizer, scheduler, early_stopper
