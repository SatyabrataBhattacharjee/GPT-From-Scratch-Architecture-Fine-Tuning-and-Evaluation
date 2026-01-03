import torch
from training.losses import calc_loss_batch

def calc_loss_loader(loader, model, device, num_batches=None):
    model.eval()
    total = 0
    n = len(loader) if num_batches is None else min(num_batches, len(loader))

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= n:
                break
            total += calc_loss_batch(x, y, model, device).item()

    model.train()
    return total / n
