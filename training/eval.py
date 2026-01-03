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
from training.metrics import perplexity

def evaluate_model(loader, model, device, num_batches=None):
    """
    Computes loss and perplexity for a given dataset.
    """
    loss = calc_loss_loader(loader, model, device, num_batches)
    ppl = perplexity(loss)
    return loss, ppl
