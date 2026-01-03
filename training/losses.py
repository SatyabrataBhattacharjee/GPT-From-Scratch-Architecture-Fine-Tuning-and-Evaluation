import torch.nn.functional as F

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
