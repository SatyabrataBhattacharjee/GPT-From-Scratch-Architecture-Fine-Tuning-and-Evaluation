import torch

def perplexity_from_loss(loss):
    """
    Computes perplexity from a scalar cross-entropy loss.
    This is for reporting only.
    """
    return torch.exp(torch.tensor(loss))
