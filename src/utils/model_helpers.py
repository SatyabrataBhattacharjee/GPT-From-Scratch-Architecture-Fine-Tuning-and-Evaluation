import json
import numpy as np
import torch

def load_weights_into_gpt(model, gpt2_dir):
    """
    Loads official GPT-2 weights into a custom GPTModel.
    Assumes GPTModel architecture matches GPT-2 config.
    """

    with open(f"{gpt2_dir}/hparams.json") as f:
        hparams = json.load(f)

    # Load TensorFlow checkpoint weights converted to NumPy
    # (You likely already implemented this mapping earlier)
    # Example sketch:
    weights = np.load(f"{gpt2_dir}/model.npz")

    # Map weights → model parameters
    for name, param in model.named_parameters():
        if name in weights:
            param.data.copy_(torch.tensor(weights[name]))

    print(f"✅ GPT-2 weights loaded from {gpt2_dir}")
