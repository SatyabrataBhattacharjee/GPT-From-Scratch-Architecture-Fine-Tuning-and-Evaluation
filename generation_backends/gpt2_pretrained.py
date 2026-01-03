import torch

from models.gpt_model import GPTModel
from generation.generate import generate_text_simple
from generation.utils import text_to_token_ids, token_ids_to_text
from tokenization.tokenizer import get_tokenizer
from utils.model_helpers import load_weights_into_gpt  


class GPT2PretrainedBackend:
    def __init__(self, model_size, weights_path, device):
        self.device = device
        self.tokenizer = get_tokenizer()

        if model_size == "124m":
            cfg = {
                "vocab_size": 50257,
                "context_length": 1024,
                "emb_dim": 768,
                "n_heads": 12,
                "n_layers": 12,
                "drop_rate": 0.0,
                "qkv_bias": True
            }
        elif model_size == "355m":
            cfg = {
                "vocab_size": 50257,
                "context_length": 1024,
                "emb_dim": 1024,
                "n_heads": 16,
                "n_layers": 24,
                "drop_rate": 0.0,
                "qkv_bias": True
            }
        else:
            raise ValueError("Unsupported GPT-2 size")

        self.model = GPTModel(cfg).to(device)
        load_weights_into_gpt(self.model, weights_path)
        self.model.eval()

    def generate(self, prompt, max_new_tokens=50):
        encoded = text_to_token_ids(prompt, self.tokenizer).to(self.device)

        with torch.no_grad():
            token_ids = generate_text_simple(
                model=self.model,
                idx=encoded,
                max_new_tokens=max_new_tokens,
                context_size=self.model.pos_emb.weight.shape[0]
            )

        return token_ids_to_text(token_ids, self.tokenizer)
