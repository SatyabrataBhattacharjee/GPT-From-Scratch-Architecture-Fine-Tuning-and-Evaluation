import torch

from models.gpt_model import GPTModel
from config.model_config import GPT_CONFIG_124M
from generation.generate import generate_text_simple
from generation.utils import text_to_token_ids, token_ids_to_text


class LocalCheckpointBackend:
    def __init__(self, checkpoint_path, tokenizer, device):
        self.device = device
        self.tokenizer = tokenizer

        self.model = GPTModel(GPT_CONFIG_124M).to(device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

    def generate(self, prompt, max_new_tokens=50):
        encoded = text_to_token_ids(prompt, self.tokenizer).to(self.device)

        with torch.no_grad():
            token_ids = generate_text_simple(
                model=self.model,
                idx=encoded,
                max_new_tokens=max_new_tokens,
                context_size=GPT_CONFIG_124M["context_length"]
            )

        return token_ids_to_text(token_ids, self.tokenizer)
