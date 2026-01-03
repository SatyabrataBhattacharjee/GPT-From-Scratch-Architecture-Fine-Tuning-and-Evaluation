import torch

from tokenization.tokenizer import get_tokenizer
from config.model_config import GPT_CONFIG_124M
from models.gpt_model import GPTModel
from generation.generate import generate_text_simple
from generation.utils import text_to_token_ids, token_ids_to_text


CHECKPOINT_PATH = "models/checkpoints/gpt_pretrained.pth"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Tokenizer (same contract as training)
    tokenizer = get_tokenizer()

    # 2. Model architecture (same config as training)
    model = GPTModel(GPT_CONFIG_124M).to(device)

    # 3. Load pretrained weights
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()  # IMPORTANT: inference mode

    # 4. Prompt
    prompt = "Every effort moves you"

    # 5. Encode prompt
    encoded = text_to_token_ids(prompt, tokenizer).to(device)

    # 6. Generate
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=GPT_CONFIG_124M["context_length"]
        )

    # 7. Decode
    generated_text = token_ids_to_text(token_ids, tokenizer)

    print("\n=== GENERATED TEXT ===")
    print(generated_text)


if __name__ == "__main__":
    main()
