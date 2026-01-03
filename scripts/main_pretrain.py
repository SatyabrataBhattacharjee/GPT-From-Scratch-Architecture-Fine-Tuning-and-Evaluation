import torch

from data.download_data import load_raw_text
from data.dataset import create_dataloader_v1
from tokenization.tokenizer import get_tokenizer
from config.model_config import GPT_CONFIG_124M
from models.gpt_model import GPTModel
from training.train import train_model_simple


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load raw data
    text = load_raw_text()

    # 2. Train / validation split
    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # 3. Tokenizer
    tokenizer = get_tokenizer()

    # 4. DataLoaders
    train_loader = create_dataloader_v1(
        train_text,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=True,
        drop_last=True
    )

    val_loader = create_dataloader_v1(
        val_text,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=False,
        drop_last=False
    )

    # 5. Model
    model = GPTModel(GPT_CONFIG_124M).to(device)

    # 6. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        weight_decay=0.1
    )

    # 7. Train
    train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )


if __name__ == "__main__":
    main()
