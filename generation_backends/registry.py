from generation_backends.local_checkpoint import LocalCheckpointBackend
from generation_backends.gpt2_pretrained import GPT2PretrainedBackend


def get_generation_backend(
    source,
    tokenizer,
    device,
    checkpoint_path="models/checkpoints/gpt_pretrained.pth"
):
    if source == "local":
        return LocalCheckpointBackend(
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            device=device
        )

    elif source == "gpt2-124m":
        return GPT2PretrainedBackend(
            model_size="124m",
            weights_path="models/pretrained/gpt2/124M",
            device=device
        )

    elif source == "gpt2-355m":
        return GPT2PretrainedBackend(
            model_size="355m",
            weights_path="models/pretrained/gpt2/355M",
            device=device
        )

    else:
        raise ValueError(f"Unknown generation source: {source}")
