import argparse
import torch

from tokenization.tokenizer import get_tokenizer
from generation_backends.registry import get_generation_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="local",
        choices=["local", "gpt2-124m", "gpt2-355m"],
        help="Model source for generation"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Every effort moves you"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()

    backend = get_generation_backend(
        source=args.source,
        tokenizer=tokenizer,
        device=device
    )

    output = backend.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens
    )

    print("\n=== GENERATED TEXT ===")
    print(output)


if __name__ == "__main__":
    main()
