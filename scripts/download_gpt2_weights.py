import os
import sys
import subprocess

def download_gpt2(model_size, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading GPT-2 {model_size} weights into {target_dir}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gpt_2_simple.download_gpt2",
            "--model_name",
            model_size,
            "--model_dir",
            target_dir,
        ],
        check=True
    )

    print(f"✅ GPT-2 {model_size} download complete")


if __name__ == "__main__":
    download_gpt2("124M", "models/pretrained/gpt2")
    download_gpt2("355M", "models/pretrained/gpt2")
