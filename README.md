# GPT-From-Scratch-Architecture-Fine-Tuning-and-Evaluation
An end-to-end implementation of a GPT-style Large Language Model built from first principles in PyTorch, covering custom tokenization, attention mechanisms, multi-head transformers, pretrained GPT-2 weight loading, fine-tuning for downstream tasks, and automated evaluation of instruction-following behavior.

## Why This Repository

Most modern LLM workflows rely heavily on high-level abstractions.
This project intentionally avoids those abstractions to demonstrate a ground-up understanding of:

- How attention and causal masking actually work
- How GPT-style models are assembled and trained
- How pretrained weights can be adapted to custom architectures
- How generative models are fine-tuned and evaluated for real tasks

