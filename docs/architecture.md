
# Architecture Overview

## 1. Purpose of This Document

The purpose of this document is to provide a bird’s-eye view of the overall architecture and design of this repository. It is intended to help the reader understand how the different components of the system fit together, without diving into implementation-level details.

The document begins by presenting the end-to-end flow of data through the system, followed by a clear definition of the architectural boundaries within which the model is designed to operate. It then explains the input representation and tokenization pipeline, describing how raw text is handled, the tokenization strategies employed, and the mapping from token IDs to embedding representations.

Next, the document outlines the core model architecture, covering the self-attention mechanism, causal masking, multi-head attention, and the composition of transformer blocks. This is followed by an explanation of the GPT model assembly, showing how the full model is constructed from stacked transformer blocks and how output projections are produced.

The later sections describe the training and fine-tuning architecture, the text generation process, and the evaluation framework used to assess model performance. Finally, the document discusses artifacts, system state, design decisions, and trade-offs, providing context on the constraints and intentional choices made during development.



---

## 2. System at a Glance

### 2.1 End-to-End Flow

At a high level, the system defines an end-to-end flow that transforms raw text inputs into token-level predictions and task-specific outputs using a GPT-style autoregressive transformer architecture.

The flow begins with raw text, which is converted into discrete token identifiers through the tokenization pipeline. These token IDs are mapped to dense vector representations using a learned token embedding layer, and positional information is added through a positional embedding layer to encode sequence order. A dropout operation is applied to the combined embeddings to improve training stability and regularization.

The embedded sequence is then passed through a stack of transformer blocks. Each transformer block applies multi-head self-attention with causal masking to ensure that each token attends only to previous tokens in the sequence, followed by a position-wise feed-forward network. Layer normalization, residual (shortcut) connections, and dropout are used throughout the block to stabilize optimization and preserve gradient flow as depth increases.

After processing through all transformer blocks, the model produces a sequence of contextualized hidden representations. These representations are projected into vocabulary-sized logits, yielding a probability distribution over the next token at each timestep.

Depending on the operational mode, these outputs are used in different ways: during training or fine-tuning, the logits are evaluated against target tokens to compute loss and update model parameters; during inference, the logits are decoded autoregressively to generate text. Throughout this flow, the system produces artifacts such as model checkpoints and evaluation outputs, which are managed externally as part of the model lifecycle.


### 2.2 Architectural Boundaries
The architecture is intentionally designed with clear boundaries to separate concerns and maintain conceptual simplicity.

The system focuses on model architecture, training pipelines, fine-tuning workflows, and evaluation mechanisms for GPT-style language models. Responsibilities such as large-scale distributed training, model serving infrastructure, and production deployment are considered out of scope for this repository.

Data handling is limited to local, reproducible workflows, with explicit separation between raw and processed datasets. Model artifacts, including pretrained weights and fine-tuned checkpoints, are managed as external state and are excluded from version control.

By constraining the scope in this way, the architecture prioritizes clarity, reproducibility, and understanding of LLM internals, while leaving infrastructure-heavy concerns to downstream systems or production environments.


---

## 3. Input Representation & Tokenization

### 3.1 Raw Text Handling

Raw text is processed outside the model using a Byte Pair Encoding (BPE) tokenizer. The tokenizer converts text into token IDs, which are then passed to the GPT model. The model itself operates purely on numerical token representations and produces token-level outputs, which are decoded back into text using the same tokenizer.


### 3.2 Tokenization Strategy


### 3.3 Token IDs to Embeddings


---

## 4. Core Model Architecture

### 4.1 Self-Attention Mechanism


### 4.2 Causal Masking


### 4.3 Multi-Head Attention


### 4.4 Transformer Block Composition


---

## 5. GPT Model Assembly

### 5.1 Stacking Transformer Blocks


### 5.2 Output Projection


---

## 6. Training & Fine-Tuning Architecture

### 6.1 Pretraining Objective


### 6.2 Task-Specific Fine-Tuning

#### 6.2.1 Classification Fine-Tuning


#### 6.2.2 Instruction Fine-Tuning


---

## 7. Text Generation Architecture


---

## 8. Evaluation Architecture

### 8.1 Loss-Based Evaluation


### 8.2 LLM-Based Evaluation


---

## 9. Artifacts, State, and Boundaries


---

## 10. Design Decisions & Trade-Offs


---

## 11. Summary

