
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

The system uses a GPT-2 compatible Byte Pair Encoding (BPE) tokenizer implemented via tiktoken. Raw text is converted into sequences of subword token IDs prior to model execution. Tokenization is performed externally to the model, allowing the GPT architecture to operate purely on numerical token representations and enabling reuse across training, fine-tuning, and inference pipelines.


### 3.3 Token IDs to Embeddings

Token IDs are converted into dense vector representations using a learned embedding layer. To encode sequence order, learned positional embeddings are added to the token embeddings. The combined embeddings are regularized with dropout before being passed into the transformer stack.

---

## 4. Core Model Architecture

### 4.1 Self-Attention Mechanism

Self-attention is the core operation that allows the model to contextualize each token in a sequence by relating it to all other tokens. Instead of processing tokens independently or sequentially, self-attention enables the model to dynamically determine which tokens are most relevant when forming a representation for a given position.

In this architecture, self-attention is implemented by projecting the input embeddings into three separate representations: queries, keys, and values. Each token embedding is linearly transformed into these three spaces using learned projection matrices. The query representation determines what a token is looking for, while the key representations determine what information each token offers. The value representations contain the actual information that may be aggregated.

Attention scores are computed by taking the dot product between query and key representations, producing a similarity measure between tokens. These scores are scaled by the square root of the key dimension to stabilize gradients and then normalized using a softmax function, yielding attention weights that sum to one across the sequence.

The final context representation for each token is obtained by taking a weighted sum of the value vectors, where the weights are given by the attention distribution. As a result, each token’s representation becomes a context-aware mixture of other tokens in the sequence, allowing the model to capture dependencies such as syntax, semantics, and long-range relationships.

Self-attention operates on the entire sequence simultaneously, making it highly parallelizable and well-suited for modeling complex interactions in language. However, in its raw form, self-attention does not enforce any notion of temporal order or causality; these constraints are introduced separately through positional embeddings and causal masking in later components of the architecture.


### 4.2 Causal Masking

Causal attention is a constrained form of self-attention designed for autoregressive language modeling. Its purpose is to ensure that, when predicting a token at a given position, the model can only attend to tokens that occur earlier in the sequence, and never to future tokens.

In this architecture, causal attention builds on standard self-attention by introducing an explicit causal mask. As in regular self-attention, the input embeddings are projected into query, key, and value representations using learned linear transformations. Attention scores are computed as similarities between queries and keys across the sequence.

Before normalization, a causal mask is applied to the attention scores. This mask blocks all positions corresponding to future tokens by assigning them a negative infinity value, effectively forcing their attention weights to zero after the softmax operation. As a result, each token can only attend to itself and to tokens that precede it in the sequence.

This masking mechanism enforces the autoregressive constraint required for language modeling, preventing information leakage from future tokens during both training and inference. Importantly, the mask is applied dynamically based on the actual sequence length, allowing the model to handle variable-length inputs while respecting a fixed maximum context window.

After masking and normalization, dropout is applied to the attention weights as a form of regularization. The final context representation for each token is computed as a weighted sum of the value vectors, using the masked attention distribution.

Causal attention enables the model to learn sequential dependencies while remaining fully parallelizable during training, making it a foundational component of GPT-style architectures.


### 4.3 Multi-Head Attention

Multi-head attention extends single-head self-attention by allowing the model to attend to information from multiple representation subspaces simultaneously. Instead of computing a single attention distribution over the entire embedding space, the input is projected into multiple parallel attention heads, each operating on a lower-dimensional subspace.

In this architecture, the input embeddings are first projected into query, key, and value representations using learned linear transformations. The resulting vectors are then reshaped to introduce an explicit head dimension, effectively splitting the embedding space into multiple attention heads. Each head independently computes scaled dot-product attention over the sequence while sharing the same causal masking mechanism.

For each head, attention scores are computed between queries and keys, masked to prevent access to future tokens, scaled for numerical stability, and normalized using a softmax function. Dropout is applied to the attention weights to improve generalization. The weighted sum of value vectors produces a context representation for each head.

After attention is computed independently across all heads, the resulting context vectors are concatenated back into a single representation. A final linear projection is applied to combine information from all heads and map the output back into the model’s embedding dimension.

By attending to different aspects of the sequence in parallel, multi-head attention allows the model to capture diverse relationships such as short-range dependencies, long-range context, and syntactic or semantic patterns. This mechanism significantly increases the expressive power of the model without sacrificing parallelism, making it a central component of GPT-style transformer architectures.


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

