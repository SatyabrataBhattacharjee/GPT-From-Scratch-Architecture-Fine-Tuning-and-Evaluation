# Pipeline Decomposition

## Common Components
The system is built around a small set of shared components that define stable interfaces between all pipelines. These components are pipeline-agnostic and are reused consistently across pretraining, generation, evaluation, and downstream fine-tuning tasks.

- Tokenization :
All pipelines rely on a GPT-2–compatible Byte Pair Encoding (BPE) tokenizer. The tokenizer defines a fixed vocabulary and a deterministic mapping between raw text and token IDs. Tokenization is performed outside the model and serves as the contract between raw textual data and the neural network. By sharing a single tokenizer instance across pipelines, the system guarantees consistency in token representations during training, evaluation, and inference.

- GPT model core:
At the center of the system is a task-agnostic GPT-style transformer model. The model maps sequences of token IDs to contextualized hidden representations and produces vocabulary-sized logits via a linear output head. The model core is responsible only for representation learning and next-token scoring; it does not implement loss computation, decoding strategies, or task-specific objectives. This separation allows the same model backbone to be reused unchanged across multiple pipelines.

- Output projection:
The final stage of the model core consists of a linear output projection that maps normalized hidden states from the embedding space into the vocabulary space. This projection produces unnormalized logits for every token position in the sequence and defines the interface between the model and downstream consumers. Training pipelines use these logits for loss computation, while generation pipelines consume them for autoregressive decoding. By centralizing output projection within the model, downstream pipelines remain simple and interchangeable.

- Shared utilities:
A collection of shared utility functions supports common operations across pipelines, including dataset construction, batching, loss computation, evaluation routines, and text generation helpers. These utilities are designed to be stateless and composable, enabling consistent behavior without embedding pipeline-specific logic into the model core. This shared layer ensures reproducibility while keeping training, evaluation, and inference workflows cleanly separated.

## Pretraining Pipeline

### Input

- Raw text corpus sourced from an external dataset
- GPT-2–compatible Byte Pair Encoding (BPE) tokenizer
- Model configuration defining vocabulary size, context length, and architectural parameters

### Stages

1. **Data Acquisition**  
   The raw text corpus is retrieved from an external source and stored locally as an immutable text file, ensuring reproducibility and traceability.

2. **Train–Validation Split**  
   The raw corpus is split into training and validation subsets using a fixed ratio prior to tokenization. Performing the split at the text level prevents information leakage between training and evaluation.

3. **Tokenization**  
   Text data is converted into token IDs using a GPT-2–compatible BPE tokenizer, producing a continuous stream of discrete token identifiers.

4. **Context Construction**  
   Token sequences are segmented into fixed-length contexts using a sliding window strategy. Each input sequence is paired with a target sequence shifted by one token, enabling autoregressive next-token prediction.

5. **Batching**  
   Input–target pairs are grouped into mini-batches using a data loader. Training batches are shuffled and truncated for efficiency, while validation batches preserve sequence order.

6. **Model Forward Pass**  
   Batched input token IDs are passed through the GPT model core, which applies token and positional embeddings, stacked transformer blocks, normalization, and output projection to produce vocabulary-sized logits.

7. **Loss Computation**  
   The model’s logits are compared against target token IDs using a cross-entropy loss computed over all token positions in the batch.

8. **Parameter Update**  
   Gradients are computed via backpropagation and model parameters are updated using the AdamW optimizer.

9. **Periodic Evaluation**  
   At configurable intervals, the model is evaluated on both training and validation data using a fixed number of batches to monitor loss trends.

10. **Qualitative Monitoring**  
    The generation pipeline is periodically invoked to produce sample text outputs, providing qualitative insight into model behavior without affecting training dynamics.

### Output

- A pretrained GPT model capable of autoregressive next-token prediction
- Training and validation loss trajectories
- Generated text samples illustrating learning progression

### Artifacts

- Model checkpoints containing learned parameters
- Recorded loss values and token-count metrics
- Optional generated text samples
- Shared tokenizer configuration


## Text Generation Pipeline

### Input

- A raw text prompt provided by the user
- A pretrained GPT model
- Generation parameters such as maximum number of new tokens and context length

### Stages

1. **Prompt Tokenization**  
   The input prompt is converted into token IDs using the shared GPT-2–compatible BPE tokenizer.

2. **Context Management**  
   If the tokenized prompt exceeds the model’s supported context length, only the most recent tokens within the context window are retained.

3. **Model Inference**  
   The tokenized context is passed through the GPT model to produce vocabulary-sized logits for each token position. Only the logits corresponding to the final timestep are used for next-token prediction.

4. **Probability Normalization**  
   The selected logits are transformed into a probability distribution over the vocabulary using a softmax operation.

5. **Decoding Strategy**  
   A decoding policy is applied to select the next token. In the simplest configuration, greedy decoding is used by selecting the token with the highest probability.

6. **Autoregressive Loop**  
   The selected token is appended to the input sequence, and the process repeats until the desired number of new tokens has been generated.

7. **Detokenization**  
   The generated token IDs are converted back into human-readable text using the shared tokenizer.

### Output

- A sequence of generated token IDs
- Decoded natural language text generated autoregressively by the model

### Artifacts

- Generated text samples
- No model parameters are modified during this pipeline


## Classification Fine-Tuning Pipeline
Input:
Stages:
Output:
Artifacts:

## Instruction Fine-Tuning Pipeline
Input:
Stages:
Output:
Artifacts:

## Evaluation Pipeline
Input:
Stages:
Output:
Artifacts:

