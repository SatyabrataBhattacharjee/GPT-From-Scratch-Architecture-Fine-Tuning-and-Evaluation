# GPT-From-Scratch-Architecture-Fine-Tuning-and-Evaluation
An end-to-end implementation of a GPT-style Large Language Model built from first principles in PyTorch, covering custom tokenization, attention mechanisms, multi-head transformers, pretrained GPT-2 weight loading, fine-tuning for downstream tasks, and automated evaluation of instruction-following behavior.

## Why This Repository

Most modern LLM workflows rely heavily on high-level abstractions.
This project intentionally avoids those abstractions to demonstrate a ground-up understanding of:

- How attention and causal masking actually work
- How GPT-style models are assembled and trained
- How pretrained weights can be adapted to custom architectures
- How generative models are fine-tuned and evaluated for real tasks


This repository implements a GPT-style language model **from scratch**, with **cleanly separated execution paths** for:

1. Pretraining a GPT model  
2. Generating text using a locally trained checkpoint  
3. Generating text using OpenAI GPT-2 pretrained weights  

The focus is **deterministic execution**, **clear file responsibilities**, and **zero overlap between training and generation**.

---

## 1. Pretraining Workflow (Raw Text → Trained Checkpoint)

### Goal
Train a GPT-style language model and persist it as a reusable checkpoint.

### Entry Point (ONLY)

    python scripts/main_pretrain.py

---

### Exact Execution Order

    main_pretrain.py
    ├─ load_raw_text()
    ├─ create_dataloader_v1()
    ├─ GPTModel(cfg)
    ├─ train_model_simple()
    └─ torch.save(model.state_dict())

---

### File-Level Execution Details

#### Step 1: Raw data loading

**File**

    data/download_data.py

**Function**

    load_raw_text()

**Responsibility**
- Downloads dataset if missing  
- Returns raw text string  
- No tokenization performed  

---

#### Step 2: Dataset & DataLoader construction

**Files**

    data/dataset.py
    tokenization/tokenizer.py

**Functions**

    create_dataloader_v1()
    GPTDatasetV1.__init__()

**What happens**
- GPT-2 BPE tokenizer initialized  
- Sliding-window autoregressive samples created  
- Input–target token alignment enforced  

---

#### Step 3: Model instantiation

**Files**

    models/gpt_model.py
    models/transformer.py
    models/attention.py
    models/layers.py
    config/model_config.py

**Class**

    GPTModel(GPT_CONFIG_124M)

- Full GPT architecture built  
- Weights randomly initialized  
- No pretrained weights loaded  

---

#### Step 4: Training loop

**Files**

    training/train.py
    training/losses.py
    training/eval.py
    generation/sample.py

**Functions**

    train_model_simple()
    calc_loss_batch()
    calc_loss_loader()
    generate_and_print_sample()

- Forward pass → loss → backpropagation  
- Periodic evaluation  
- Sample generation for monitoring only  

---

#### Step 5: Checkpoint creation

**File**

    scripts/main_pretrain.py

**Operation**

    torch.save(model.state_dict(),
               "models/checkpoints/gpt_pretrained.pth")

**Result**

    models/checkpoints/gpt_pretrained.pth

Pretraining ends here.

---

## 2. Generation Workflow (Common Entry Point)

### Goal
Generate text without retraining.

### Entry Point

    python scripts/main_generate.py --source <mode>

### Common Execution Flow

    main_generate.py
    ├─ parse arguments (--source)
    ├─ get_generation_backend()
    └─ backend.generate()

Execution diverges based on `--source`.

---

## 3. Generation Option A: Local Trained Model

### Command

    python scripts/main_generate.py --source local

### Execution Path

    main_generate.py
    └─ generation_backends/registry.py
       └─ LocalCheckpointBackend
          ├─ GPTModel(cfg)
          ├─ model.load_state_dict()
          └─ generate_text_simple()

### Files

    generation_backends/registry.py
    generation_backends/local_checkpoint.py

**Key Operations**

    model.load_state_dict("models/checkpoints/gpt_pretrained.pth")
    model.eval()

- Loads locally trained weights  
- No training occurs  
- Model is frozen  

---

## 4. Generation Option B: OpenAI GPT-2 Pretrained Weights

### One-Time Requirement

    models/pretrained/gpt2/124M/
    models/pretrained/gpt2/355M/

### Command

    python scripts/main_generate.py --source gpt2-124m

### Execution Path

    main_generate.py
    └─ generation_backends/registry.py
       └─ GPT2PretrainedBackend
          ├─ GPTModel(GPT-2 config)
          ├─ load_weights_into_gpt()
          └─ generate_text_simple()

### Files

    generation_backends/registry.py
    generation_backends/gpt2_pretrained.py
    utils/model_helpers.py

**Critical Call**

    load_weights_into_gpt(model, weights_path)

- Injects official GPT-2 weights  
- No checkpoint written  
- No training triggered  

---

## 5. Execution Summary

### Pretraining

    main_pretrain.py
    → data/*
    → models/*
    → training/*
    → checkpoint saved

### Generation (Local)

    main_generate.py
    → LocalCheckpointBackend
    → load checkpoint
    → generate

### Generation (GPT-2)

    main_generate.py
    → GPT2PretrainedBackend
    → load GPT-2 weights
    → generate

---

## Design Guarantees

- Training and generation never overlap  
- Generation never modifies weights  
- Pretrained assets are immutable  
- Same GPT architecture reused across all weight sources  

This repository is intentionally **transparent and execution-order explicit**.


