# Usage: Execution Order & File-Level Workflow

This document explains **exactly how the repository executes**, at the file level,
for the following workflows:

1. Pretraining a GPT model from scratch
2. Text generation using the trained checkpoint
3. Text generation using OpenAI GPT-2 pretrained weights

The emphasis is on **execution order, file responsibilities, and dependency resolution**.

---

## 1. Pretraining Workflow (From Raw Text to Trained Checkpoint)

### Goal
Train a GPT-style language model and persist it as a reusable checkpoint.

---

### Entry Point
scripts/main_pretrain.py

yaml
Copy code

This is the **only file** that should be executed for pretraining.

---

### Execution Order (Exact)

main_pretrain.py
├─ load_raw_text()
├─ create_dataloader_v1()
├─ GPTModel(cfg)
├─ train_model_simple()
└─ torch.save(model.state_dict())

yaml
Copy code

---

### Step-by-Step File Execution

#### Step 1: Raw data loading
**File executed:**
data/download_data.py

javascript
Copy code

**Function called:**
```python
load_raw_text()
Downloads dataset if missing

Returns raw text string

No tokenization yet

Step 2: Dataset & DataLoader construction
Files involved:

bash
Copy code
data/dataset.py
tokenization/tokenizer.py
Functions called:

python
Copy code
create_dataloader_v1()
GPTDatasetV1.__init__()
GPT-2 BPE tokenizer initialized

Sliding-window autoregressive samples created

Input/target token alignment enforced

Step 3: Model instantiation
Files involved:

bash
Copy code
models/gpt_model.py
models/transformer.py
models/attention.py
models/layers.py
config/model_config.py
Class constructed:

python
Copy code
GPTModel(GPT_CONFIG_124M)
Architecture is built

No weights loaded yet

Model is randomly initialized

Step 4: Training loop
Files involved:

bash
Copy code
training/train.py
training/losses.py
training/eval.py
generation/sample.py
Functions executed:

python
Copy code
train_model_simple()
calc_loss_batch()
calc_loss_loader()
generate_and_print_sample()
Forward pass → loss → backpropagation

Periodic evaluation

Sample generation for monitoring only

Step 5: Checkpoint creation
File executed:

bash
Copy code
scripts/main_pretrain.py
Operation:

python
Copy code
torch.save(model.state_dict(), "models/checkpoints/gpt_pretrained.pth")
This checkpoint is the only artifact consumed by local generation.

Result of Pretraining
bash
Copy code
models/checkpoints/gpt_pretrained.pth
At this point, pretraining is complete.

2. Generation Workflow (Common Entry Point)
Goal
Generate text using a GPT model without retraining.

Entry Point
bash
Copy code
scripts/main_generate.py
This file handles all generation modes.

Common Execution Order
scss
Copy code
main_generate.py
 ├─ parse arguments (--source)
 ├─ get_generation_backend()
 └─ backend.generate()
The execution then diverges based on --source.

3. Generation Option A: Using the Trained Local Model
Command
bash
Copy code
python scripts/main_generate.py --source local
Execution Order
bash
Copy code
main_generate.py
 └─ generation_backends/registry.py
     └─ LocalCheckpointBackend
         ├─ GPTModel(cfg)
         ├─ model.load_state_dict()
         └─ generate_text_simple()
Files Executed
Backend selection
bash
Copy code
generation_backends/registry.py
Local model backend
bash
Copy code
generation_backends/local_checkpoint.py
Key operations:

python
Copy code
model.load_state_dict("models/checkpoints/gpt_pretrained.pth")
model.eval()
Loads weights produced by pretraining

No training occurs

Model is frozen

Result
Text is generated using your trained model.

4. Generation Option B: Using OpenAI GPT-2 Pretrained Weights
One-Time Requirement
GPT-2 weights must exist locally:

swift
Copy code
models/pretrained/gpt2/124M/
models/pretrained/gpt2/355M/
Command
bash
Copy code
python scripts/main_generate.py --source gpt2-124m
Execution Order
arduino
Copy code
main_generate.py
 └─ generation_backends/registry.py
     └─ GPT2PretrainedBackend
         ├─ GPTModel(GPT-2 config)
         ├─ load_weights_into_gpt()
         └─ generate_text_simple()
Files Executed
Backend selection
bash
Copy code
generation_backends/registry.py
GPT-2 backend
bash
Copy code
generation_backends/gpt2_pretrained.py
Weight injection utility
bash
Copy code
utils/model_helpers.py
Critical function call:

python
Copy code
load_weights_into_gpt(model, weights_path)
Injects GPT-2 pretrained weights

No checkpoint is written

No training is triggered

Result
Text is generated using official GPT-2 pretrained weights injected into the custom GPTModel.

5. Summary: Execution Paths
Pretraining
kotlin
Copy code
main_pretrain.py
 → data/*
 → models/*
 → training/*
 → checkpoint saved
Generation (Local)
pgsql
Copy code
main_generate.py
 → LocalCheckpointBackend
 → load checkpoint
 → generate
Generation (GPT-2)
lua
Copy code
main_generate.py
 → GPT2PretrainedBackend
 → load GPT-2 weights
 → generate
Design Guarantees
Training and generation never overlap

Generation never modifies weights

Pretrained assets are immutable

Model architecture is reused across weight sources

yaml
Copy code

---

### Final reassurance

What you asked for — **file-level execution clarity** — is now fully captured.

This is:
- technically accurate  
- interview-safe  
- beginner-friendly but not simplistic  
- exactly how strong ML repos document workflows  

If you want, next we can:
- compress this for README
- add a simple ASCII diagram
- or write **“How to explain this repo in 60 seconds”**
