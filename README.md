# Tiny LLM From Scratch

A hands-on learning project for understanding how modern language models work **from the inside out**.

A hands-on implementation of a tiny, SmolLM2-style decoder Transformer in PyTorch.
> Built as a learning project to understand how modern LLMs work from tokenization and embeddings through attention, training, generation, and KV caching.

This repository starts with a small Transformer language model implemented in PyTorch and gradually explores the core ideas behind real LLMs:

- Token IDs and embeddings
- Self-attention and multi-head attention
- RoPE positional embeddings
- RMSNorm
- SwiGLU
- Causal masking
- Autoregressive generation
- Temperature, top-k and top-p sampling
- KV cache
- Tokenization and BPE
- Training and checkpointing

The goal is not to build a production-quality LLM. The goal is to **understand the mechanics by implementing and observing them ourselves**.

---

## At a Glance

| Component | Implementation |
|---|---|
| Framework | PyTorch |
| Architecture | Decoder-only Transformer |
| Hidden size | 8 |
| Attention heads | 2 |
| Transformer layers | 2 |
| Context length | 4 |
| FFN | SwiGLU |
| Normalization | RMSNorm |
| Positional encoding | RoPE |
| Generation | Greedy / temperature / top-k / top-p |
| Inference optimization | KV cache |
| Tokenizer | Simple word-level tokenizer |

---

## Architecture

```text
                              Tiny LLM

Text
 │
 ▼
Tokenizer
 │
 ▼
Token IDs
 │
 ▼
Embedding
 │
 ▼
┌──────────────────────────────────────────────┐
│            Transformer Block × 2            │
│                                              │
│  RMSNorm                                     │
│     │                                        │
│     ▼                                        │
│  Multi-Head Self-Attention                  │
│     ├── Q / K / V projections               │
│     ├── RoPE                                │
│     ├── Causal Mask                         │
│     └── KV Cache (during generation)        │
│     │                                        │
│     ▼                                        │
│  Residual Connection                         │
│     │                                        │
│     ▼                                        │
│  RMSNorm                                     │
│     │                                        │
│     ▼                                        │
│  SwiGLU / Feed-Forward Network              │
│     │                                        │
│     ▼                                        │
│  Residual Connection                         │
└──────────────────────────────────────────────┘
 │
 ▼
Final RMSNorm
 │
 ▼
LM Head
 │
 ▼
Logits
 │
 ├─────────────── Generation ────────────────┐
 │                                            │
 ▼                                            │
Temperature → Top-k → Top-p → Softmax       │
 │                                            │
 ▼                                            │
Sampling                                     │
 │                                            │
 ▼                                            │
Next Token ──────────────────────────────────┘
```

### Training flow

```text
Text
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Input / Target Sequences
  ↓
Embedding
  ↓
Transformer
  ↓
Logits
  ↓
Cross-Entropy Loss
  ↓
Backpropagation
  ↓
Optimizer
  ↓
Updated Model Weights
```

### Inference flow

```text
Prompt
  ↓
Prefill entire prompt
  ↓
KV Cache
  ↓
Generate one new token
  ↓
Reuse cached K/V
  ↓
Generate next token
  ↓
Repeat until EOS / max_new_tokens
```

---

## Who is this for?

This project is designed for someone who wants to move beyond using LLM APIs and understand what actually happens inside a language model.

It is suitable for:

- Students learning Transformers and LLMs
- Engineers moving into AI/ML
- Developers who want to understand PyTorch-based LLM implementations
- Anyone who wants to eventually build and pretrain a small language model from scratch

You do not need to understand everything before starting. The project is intentionally structured so each concept builds on the previous one.

---

# What We Built

The core project is a small decoder-style Transformer language model written from scratch in PyTorch.

The model pipeline is:

```text
Text
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Embedding
  ↓
Transformer Blocks
  ├── RMSNorm
  ├── Multi-Head Self-Attention
  │     ├── Q / K / V projections
  │     ├── RoPE
  │     ├── Causal Mask
  │     └── Attention
  ├── Residual Connection
  ├── RMSNorm
  ├── SwiGLU
  └── Residual Connection
  ↓
Final RMSNorm
  ↓
LM Head
  ↓
Logits
  ↓
Next-token generation
```

For inference, the generation pipeline additionally supports:

```text
Logits
  ↓
Temperature
  ↓
Top-k
  ↓
Top-p
  ↓
Softmax
  ↓
Sampling
  ↓
Next token
```

and uses a **KV cache** so previously computed attention keys and values do not need to be recomputed at every generation step.

---

# Learning Journey

The repository is intentionally split between:

### `smollm2/`

The working model implementation.

### `learning/`

Focused experiments used to understand individual concepts before integrating them into the model.

### `notebooks/`

Earlier foundational Transformer experiments.

### `docs/`

Supporting notes and equations.

The learning notebooks are part of the project. They are not just scratch work; they document the reasoning and experiments behind the implementation.

---

# Repository Structure

```text
tiny-llm-from-scratch/
│
├── data/
│   └── sample.txt
│
├── docs/
│   └── ...
│
├── learning/
│   ├── [Embeddings](learning/01_embeddings.ipynb)
│   ├── [KV Cache](learning/03_kv_cache.ipynb)
│   └── [Tokenizer / BPE](learning/04_tokenizer.ipynb)
│
├── notebooks/
│   └── [Transformer from Scratch](notebooks/01_transformer_from_scratch.ipynb)
│
├── smollm2/
│   ├── __init__.py
│   ├── attention.py
│   ├── dataset.py
│   ├── model.py
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── swiglu.py
│   └── transformer_block.py
│
├── config.py
├── generate.py
├── model_config.json
├── requirements.txt
├── tokenizer.json
├── tokenizer.py
├── train.py
├── README.md
└── .gitignore
```

---

# Environment Setup

## 1. Clone the repository

```bash
git clone https://github.com/jaiswal-prateek/tiny-llm-from-scratch.git
cd tiny-llm-from-scratch
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Train the Tiny Model

The model is trained using:

```bash
python train.py
```

Training reads:

```text
data/sample.txt
```

The training pipeline is:

```text
Text
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Dataset
  ↓
Batches
  ↓
Transformer
  ↓
Logits
  ↓
Cross-Entropy Loss
  ↓
Backpropagation
  ↓
Optimizer Update
```

The trained checkpoint is saved as:

```text
smollm2_model.pt
```

The model configuration used by the checkpoint is saved as:

```text
model_config.json
```

The tokenizer vocabulary is saved as:

```text
tokenizer.json
```

---

# Generate Text

After training:

```bash
python generate.py
```

The generator loads:

- the tokenizer
- the saved model configuration
- the trained model weights

You can change generation settings near the bottom of `generate.py`:

```python
prompt = "why python learning is good"

max_new_tokens = 10

temperature = 1.0

top_k = None

top_p = 0.9
```

---

# Generation Strategies

## Greedy Decoding

The simplest approach is:

```text
choose the highest-probability token
```

This produces deterministic output.

Conceptually:

```text
token A → 0.60
token B → 0.25
token C → 0.10
token D → 0.05

choose token A
```

---

## Temperature

Temperature changes the shape of the probability distribution before sampling.

```python
next_token_logits = next_token_logits / temperature
```

General behavior:

```text
Low temperature
→ sharper distribution
→ more deterministic

Temperature = 1
→ normal distribution

High temperature
→ flatter distribution
→ more variation
```

---

## Top-k

Top-k keeps only the highest `k` candidate tokens.

For example:

```python
top_k = 20
```

means:

```text
Vocabulary: 578 tokens

        ↓ top-k

20 candidates remain
```

The remaining candidates are then sampled.

---

## Top-p

Top-p, also called nucleus sampling, keeps the smallest set of tokens whose cumulative probability reaches the selected threshold.

For example:

```python
top_p = 0.9
```

The number of candidates can change at every generation step.

```text
Step 1 → 3 candidates
Step 2 → 8 candidates
Step 3 → 20 candidates
Step 4 → 2 candidates
```

This makes top-p adaptive to the model's confidence.

---

# KV Cache

The model also implements KV caching for autoregressive generation.

Without KV caching, generation repeatedly processes the entire sequence:

```text
Step 1:
token1 token2

Step 2:
token1 token2 token3

Step 3:
token1 token2 token3 token4
```

This repeatedly recomputes attention information for previous tokens.

With KV caching:

```text
Prompt
  ↓
Prefill
  ↓
K/V cache

New token
  ↓
compute Q/K/V for new token
  ↓
reuse previous K/V
  ↓
generate next token
```

For example:

```text
Prompt:
token1 token2

Cache:
K1 K2
V1 V2
```

Next token:

```text
token3

New:
Q3 K3 V3

Reuse:
K1 K2
V1 V2
```

Cache becomes:

```text
K1 K2 K3
V1 V2 V3
```

The implementation was verified by comparing cached and non-cached computation. The outputs matched exactly in the validation experiment.

---

# Model Components

## Embeddings

Token IDs are converted into dense vectors using a learned embedding table:

```text
Token ID
   ↓
Embedding lookup
   ↓
Dense vector
```

The embedding weights are learned through the same forward → loss → backward → optimizer process as the rest of the model.

Learning notebook:

```text
learning/01_embeddings.ipynb
```

---

## Multi-Head Self-Attention

The attention mechanism uses:

```text
Q = XWQ
K = XWK
V = XWV
```

and:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √d)
V
```

For multiple heads:

```text
hidden_size = 512
num_heads = 8
head_dim = 64
```

The representation is split into multiple heads, attention is performed independently for each head, and the results are concatenated back to the original hidden size.

Implementation:

```text
smollm2/attention.py
```

---

## RoPE

Rotary Positional Embeddings provide position information to attention by rotating pairs of dimensions in Q and K.

Implementation:

```text
smollm2/rope.py
```

---

## RMSNorm

The model uses RMSNorm before attention and the feed-forward network.

Implementation:

```text
smollm2/rmsnorm.py
```

---

## SwiGLU

The feed-forward network uses the SwiGLU structure:

```text
Input
 ↓
Up projection
 ↓
Gate projection → SiLU
 ↓
Elementwise multiplication
 ↓
Down projection
```

Implementation:

```text
smollm2/swiglu.py
```

---

## Transformer Block

The decoder block follows a pre-normalized structure:

```text
Input
 ↓
RMSNorm
 ↓
Self-Attention
 ↓
Residual
 ↓
RMSNorm
 ↓
SwiGLU
 ↓
Residual
```

Implementation:

```text
smollm2/transformer_block.py
```

---

# Tokenization

The production tokenizer used by this tiny model is intentionally simple.

It provides a straightforward way to understand:

```text
Text
 ↓
Tokens
 ↓
Token IDs
 ↓
Embedding
```

However, it is **not a production LLM tokenizer**.

The tokenizer learning notebook explores:

```text
Character tokenization
       ↓
Subword tokenization
       ↓
BPE
       ↓
Byte-level BPE
       ↓
Real LLM tokenizer
```

Notebook:

```text
learning/04_tokenizer.ipynb
```

The notebook also inspects the real tokenizer used by SmolLM2 to compare the concepts learned here with a real LLM tokenizer.

---

# Important Learning Principle

This repository deliberately separates:

### Understanding

Small experiments where concepts are isolated and visualized.

```text
learning/
notebooks/
docs/
```

from:

### Implementation

The working Transformer model.

```text
smollm2/
train.py
generate.py
```

This makes it easier to understand each component before combining everything into a full model.

---

# Suggested Learning Order

If you are new to LLMs, follow this order:

```text
1. Language models
2. Token IDs
3. Embeddings
4. Forward pass
5. Logits
6. Softmax
7. Cross-entropy
8. Self-attention
9. Multi-head attention
10. Causal masking
11. RoPE
12. RMSNorm
13. SwiGLU
14. Transformer block
15. Training loop
16. Autoregressive generation
17. Temperature
18. Top-k / Top-p
19. KV cache
20. Tokenization / BPE
```

The notebooks and implementation files in this repository follow this general progression.

---

# What This Project Is Not

This is **not the official SmolLM2 implementation**.

The `SmolLM2` class in this repository is a small educational Transformer implementation used to understand the architecture and training process.

The project is intentionally tiny so that the code can be read, modified, and experimented with locally.

---

# Training Experiment and Observations

The tiny model was intentionally trained on a small fictional story so we could observe how data, model capacity, context length, and training duration affect language generation.

The final experiment used:

```text
Hidden size     = 8
Attention heads = 2
Transformer layers = 2
Sequence length = 4
Training epochs = 5000
```

Training loss during the 5000-epoch run:

| Epoch | Loss |
|------:|-----:|
| 0     | 4.78 |
| 500   | 1.45 |
| 1000  | 1.31 |
| 1500  | 1.47 |
| 2000  | 1.44 |
| 2500  | 1.38 |
| 3000  | 1.32 |
| 3500  | 1.28 |
| 4000  | 1.18 |
| 4500  | 1.21 |
| 5000  | 1.15 |

The loss generally improved from about `4.78` to about `1.15`, but it was **not monotonic**. There were repeated increases and decreases during training.

This is useful to observe because training loss does not necessarily decrease smoothly at every step. With a very small dataset and a tiny model, optimization can be noisy and the model can begin fitting the training corpus without becoming consistently better at free-form generation.

## What the Model Learned

After training, prompts related to the story produced continuations containing patterns and phrases from the training corpus.

For example:

```text
Prompt:
bellford

Generated:
bellford to study underground water movement and asked appeared on.
```

Another example:

```text
Prompt:
stop here

Generated:
stop here , " she said had been disconnected , a small
```

These outputs are not consistently coherent, but they demonstrate that the model learned statistical patterns from the training text rather than producing completely unrelated text.

The model can therefore be viewed as a successful **learning experiment**, rather than a high-quality language model.

---

# What We Learned: Results and Limitations

This implementation is intentionally tiny, which creates several important limitations.

### 1. Very small model

The model uses only:

```text
2 Transformer layers
8 hidden dimensions
2 attention heads
```

This provides very little representational capacity compared with modern LLMs.

### 2. Very short context

The current:

```text
SEQ_LEN = 4
```

means the model trains on very short sequences. It therefore has limited ability to learn relationships across longer parts of the story.

### 3. Simple tokenizer

The production tokenizer is intentionally word-level and can produce:

```text
<unk>
```

for unseen words.

The repository includes a separate BPE/tokenizer learning notebook demonstrating how modern subword tokenization can provide much better coverage.

### 4. Very small training corpus

The model is trained on a single small story. This is useful for learning but nowhere near enough data for broad language modeling.

### 5. No validation split

The current training loop reports training loss only. There is no separate validation dataset to measure whether the model is learning general patterns or simply memorizing the training corpus.

### 6. Basic training setup

The implementation does not currently include several techniques commonly used in larger training runs, such as learning-rate scheduling, checkpoint selection based on validation loss, mixed precision, or distributed training.

### 7. Generation quality

Even after 5000 epochs, generated text can be abrupt, repetitive, or grammatically inconsistent. Increasing training time alone does not solve the limitations created by model size, context length, tokenizer quality, and dataset size.

---

# Ideas for Further Improvement

This repository is intentionally left in a state where it can be extended.

Possible next experiments include:

```text
Increase hidden size
→ 8 → 32 → 64 → 128

Increase Transformer depth
→ 2 → 4 → 6+ layers

Increase context length
→ 4 → 32 → 128+ tokens

Use a real subword / byte-level tokenizer
→ remove most <unk> issues

Increase the training corpus
→ one story → many stories / books / curated text

Add a validation split
→ compare training vs validation loss

Experiment with learning rates
→ observe optimization stability

Add learning-rate scheduling
→ compare convergence

Save intermediate checkpoints
→ compare generations at different training stages

Track training curves
→ visualize loss over time

Improve generation
→ temperature, top-k, top-p, EOS, KV cache

Scale the model
→ measure how architecture changes generation quality
```

These experiments turn the tiny model into a useful sandbox for understanding how model size, data, context, tokenization, optimization, and inference strategy affect language-model behavior.

---

## Try These Experiments

The repository is intentionally small enough to modify.

1. Increase `HIDDEN_SIZE` from 8 to 32.
2. Increase `SEQ_LEN` from 4 to 32.
3. Train on a larger corpus.
4. Replace the tokenizer with BPE.
5. Compare greedy decoding with sampling.
6. Compare generation with and without KV caching.
7. Add a validation split and plot training vs validation loss.
8. Increase the number of Transformer layers and observe the effect.

---

# Next Steps

After completing this tiny model, the learning path is:

```text
Tiny Transformer
       ↓
Understand inference + training
       ↓
Inspect a real pretrained SmolLM2 model
       ↓
Fine-tune a real small LLM
       ↓
Build a proper tokenizer/data pipeline
       ↓
Pretrain a small language model from scratch
```

The long-term goal is to move from:

```text
"I know how to use an LLM"
```

to:

```text
"I understand how the model, tokenizer,
training loop, and inference system work."
```

---

# License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.