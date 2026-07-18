# Building an LLM from Scratch — Notes Overview

These notes walk through code files that together form pipeline used in Sebastian Raschka's *Build a Large Language Model (From Scratch)*.
(currently at selfattention)

## The Pipeline

```
Raw Text
   │
   ▼
[1] TOKENIZATION  (Tokenizer.py)
   Turn text into a list of integer IDs the model can consume.
   │
   ▼
[2] EMBEDDING  (Embedding.py)
   Turn each integer ID into a dense vector (meaning),
   then add a positional vector (order).
   │
   ▼
[3] SELF-ATTENTION  (SelfAttention.py)
   Let each token vector "look at" other token vectors
   and blend in relevant context, respecting causality
   (can't look at future tokens) and using multiple heads
   to capture multiple relationships at once.
   │
   ▼
Context-rich vectors → fed into the rest of the transformer
(feed-forward layers, more attention blocks, etc. — not covered yet)
```

## Files in this note set

| Note file | Covers | Source code |
|---|---|---|
| `01_Tokenizer.md` | Turning raw text into token IDs, custom vocab tokenizers, BPE (tiktoken), sliding-window dataset/dataloader | `Tokenizer.py` |
| `02_Embedding.md` | Token embedding lookup tables, positional embeddings, combining them | `Embedding.py` |
| `03_SelfAttention.md` | Simplified attention, trainable Q/K/V attention, causal masking, multi-head attention | `SelfAttention.py` |

## Key vocabulary cheat-sheet

- **Token** — a chunk of text (word, subword, or punctuation mark) treated as one unit.
- **Vocabulary** — the fixed set of all possible tokens the model knows.
- **Token ID** — the integer index of a token in the vocabulary.
- **Embedding** — a dense vector that represents the *meaning* of a token id (learned during training).
- **Positional embedding** — a dense vector that represents *where* a token sits in the sequence (since attention alone is order-agnostic).
- **Context length** — the maximum number of tokens the model looks at in one pass.
- **Attention score** — a raw similarity measure between two tokens (dot product of query and key).
- **Attention weight** — the score after softmax, so all weights for a token sum to 1.
- **Context vector** — the final output for a token: a weighted blend of all (allowed) tokens' value vectors.
- **Causal mask** — a rule that hides future tokens from a given position, so the model can't "cheat" by seeing what comes next.
- **Multi-head attention** — running several smaller attention operations in parallel, each potentially learning a different kind of relationship, then concatenating and projecting the results.

Read the notes in order (Tokenizer → Embedding → SelfAttention) — each stage's output is the next stage's input.
