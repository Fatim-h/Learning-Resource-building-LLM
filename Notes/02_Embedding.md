# Notes: `Embedding.py` — Turning Token IDs into Vectors

**Goal of this file:** show how integer token IDs (from `Tokenizer.py`) get converted into dense vectors the model can actually do math on, and how positional information gets injected since embeddings alone have no sense of order.

---

## Step 1 — A tiny toy embedding layer

```python
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```

- `nn.Embedding(vocab_size, output_dim)` internally creates a **weight matrix of shape `[vocab_size, output_dim]`** — one row per possible token, each row a learnable vector of length `output_dim`.
- Conceptually this is equivalent to: one-hot-encode the token ID, then multiply by a `[vocab_size, output_dim]` weight matrix. `nn.Embedding` just does this as an efficient lookup instead of an actual matrix multiply.
- `torch.manual_seed(123)` makes the random initialization reproducible.
- `print(embedding_layer.weight)` shows the full 6×3 matrix — this is what gets *learned* during training.

```python
print(embedding_layer(torch.tensor([3])))
```
- Looking up token ID `3` simply returns **row 3** of that weight matrix — a vector of length 3. This vector is the model's current "understanding" of token 3's meaning (random and meaningless before training; learned/updated during training).

**Key idea:** an embedding layer is just a lookup table, `token ID → vector`, and that table's values are trainable parameters.

## Step 2 — Why we also need *positional* information

- Later in the pipeline, self-attention processes all tokens **in parallel** and computes weighted averages — it has no built-in concept of "this word came before that word." Two sentences with the same words in different order would look identical to plain attention.
- Solution: add a second embedding table, indexed by **position** (0, 1, 2, 3, ...) instead of by token identity, and add it to the token embedding.

## Step 3 — Realistic GPT-style dimensions

```python
vocab_size = 50257     # matches GPT-2's BPE vocab from tiktoken (Tokenizer.py)
output_dim = 256        # embedding dimension (GPT-3 uses 12,288 — this is a downscaled demo)
context_length = 4      # max sequence length the model looks at
```

## Step 4 — Token embeddings for a batch of real sequences

```python
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

inputs = torch.tensor([
    [40,   367,  2885,  1464],
    [1807, 3619,  402,   271],
    ...
])   # shape: [8, 4]  (8 sequences, 4 tokens each — this is the `inputs` batch from Tokenizer.py)

token_embeddings = token_embedding_layer(inputs)
```

- `inputs` shape is `[batch=8, seq_len=4]` — this is literally the kind of batch `create_dataloader_v1` produces.
- Passing a whole 2D tensor of IDs into the embedding layer looks up a vector for *every* ID at once.
- Result shape: `[8, 4, 256]` — for each of the 8 sequences, each of the 4 tokens now has a 256-dim vector instead of a bare integer.

## Step 5 — Positional embeddings

```python
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)  # [4, 256] table

pos_indices = torch.arange(context_length)     # tensor([0, 1, 2, 3])
pos_embeddings = pos_embedding_layer(pos_indices)   # shape: [4, 256]
```

- A separate embedding table, this time with only `context_length` (4) rows — one vector per possible *position* in the sequence, not per token identity.
- `torch.arange(context_length)` generates the position indices `0, 1, 2, 3`.
- Looking these up gives a `[4, 256]` matrix: one 256-dim "position vector" for each of the 4 slots.

## Step 6 — Combine token meaning + position

```python
input_embeddings = token_embeddings + pos_embeddings
```

- `token_embeddings` is `[8, 4, 256]` and `pos_embeddings` is `[4, 256]`.
- PyTorch **broadcasts** the `[4, 256]` positional matrix across the batch dimension, adding the *same* positional vectors to every one of the 8 sequences.
- Effect: position 0 in every sequence gets `+pos_vector_0`, position 1 gets `+pos_vector_1`, etc. Now identical tokens in different positions end up with different final vectors — order is encoded.
- Final shape: `[8, 4, 256]` — this is exactly what gets fed into the transformer's attention layers next.

---

### Summary of what this file demonstrates
1. `nn.Embedding` = a trainable lookup table mapping token ID → dense vector.
2. Token embeddings capture *meaning*; on their own they carry no order information.
3. A second embedding table, indexed by position instead of token ID, produces vectors that represent *where* a token sits in the sequence.
4. Adding token embeddings + positional embeddings (broadcast across the batch) produces the final input representation — shape `[batch, seq_len, output_dim]` — that gets handed to self-attention (see `03_SelfAttention.md`).
