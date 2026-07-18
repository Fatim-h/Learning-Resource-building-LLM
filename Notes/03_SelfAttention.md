# Notes: `SelfAttention.py` — Letting Tokens Look at Each Other

**Goal of this file:** build up self-attention in four increasing stages:
1. Simplified attention with no trainable weights.
2. Trainable attention (scaled dot-product) using raw parameters, then `nn.Linear`.
3. Causal (masked) attention, so a token can't "see" future tokens.
4. Multi-head attention, running several attention operations in parallel.

---

## Part 1 — Simplified self-attention (no learned weights)

```python
inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    ...
])  # shape [6, 3] — 6 tokens, each a 3-dim embedding (output of Embedding.py, simplified)
```

### 1. Attention scores — how similar are tokens to each other?
```python
attn_scores = inputs @ inputs.T   # [6,3] @ [3,6] = [6,6]
```
- Every token's vector is dot-producted with every other token's vector.
- A **higher dot product** = the two vectors point in a similar direction = the model considers them more related/relevant to each other.
- Result: a `[6, 6]` matrix where entry `(i, j)` = how much token `i` "relates to" token `j`.

### 2. Attention weights — normalize with softmax
```python
attn_weights = torch.softmax(attn_scores, dim=-1)
```
- Softmax is applied **per row** (`dim=-1`), converting each row of raw scores into a probability distribution: all positive, summing to 1.
- This is verified with `attn_weights.sum(dim=-1)` → all rows should print `1.0`.
- Interpretation: for token `i`, `attn_weights[i, j]` = "what fraction of my attention goes to token `j`."

### 3. Context vectors — weighted blend
```python
all_context_vecs = attn_weights @ inputs   # [6,6] @ [6,3] = [6,3]
```
- Each token's new representation = a weighted sum of **every** token's original vector, weighted by how much attention it pays to each.
- Output shape matches input shape `[6, 3]`, but now each vector is "context-aware" instead of standalone.

### Manual verification (for row index 1, "journey")
```python
query = inputs[1]
attn_scores_2[i] = torch.dot(x_i, query)   # for each token x_i
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
context_vec_2 += attn_weights_2[i] * x_i   # weighted sum, one token at a time
```
- Recomputes the same result with an explicit loop instead of matrix ops, then confirms it matches `all_context_vecs[1]` — proving the matrix formulation is just a vectorized version of "for each token, weight and sum all the others."

**Limitation of this simplified version:** there are no learnable parameters — the token's own raw embedding is used directly as its "query," "key," and "value." Real attention learns three separate *projections* of each token, so it can learn different notions of relevance.

---

## Part 2 — Trainable attention (scaled dot-product attention)

Introduces three learned weight matrices:
- **W_query** — projects a token into a "what am I looking for" vector.
- **W_key** — projects a token into a "what do I offer" vector.
- **W_value** — projects a token into "what information do I actually carry" vector.

```python
d_in = inputs.shape[1]   # 3 (input embedding size)
d_out = 2                # 2 (attention/output size — can differ from d_in)
```

### `SelfAttention_v1` — raw `nn.Parameter` weights
```python
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
```
- `queries`, `keys`, `values`: each token's embedding is projected through its own weight matrix → shape `[6, 2]` each.
- `attn_scores = queries @ keys.T`: same idea as Part 1, but now scores are computed between *projected* queries and keys, not raw embeddings — the model can learn what makes two tokens "relevant."
- **Scaling by `keys.shape[-1]**0.5`** (i.e. `sqrt(d_out)`): dividing scores by the square root of the key dimension before softmax. This prevents scores from growing too large as dimensionality increases, which would otherwise push softmax into regions with extremely small gradients (this is the "scaled" in "scaled dot-product attention," from the original Transformer paper).
- `context_vec = attn_weights @ values`: same weighted-sum idea as before, but now blending the *value* projections, not the raw inputs.

### `SelfAttention_v2` — using `nn.Linear` instead
```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    ...
```
- Functionally the same computation as V1, but `nn.Linear` is the standard, better-initialized, bias-optional way to do a learned linear projection in PyTorch. This is the version real implementations use.
- Note: `nn.Linear` stores its weight as `[d_out, d_in]` internally (transposed relative to the raw `nn.Parameter` version), and computes `x @ W.T` under the hood.

### Exercise: making V1 match V2 exactly
```python
sa_v1_aligned.W_query = nn.Parameter(sa_v2.W_query.weight.T)
sa_v1_aligned.W_key   = nn.Parameter(sa_v2.W_key.weight.T)
sa_v1_aligned.W_value = nn.Parameter(sa_v2.W_value.weight.T)
```
- Since `nn.Linear`'s weight is stored transposed compared to the raw parameter version, copying `.weight.T` (transpose) into V1's parameters makes the two implementations mathematically identical.
- `torch.allclose(v1_aligned_output, v2_output)` → `True` confirms this — it's the same operation, just two different PyTorch APIs for expressing it.

---

## Part 3 — Causal attention (masking the future)

**Problem:** so far, every token can attend to *every* other token, including ones that come *after* it. For a language model that predicts the next word, letting a token see future words during training would be cheating — the model needs to learn to predict using only what came before.

**Solution:** mask out (zero-probability) any attention from token `i` to any token `j > i`.

```python
inputs = torch.tensor([...])  # shape [2, 6, 3] — now batched: (batch=2, tokens=6, features=3)
```

### `SingleCausalAttention`
```python
self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
```
- `torch.triu(..., diagonal=1)` builds an **upper-triangular matrix of 1s**, excluding the diagonal — i.e., a `[context_length, context_length]` grid where entry `(i, j)` is `1` if `j > i` (strictly future position) and `0` otherwise.
- `register_buffer` stores this as part of the model's state (moves with `.to(device)`, saved/loaded with the model) but it's **not a trainable parameter** — it's a fixed rule, not something learned.

```python
attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))
```
- Every score at a "future" position gets overwritten with `-inf`.
- Why `-inf` and not `0`? Because the mask is applied **before** softmax. `softmax(-inf) = 0`, so after normalization those positions get exactly zero attention weight. (Zeroing scores directly, before softmax, wouldn't guarantee zero weight after softmax — `e^0 = 1`, not 0.)

```python
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
context_vec = attn_weights @ values
```
- Same scaled softmax + weighted sum as Part 2, but now each token's context vector is a blend of **only itself and earlier tokens** — never future ones.

---

## Part 4 — Multi-head attention (parallel attention "perspectives")

**Motivating example given in the code's docstring:** in *"The animal didn't cross the street because it was too tired,"* a single attention head has to choose one interpretation of "it." Change the sentence to *"...because it was too wide"* and "it" now refers to the street. **Multiple heads let the model track several relationships simultaneously** — one head might link "it" → "animal" (subject tracking), another might link "it" → "tired"/"wide" (state/descriptor tracking) — instead of forcing one head to average across all of them.

```python
class ParallelMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads):
        super().__init__()
        assert d_out % num_heads == 0
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key   = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
```
- Rather than creating `num_heads` separate small attention modules, this implementation does one **big** projection (`d_in → d_out`) and then **splits** the result into heads — this is more efficient (fewer, larger matrix multiplies instead of many small ones).
- `head_dim = d_out // num_heads`: each head gets an equal slice of the total output dimension.

### Forward pass, step by step
```python
queries = self.W_query(x)   # [b, tokens, d_out]
keys    = self.W_key(x)
values  = self.W_value(x)
```
1. Project the whole batch into unified Q/K/V spaces, same as single-head attention.

```python
queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
...
queries = queries.transpose(1, 2)   # -> [b, heads, tokens, head_dim]
```
2. **Reshape** `d_out` into `(num_heads, head_dim)` — splitting the big vector into `num_heads` smaller chunks.
3. **Transpose** so `heads` becomes a batch-like dimension, letting all heads be computed in one batched matrix multiply instead of a Python loop.

```python
attn_scores = queries @ keys.transpose(2, 3)   # [b, heads, tokens, tokens]
attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
context_vec = attn_weights @ values            # [b, heads, tokens, head_dim]
```
4. Exactly the same causal, scaled-softmax attention math as `SingleCausalAttention` — just applied independently, per head, all at once via batched matmul. The same causal mask is reused/broadcast across every head.

```python
context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
return self.out_proj(context_vec)
```
5. **Re-assemble:** transpose back so tokens are the second dimension again, then flatten `(heads, head_dim)` back into one `d_out`-sized vector per token — concatenating what each head learned.
6. **`out_proj`**: one final learned linear layer that mixes information across heads (lets the model combine/reweight what different heads discovered, rather than just leaving them side-by-side).

### Verification
```python
single_out = single_head(inputs)   # [2, 6, 2]
mha_out = mha_block(inputs)        # [2, 6, 2]
```
- Both produce the same output shape (`[batch, tokens, d_out]`) even though multi-head internally splits work into smaller pieces — confirming multi-head attention is a drop-in, more expressive replacement for single-head attention.

---

### Summary of what this file demonstrates
1. **Simplified attention**: dot-product similarity → softmax → weighted sum, using raw embeddings directly (no learning).
2. **Trainable attention**: learn separate `Query`, `Key`, `Value` projections; scale scores by `sqrt(d_out)` before softmax for stable training.
3. **Causal masking**: force each token to attend only to itself and earlier tokens (via an upper-triangular `-inf` mask before softmax), which is essential for autoregressive next-token prediction.
4. **Multi-head attention**: split the projected Q/K/V into several smaller "heads" computed in parallel, so the model can capture multiple types of relationships between tokens simultaneously, then merge the heads back together with a final linear projection.

This is the attention mechanism at the heart of GPT-style transformers; combined with `Embedding.py` (turning tokens into positioned vectors) and `Tokenizer.py` (turning raw text into tokens), these three files cover the full path from raw text to context-aware token representations.
