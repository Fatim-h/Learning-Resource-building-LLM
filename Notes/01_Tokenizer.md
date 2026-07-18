# Notes: `Tokenizer.py` — Turning Text into Token IDs

**Goal of this file:** show the code evolution from a naive regex splitter to a Byte Pair Encoding (BPE) tokenizer, and finally build a sliding-window `Dataset`/`DataLoader` that produces (input, target) training pairs.

---

## Step 1 — Load the raw training text

```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
```

- Loads a short story ("The Verdict") as a plain string. This acts as a toy training corpus.
- `len(raw_text)` and a text preview are printed just to sanity-check the load.

## Step 2 — A first, naive tokenizer (regex splitting)

```python
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
```

- `re.split` with a **capturing group** splits the string *and keeps the delimiters* (punctuation, `--`, whitespace) as separate list items.
- The list comprehension throws away empty/whitespace-only fragments.
- Applied to the whole `raw_text`, this produces `preprocessed`, a list of ~thousands of word/punctuation tokens.

**Why this matters:** this is the conceptual core of tokenization — decide the rules for what counts as one "unit" of text.

## Step 3 — Build a vocabulary

```python
all_words = sorted(set(preprocessed))
vocab = {token: integer for integer, token in enumerate(all_words)}
```

- `set()` gets unique tokens, `sorted()` gives them a stable order, then each unique token is assigned an integer ID.
- This `vocab` dict (`token → id`) is the entire "language" the tokenizer can understand.

## Step 4 — `SimpleTokenizerV1`: encode/decode with a fixed vocab

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # remove space before punctuation
        return text
```

- `encode`: text → list of token IDs, using the same splitting rule as before, looked up in `str_to_int`.
- `decode`: IDs → words joined by spaces → regex cleanup to remove the extra space that ends up before punctuation (`"world ."` → `"world."`).
- **Limitation:** if `encode()` sees a word that isn't in `vocab`, it throws a `KeyError`. There's no way to handle unknown words.

## Step 5 — `SimpleTokenizerV2`: adding special tokens

```python
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab2 = {token: integer for integer, token in enumerate(all_tokens)}
```

- Two **special tokens** are added to the vocabulary:
  - `<|endoftext|>` — a separator inserted between independent documents/texts, so the model learns "this is where one document ends and another begins."
  - `<|unk|>` — a fallback for any word not seen in training (out-of-vocabulary).
- `SimpleTokenizerV2.encode` adds one line versus V1:
  ```python
  preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
  ```
  This swaps any unknown word for `<|unk|>` instead of crashing.
- Two texts are joined with `<|endoftext|>` in between to demonstrate the separator in action.

**Takeaway:** real tokenizers need a strategy for (a) unknown words and (b) document boundaries — this is exactly what GPT-style special tokens solve.

## Step 6 — Byte Pair Encoding (BPE) via `tiktoken`

```python
tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
strings = tokenizer.decode(integers)
```

- Word-level vocabularies like V1/V2 don't scale: real language has near-infinite unique words (typos, names, made-up words, other languages).
- **BPE** solves this by building a vocabulary of common *subword* chunks (learned by repeatedly merging the most frequent adjacent byte/character pairs). This means:
  - Common words get their own token.
  - Rare/unknown words get broken into smaller known pieces (even down to individual bytes), so **there is no `<|unk|>` needed** — anything can be represented.
- `tiktoken.get_encoding("gpt2")` loads OpenAI's pretrained GPT-2 BPE tokenizer (vocab size 50,257 — this is why `Embedding.py` later uses `vocab_size = 50257`).
- `allowed_special={"<|endoftext|>"}` tells the encoder to treat that string as one atomic special token rather than splitting it up.

## Step 7 — Sliding-window dataset for next-token prediction

This is the key idea that turns raw text into supervised training data for a language model: **predict the next token given the previous ones.**

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
```

- The entire corpus is BPE-encoded once into `token_ids` (one long list of integers).
- A window of size `max_length` slides across `token_ids`, moving `stride` tokens each step.
- For every window:
  - `input_chunk` = tokens `[i, i+max_length)`
  - `target_chunk` = the **same window shifted one position to the right** — i.e., "what comes next" at every position.
- Example with `max_length=4`: if input is `[The, cat, sat, on]`, target is `[cat, sat, on, the]`. Each position's target is literally the next token — this is what teaches the model to predict the next word.
- `stride` controls overlap:
  - `stride == max_length` → no overlap between chunks (each token used once).
  - `stride < max_length` → overlapping windows (more training samples, more redundancy).

```python
def __len__(self):
    return len(self.input_ids)

def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]
```
Standard PyTorch `Dataset` interface so it can plug into a `DataLoader`.

## Step 8 — `create_dataloader_v1`: wraps it all up

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                          shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             drop_last=drop_last, num_workers=num_workers)
    return dataloader
```

- Ties tokenizer + dataset + PyTorch `DataLoader` together into one convenience function.
- `drop_last=True` discards a final incomplete batch so every batch has the same shape (important for stable, efficient training).

## Step 9 — Example run

```python
dataloader = create_dataloader_v1(txt=raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
```

- Produces a batch of shape `[8, 4]`: 8 samples, each 4 tokens long.
- `inputs` are the token windows; `targets` are those windows shifted by one — exactly the tensors seen at the top of `Embedding.py`.

---

### Summary of what this file demonstrates
1. Naive whitespace/punctuation tokenization → assign each unique word an integer ID.
2. Handle unknowns and document boundaries with special tokens (`<|unk|>`, `<|endoftext|>`).
3. Switch to subword-level BPE (`tiktoken`, GPT-2's 50,257-token vocab) to remove the unknown-word problem entirely and scale to any language/vocabulary.
4. Convert a long stream of token IDs into supervised (input, target) training pairs using a sliding window — this is the actual data the model trains on.
