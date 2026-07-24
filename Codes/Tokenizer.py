#Source for 'the-vedict.txt': https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/the-verdict.txt
with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

#Tokenizer
import re
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
print(result)

#Token the source text
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

#Set tokn ids
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
  print(item)
  if i >= 50:
    break
# ==========================================
# Now everything together(Complete Tkenizer, that makes IDs)
# ==========================================

import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))


#this token also has special tokes(|endoftext|, |unk|)
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab2 = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab2.items()))

class SimpleTokenizerV2:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str = { i:s for s,i in vocab.items()}
  def encode(self, text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
    ids = [self.str_to_int[s] for s in preprocessed]
    return ids
  def decode(self, ids):
    text = " ".join([self.int_to_str[i] for i in ids])
    text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
    return text
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
tokenizer2 = SimpleTokenizerV2(vocab2)
print(tokenizer2.encode(text))
print(tokenizer2.decode(tokenizer2.encode(text)))

# ==========================================
# Now BytePAir Encoding
# ==========================================
#pip install tiktoken

from importlib.metadata import version
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace.")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)


# ==========================================
# Data Sampling(The sliding window)
# ==========================================

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


# 1. Define the Dataset Class
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire raw text corpus into a master list of token IDs
        token_ids = tokenizer.encode(txt)

        # Slide a window across the entire list of token IDs
        for i in range(0, len(token_ids) - max_length, stride):
            # Extract the input chunk based on max_length
            input_chunk = token_ids[i : i + max_length]
            # Extract the target chunk by shifting forward 1 position
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            # Store them as PyTorch tensors
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # Return the total number of chunks sampled from the text
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return a single (input, target) pair of tensors
        return self.input_ids[idx], self.target_ids[idx]

# 2. Define the DataLoader Factory Function

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    # Initialize the target BPE tokenizer (e.g., gpt2)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Initialize our custom sliding window dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Instantiate the PyTorch DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # drop_last=True drops any incomplete trailing batch to minimize training spikes
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

# 3. Example Usage & Testing

if __name__ == "__main__":

    # Instantiate dataloader matching the textbook evaluation parameter setup:
    # batch_size=8, max_length=4, stride=4, shuffle=False
    dataloader = create_dataloader_v1(
        txt=raw_text,
        batch_size=8,
        max_length=4,
        stride=4,
        shuffle=False
    )

    # Fetch and inspect the very first batch
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Inputs Tensor:\n", inputs)
    print("\nTargets Tensor:\n", targets)

# ==========================================
#Next you create the embedding layer from these tokens,
# ==========================================