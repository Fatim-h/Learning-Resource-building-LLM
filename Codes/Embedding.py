# ==========================================
# for simplicity we are using a vocab of 6 words, 
# and embedding size 3(GPT-3 has 12,288 dimensions)
# ==========================================
import torch
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#one hot followed by matrix multiplication
print(embedding_layer.weight)
print("\n")
print(embedding_layer(torch.tensor([3])))

#using the dataloader from Tokenizer.py
"""""""""
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
    )
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
"""""""""
# ==========================================
# Positional Embedding: because the nonsequntial processing of 
# the data removes the concept of position so we create a unique 
# vector for each position (Position 0, Position 1, etc.) and 
# add it to the token's meaning vector.
# ==========================================

# realistic dimensions (GPT-style but downscaled for experimentation)
vocab_size = 50257
output_dim = 256
context_length = 4  # Maximum length of the input text sequence

torch.manual_seed(123)

# Token Embedding Layer
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

inputs = torch.tensor([
    [40,   367,  2885,  1464],
    [1807, 3619,  402,   271],
    [10899, 2138,  257,  7026],
    [15632,  438, 2016,   257],
    [922,  5891, 1576,   438],
    [568,   340,  373,   645],
    [1049, 5975,  284,   502],
    [284,  3285,  326,    11]
])

print("Inputs shape (Batch, Seq_Len):", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("Token Embeddings shape:", token_embeddings.shape)  # Expected: [8, 4, 256]


# Positional Embedding Layer
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# Create a sequence of position indices: tensor([0, 1, 2, 3])
pos_indices = torch.arange(context_length) 

# Generate Positional Embeddings
pos_embeddings = pos_embedding_layer(pos_indices)
print("Positional Embeddings shape:", pos_embeddings.shape)  
# Expected: [4, 256]


# 4. Add together 
input_embeddings = token_embeddings + pos_embeddings

print("\n--- Final Output ---")
print("Final Input Embeddings shape sent to LLM:", input_embeddings.shape)  
# Expected: [8, 4, 256]