"""
Simplified Self-Attention: The mechanism works in three distinct phases:

1. Compute Attention Scores (Similarity): dot product of the word vectors. 
   A higher dot product means the two words are highly aligned or relevant to each other.
2. Compute Attention Weights (Normalization): pass the raw scores through 
    a softmax function. This ensures they all add up to 1 for training stability.
3. Compute Context Vectors (Weighted Sum): We multiply each word's vector by its attention 
   weight and sum them up. This creates a new, context-rich vector that blends the words 
   original meaning with the meanings of the words it attends to.
"""
import torch

# 1. Input sentence embeddings 
#(6 tokens, each with a 3-dimensional embedding)
# "Your journey starts with one step"
inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your    (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts  (x^3)
    [0.22, 0.58, 0.33], # with    (x^4)
    [0.77, 0.25, 0.10], # one     (x^5)
    [0.05, 0.80, 0.55]  # step    (x^6)
])

print("Inputs shape:", inputs.shape) # Expected: [6, 3]

# --- STEP 1: Compute Attention Scores ---
attn_scores = inputs @ inputs.T
print("\nAttention Scores Matrix:\n", attn_scores)
print("Attention Scores shape:", attn_scores.shape) # Expected: [6, 6]

# --- STEP 2: Compute Attention Weights ---
attn_weights = torch.softmax(attn_scores, dim=-1)
print("\nAttention Weights (Softmax Normalized):\n", attn_weights)

# Verification step: Check that all rows sum up perfectly to 1
print("All row sums:", attn_weights.sum(dim=-1))

# --- STEP 3: Compute Context Vectors ---
all_context_vecs = attn_weights @ inputs
print("\nAll Context Vectors:\n", all_context_vecs)
print("Context Vectors shape:", all_context_vecs.shape) 
# Expected: [6, 3]

# --- Verification with Individual Calculation (Section 3.3.1) ---
# to show it matches row index 1 of our matrix operation.
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print("\n--- Verification ---")
print("Manually computed 2nd context vector: ", context_vec_2)
print("Matrix-computed 2nd context vector (row 1):", all_context_vecs[1])

"""
With trainable weights(scaled-dot product)
"""
import torch.nn as nn

# 1. Prepare input sentence embeddings 
# "Your journey starts with one step" (6 tokens, 3-dimensional embeddings)
inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your    
    [0.55, 0.87, 0.66], # journey 
    [0.57, 0.85, 0.64], # starts  
    [0.22, 0.58, 0.33], # with    
    [0.77, 0.25, 0.10], # one     
    [0.05, 0.80, 0.55]  # step    
])

d_in = inputs.shape[1]  # Input size: 3
d_out = 2               # Output size: 2

# --- Implementation 1: Using nn.Parameter (SelfAttention_v1) ---
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
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

# --- Implementation 2: Using PyTorch Linear Layers (SelfAttention_v2) ---
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # nn.Linear automatically manages weights (and optionally biases)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

print("--- Version 1 Outputs (Random Weights) ---")
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

print("\n--- Version 2 Outputs (Linear Default Initialization) ---")
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

"""
To make SelfAttention_v1 produce the exact same results as 
SelfAttention_v2, we assign the transposed weights from v2 into v1.
"""
# Create a new v1 instance to copy weights into
sa_v1_aligned = SelfAttention_v1(d_in, d_out)

# Copy the weights from sa_v2 to sa_v1 (transposing them back to d_in x d_out)
# We wrap them in nn.Parameter to preserve requirements
sa_v1_aligned.W_query = nn.Parameter(sa_v2.W_query.weight.T)
sa_v1_aligned.W_key   = nn.Parameter(sa_v2.W_key.weight.T)
sa_v1_aligned.W_value = nn.Parameter(sa_v2.W_value.weight.T)

# Run verification
v1_aligned_output = sa_v1_aligned(inputs)
v2_output = sa_v2(inputs)

print("\n--- Exercise 3.1 Verification ---")
print("Are the aligned output matrices mathematically equal?")
print(torch.allclose(v1_aligned_output, v2_output))
"""
Causal Attention
"""