# Notes: `pytorch.py` — the DL tensor library

**Goal of this file:** learn the PyTorch fundamentals
1. Simplified attention with no trainable weights.
2. Trainable attention (scaled dot-product) using raw parameters, then `nn.Linear`.
3. Causal (masked) attention, so a token can't "see" future tokens.
4. Multi-head attention, running several attention operations in parallel.

## 1 — Installation
There are two versions of PyTorch: a leaner version that only supports CPU computing and a full version that supports both CPU and GPU computing.

```bash
pip install torch
```

Check installation:
```bash
import torch
torch.__version__
```
## 2 — Functions

### Creating tensor
```python
import torch
tensor0d = torch.tensor(1) # 0 dimensional tensor
tensor1d = torch.tensor([1, 2, 3]) #1 dimensional tensor
tensor2d = torch.tensor([[1, 2], [3, 4]]) # 2 dimensional tensor
tensor3d = torch.tensor([[[1, 2], [3, 4]], 
                        [[5, 6], [7, 8]]]) 
                        # 3 dimensional tensor

```
With this example the tensor datatype is integer(64-bit).

```python
floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)
```
This will be `torch.float32`

You can change datatypes:
```python
floatvec = tensor1d.to(torch.float32)
```

### Common Operations
#### .shape

```python
tensor2d = torch.tensor([[1, 2, 3], 
                 [4, 5, 6]]) 

#prints torch.Size([2, 3])
```

#### .reshape, .view
```python
print(tensor2d.reshape(3, 2)) #reshapes the tensor
print(tensor2d.view(3, 2)) #more common way o reshape

"""
output:
tensor([[1, 2],
        [3, 4],
        [5, 6]])
"""
```

#### .transpose
```python
print(tensor2d.T)

"""
output:
tensor([[1, 4],
       [2, 5],
       [3, 6]])]
"""
```

#### .matmul
```python
print(tensor2d.matmul(tensor2d.T))
"""
output:
tensor([[14, 32],
[32, 77]]) 
"""
```

#### autograd
Allows to compute gradients in dynamic computational graphs automatically.

```python
#Logistic Regression Forward Pass
import torch.nn.functional as F

y = torch.tensor([1.0])  #true label        
x1 = torch.tensor([1.1]) #input feature  
w1 = torch.tensor([2.2]) #weight parameter
b = torch.tensor([0.0])  #bias unit
z = x1 * w1 + b          #net input      
a = torch.sigmoid(z)     #activation 
loss = F.binary_cross_entropy(a, y)
```
```mermaid
flowchart LR
    %% Class Definitions for High Visibility
    classDef weight fill:#52b7e8,stroke:#1e1e1e,stroke-width:1.5px,color:#000,font-weight:bold;
    classDef input fill:#a2d071,stroke:#1e1e1e,stroke-width:1.5px,color:#000,font-weight:bold;
    classDef bias fill:#52b7e8,stroke:#1e1e1e,stroke-width:1.5px,color:#000,font-weight:bold;
    classDef label fill:#ef85b1,stroke:#1e1e1e,stroke-width:1.5px,color:#000,font-weight:bold;
    classDef op fill:#ffffff,stroke:#1e1e1e,stroke-width:1.5px,color:#000,font-weight:bold;
    classDef box fill:#ffffff,stroke:#1e1e1e,stroke-width:1.5px,color:#000,font-weight:bold,rx:5,ry:5;
    classDef text fill:none,stroke:none,color:#222,font-weight:bold;

    %% Annotations
    T1["A trainable weight<br>parameter"]
    T2["The input data"]
    T3["An intermediate result in<br>the computation graph"]
    T4["A trainable bias unit"]
    T5["The target label"]

    %% Diagram Nodes
    W["w₁"]:::weight
    X["x₁"]:::input
    MULT(("⊗")):::op
    U["u = w₁ × x₁"]:::box
    B["b"]:::bias
    ADD(("+")):::op
    Z["z = u + b"]:::box
    A["a = σ(z)"]:::box
    LOSS["loss = L(a, y)"]:::box
    Y["y"]:::label

    %% Annotation Links
    T1 -.- W
    T2 -.- X
    T3 -.- U
    T4 -.- B
    T5 -.- Y

    %% Computation Flow Links
    W --> MULT
    X --> MULT
    MULT --> U
    U --> ADD
    B --> ADD
    ADD --> Z
    Z --> A
    A --> LOSS
    Y --> LOSS

    %% Apply Text Classes
    class T1,T2,T3,T4,T5 text;
```

###  A multilayer perceptron with two hidden layers 
```python
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):   
super().__init__()
self.layers = torch.nn.Sequential(
    # 1st hidden layer
    torch.nn.Linear(num_inputs, 30),   
    torch.nn.ReLU(),              
    # 2nd hidden layer
    torch.nn.Linear(30, 20),   
    torch.nn.ReLU(),
    # output layer
    torch.nn.Linear(20, num_outputs),
)
    def forward(self, x):
logits = self.layers(x)
return logits 

# instantiate model (50 input features, 3 target classes)
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=50, num_outputs=3)

# inspect
print("Model Structure:")
print(model)

"""
output:
NeuralNetwork(
  (layers): Sequential(
    (0): Linear(in_features=50, out_features=30, bias=True)
    (1): ReLU()
    (2): Linear(in_features=30, out_features=20, bias=True)
    (3): ReLU()
    (4): Linear(in_features=20, out_features=3, bias=True)
  )
)
"""
# check total trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nTotal trainable parameters:", num_params)
#prints: Total number of trainable model parameters: 2213

print(model.layers[0].weight)
"""
output:
Parameter containing:
tensor([[ 0.1174, -0.1350, -0.1227,  ...,  0.0275, -0.0520, -0.0192],
        [-0.0169,  0.1265,  0.0255,  ..., -0.1247,  0.1191, -0.0698],
        [-0.0973, -0.0974, -0.0739,  ..., -0.0068, -0.0892,  0.1070],
        ...,
        [-0.0681,  0.1058, -0.0315,  ..., -0.1081, -0.0290, -0.1374],
        [-0.0159,  0.0587, -0.0916,  ..., -0.1153,  0.0700,  0.0770],
        [-0.1019,  0.1345, -0.0176,  ...,  0.0114, -0.0559, -0.0088]],
       requires_grad=True)
"""

# forward Pass (single sample batch)
X = torch.rand((1, 50))

# forward pass for training (tracks computational graph grad_fn)
logits = model(X)

# forward pass for inference (saves memory by disabling gradient tracking)
with torch.no_grad():
    probabilities = torch.softmax(model(X), dim=1)

```
```mermaid
flowchart LR
    %% High-contrast class definitions
    classDef input fill:#a2d071,stroke:#1e1e1e,stroke-width:1.5px,color:#000,font-weight:bold;
    classDef layer fill:#52b7e8,stroke:#1e1e1e,stroke-width:1.5px,color:#000,font-weight:bold;
    classDef act fill:#ffffff,stroke:#1e1e1e,stroke-width:1.5px,color:#000,font-weight:bold,rx:5,ry:5;
    classDef output fill:#ef85b1,stroke:#1e1e1e,stroke-width:1.5px,color:#000,font-weight:bold;
    classDef note fill:none,stroke:none,color:#222,font-weight:bold;

    %% Network Layers
    X["Input X<br>(1, 50)"]:::input
    L1["Linear 1<br>(50 → 30)"]:::layer
    R1["ReLU()"]:::act
    L2["Linear 2<br>(30 → 20)"]:::layer
    R2["ReLU()"]:::act
    L3["Linear 3<br>(20 → 3)"]:::layer
    LOGITS["Logits<br>(1, 3)"]:::output
    PROBS["Softmax<br>Probabilities"]:::output

    %% Context annotations
    T1["Features input vector"]
    T2["Hidden Layer 1"]
    T3["Hidden Layer 2"]
    T4["Unnormalized scores"]
    T5["Sum to 1.0 (Inference)"]

    %% Connections
    X --> L1
    L1 --> R1
    R1 --> L2
    L2 --> R2
    R2 --> L3
    L3 --> LOGITS
    LOGITS -. "torch.softmax()" .-> PROBS

    %% Annotation Links
    T1 -.- X
    T2 -.- L1
    T3 -.- L2
    T4 -.- LOGITS
    T5 -.- PROBS

    class T1,T2,T3,T4,T5 note;
```