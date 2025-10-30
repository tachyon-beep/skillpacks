---
name: graph-neural-networks-basics
description: Graph Neural Networks: message passing, GCN, GraphSAGE, GAT architectures and selection
dependencies:
  - using-neural-architectures
related:
  - transformer-architecture-deepdive (attention mechanism)
  - cnn-families-and-selection (grid structure)
---

# Graph Neural Networks Basics

## When to Use This Skill

Use this skill when you need to:
- ✅ Work with graph-structured data (molecules, social networks, citations)
- ✅ Understand why CNN/RNN don't work on graphs
- ✅ Learn message passing framework
- ✅ Choose between GCN, GraphSAGE, GAT
- ✅ Decide if GNN is appropriate (vs simple model)
- ✅ Implement permutation-invariant aggregations

**Do NOT use this skill for:**
- ❌ Sequential data (use RNN/Transformer)
- ❌ Grid data (use CNN)
- ❌ High-level architecture selection (use `using-neural-architectures`)

---

## Core Principle

**Graphs have irregular structure.** CNN (grid) and RNN (sequence) don't work.

**GNN solution:** Message passing
- Nodes aggregate information from neighbors
- Multiple layers = multi-hop neighborhoods
- Permutation invariant (order doesn't matter)

**Critical question:** Does graph structure actually help? (Test: Compare with/without edges)

---

## Part 1: Why GNN (Not CNN/RNN)

### Problem: Graph Structure

**Graph components:**
- **Nodes**: Entities (atoms, users, papers)
- **Edges**: Relationships (bonds, friendships, citations)
- **Features**: Node/edge attributes

**Key property:** Irregular structure
- Each node has variable number of neighbors
- No fixed spatial arrangement
- Permutation invariant (node order doesn't matter)

### Why CNN Doesn't Work

**CNN assumption:** Regular grid structure

**Example:** Image (2D grid)
```
Every pixel has exactly 8 neighbors:
[■][■][■]
[■][X][■]  ← Center pixel has 8 neighbors (fixed!)
[■][■][■]

CNN kernel: 3×3 (fixed size, fixed positions)
```

**Graph reality:** Irregular neighborhoods
```
Node A: 2 neighbors (H, C)
Node B: 4 neighbors (C, C, C, H)
Node C: 1 neighbor (H)

No fixed kernel size or position!
```

**CNN limitations:**
- Requires fixed-size neighborhoods → Graphs have variable-size
- Assumes spatial locality → Graphs have arbitrary connectivity
- Depends on node ordering → Should be permutation invariant

### Why RNN Doesn't Work

**RNN assumption:** Sequential structure

**Example:** Text (1D sequence)
```
"The cat sat" → [The] → [cat] → [sat]
Clear sequential order, temporal dependencies
```

**Graph reality:** No inherent sequence
```
Social network:
A — B — C
|       |
D ——————E

What's the "sequence"? A→B→C? A→D→E? No natural ordering!
```

**RNN limitations:**
- Requires sequential order → Graphs have no natural order
- Processes one element at a time → Graphs have parallel connections
- Order-dependent → Should be permutation invariant

### GNN Solution

**Key innovation:** Message passing on graph structure
- Operate directly on nodes and edges
- Variable-size neighborhoods (handled naturally)
- Permutation invariant aggregations

---

## Part 2: Message Passing Framework

### Core Mechanism

**Message passing in 3 steps:**

**1. Aggregate neighbor messages**
```python
# Node i aggregates from neighbors N(i)
messages = [h_j for j in neighbors(i)]
aggregated = aggregate(messages)  # e.g., mean, sum, max
```

**2. Update node representation**
```python
# Combine own features with aggregated messages
h_i_new = update(h_i_old, aggregated)  # e.g., neural network
```

**3. Repeat for L layers**
- Layer 1: Node sees 1-hop neighbors
- Layer 2: Node sees 2-hop neighbors
- Layer L: Node sees L-hop neighborhood

### Concrete Example: Social Network

**Task:** Predict user interests

**Graph:**
```
     B (sports)
     |
A ---+--- C (cooking)
     |
     D (music)
```

**Layer 1: 1-hop neighbors**
```python
# Node A aggregates from direct friends
h_A_layer1 = update(
    h_A,
    aggregate([h_B, h_C, h_D])
)
# Now h_A includes friend interests
```

**Layer 2: 2-hop neighbors (friends of friends)**
```python
# B's friends: E, F
# C's friends: G, H
# D's friends: I

h_A_layer2 = update(
    h_A_layer1,
    aggregate([h_B', h_C', h_D'])  # h_B' includes E, F
)
# Now h_A includes friends-of-friends!
```

**Key insight:** More layers = larger receptive field (L-hop neighborhood)

### Permutation Invariance

**Critical property:** Same graph → same output (regardless of node ordering)

**Example:**
```python
Graph: A-B, B-C

Node list 1: [A, B, C]
Node list 2: [C, B, A]

Output MUST be identical! (Same graph, different ordering)
```

**Invariant aggregations:**
- ✅ Mean: `mean([1, 2, 3]) == mean([3, 2, 1])`
- ✅ Sum: `sum([1, 2, 3]) == sum([3, 2, 1])`
- ✅ Max: `max([1, 2, 3]) == max([3, 2, 1])`

**NOT invariant:**
- ❌ LSTM: `LSTM([1, 2, 3]) != LSTM([3, 2, 1])`
- ❌ Concatenate: `[1, 2, 3] != [3, 2, 1]`

**Implementation:**
```python
# CORRECT: Permutation invariant
def aggregate(neighbor_features):
    return torch.mean(neighbor_features, dim=0)

# WRONG: Order-dependent!
def aggregate(neighbor_features):
    return LSTM(neighbor_features)  # Output depends on order
```

---

## Part 3: GNN Architectures

### Architecture 1: GCN (Graph Convolutional Network)

**Key idea:** Spectral convolution on graphs (simplified)

**Formula:**
```python
h_i^(l+1) = σ(∑_{j∈N(i)} W^(l) h_j^(l) / √(|N(i)| |N(j)|))

# Normalize by degree (√(deg(i) * deg(j)))
```

**Aggregation:** Weighted mean (degree-normalized)

**Properties:**
- Transductive (needs full graph at training)
- Computationally efficient
- Good baseline

**When to use:**
- Full graph available at training time
- Starting point (simplest GNN)
- Small to medium graphs

**Implementation:**
```python
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: Node features (N, in_channels)
        # edge_index: Graph connectivity (2, E)

        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)

        return x
```

### Architecture 2: GraphSAGE

**Key idea:** Sample and aggregate (inductive learning)

**Formula:**
```python
# Sample fixed-size neighborhood
neighbors_sampled = sample(neighbors(i), k=10)

# Aggregate
h_N = aggregate({h_j for j in neighbors_sampled})

# Concatenate and transform
h_i^(l+1) = σ(W^(l) [h_i^(l); h_N])
```

**Aggregation:** Mean, max, or LSTM (but mean/max preferred for invariance)

**Key innovation:** Sampling
- Sample fixed number of neighbors (e.g., 10)
- Makes computation tractable for large graphs
- Enables inductive learning (generalizes to unseen nodes)

**When to use:**
- Large graphs (millions of nodes)
- Need inductive capability (new nodes appear)
- Training on subset, testing on full graph

**Implementation:**
```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

### Architecture 3: GAT (Graph Attention Network)

**Key idea:** Learn attention weights for neighbors

**Formula:**
```python
# Attention scores
α_ij = attention(h_i, h_j)  # How important is neighbor j to node i?

# Normalize (softmax)
α_ij = softmax_j(α_ij)

# Weighted aggregation
h_i^(l+1) = σ(∑_{j∈N(i)} α_ij W h_j^(l))
```

**Key innovation:** Learned neighbor importance
- Not all neighbors equally important
- Attention mechanism decides weights
- Multi-head attention (like Transformer)

**When to use:**
- Neighbors have varying importance
- Need interpretability (attention weights)
- Have sufficient data (attention needs more data to learn)

**Implementation:**
```python
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

### Architecture Comparison

| Feature | GCN | GraphSAGE | GAT |
|---------|-----|-----------|-----|
| Aggregation | Degree-weighted mean | Mean/max/LSTM | Attention-weighted |
| Neighbor weighting | Fixed (by degree) | Equal | Learned |
| Inductive | No | Yes | Yes |
| Scalability | Medium | High (sampling) | Medium |
| Interpretability | Low | Low | High (attention) |
| Complexity | Low | Medium | High |

### Decision Tree

```
Starting out / Small graph:
→ GCN (simplest baseline)

Large graph (millions of nodes):
→ GraphSAGE (sampling enables scalability)

Need inductive learning (new nodes):
→ GraphSAGE or GAT

Neighbors have different importance:
→ GAT (attention learns importance)

Need interpretability:
→ GAT (attention weights explain predictions)

Production deployment:
→ GraphSAGE (most robust and scalable)
```

---

## Part 4: When NOT to Use GNN

### Critical Question

**Does graph structure actually help?**

**Test:** Compare model with and without edges
```python
# Baseline: MLP on node features only
mlp_accuracy = train_mlp(node_features, labels)

# GNN: Use node features + graph structure
gnn_accuracy = train_gnn(node_features, edges, labels)

# Decision:
if gnn_accuracy - mlp_accuracy < 2%:
    print("Graph structure doesn't help much")
    print("Use simpler model (MLP or XGBoost)")
else:
    print("Graph structure adds value")
    print("Use GNN")
```

### Scenarios Where GNN Doesn't Help

**1. Node features dominate**
```
User churn prediction:
- Node features: Usage hours, demographics, subscription → Highly predictive
- Graph edges: Sparse user interactions → Weak signal
- Result: MLP 85%, GNN 86% (not worth complexity!)
```

**2. Sparse graphs**
```
Graph with 1000 nodes, 100 edges (0.01% density):
- Most nodes have 0-1 neighbors
- No information to aggregate
- GNN reduces to MLP
```

**3. Random graph structure**
```
If edges are random (no homophily):
- Neighbor labels uncorrelated
- Aggregation adds noise
- Simple model better
```

### When GNN DOES Help

✅ **Molecular property prediction**
- Structure is PRIMARY signal
- Atom types + bonds determine properties
- GNN: Huge improvement over fingerprints

✅ **Citation networks**
- Paper quality correlated with neighbors
- "You are what you cite"
- Clear homophily

✅ **Social recommendation**
- Friends have similar preferences
- Graph structure informative
- GNN: Moderate to large improvement

✅ **Knowledge graphs**
- Entities connected by relations
- Multi-hop reasoning valuable
- GNN captures complex patterns

### Decision Framework

```
1. Start simple:
   - Try MLP or XGBoost on node features
   - Establish baseline performance

2. Check graph structure value:
   - Does edge information correlate with target?
   - Is there homophily (similar nodes connected)?
   - Test: Remove edges, compare performance

3. Use GNN if:
   - Graph structure adds >2-5% accuracy
   - Structure is interpretable (not random)
   - Have enough nodes for GNN to learn

4. Stick with simple if:
   - Node features alone sufficient
   - Graph structure weak/random
   - Small dataset (< 1000 nodes)
```

---

## Part 5: Practical Implementation

### Using PyTorch Geometric

**Installation:**
```bash
pip install torch-geometric
```

**Basic workflow:**
```python
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# 1. Create graph data
x = torch.tensor([[feature1], [feature2], ...])  # Node features
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Edges (COO format)
y = torch.tensor([label1, label2, ...])  # Node labels

data = Data(x=x, edge_index=edge_index, y=y)

# 2. Define model
class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(in_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 3. Train
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
```

### Edge Index Format

**COO (Coordinate) format:**
```python
# Edge list: (0→1), (1→2), (2→0)
edge_index = torch.tensor([
    [0, 1, 2],  # Source nodes
    [1, 2, 0]   # Target nodes
])

# For undirected graph, include both directions:
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 0],  # Source
    [1, 0, 2, 1, 0, 2]   # Target
])
```

### Mini-batching Graphs

**Problem:** Graphs have different sizes

**Solution:** Batch graphs as one large disconnected graph
```python
from torch_geometric.data import DataLoader

# Create dataset
dataset = [Data(...), Data(...), ...]

# DataLoader handles batching
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # batch contains multiple graphs as one large graph
    # batch.batch: Indicator which nodes belong to which graph
    out = model(batch.x, batch.edge_index)
```

---

## Part 6: Common Mistakes

### Mistake 1: LSTM Aggregation

**Symptom:** Different outputs for same graph with reordered nodes
**Fix:** Use mean/sum/max aggregation (permutation invariant)

### Mistake 2: Forgetting Edge Direction

**Symptom:** Information flows wrong way
**Fix:** For undirected graphs, add edges in both directions

### Mistake 3: Too Many Layers

**Symptom:** Performance degrades, over-smoothing
**Fix:** Use 2-3 layers (most graphs have small diameter)
**Explanation:** Too many layers → all nodes converge to same representation

### Mistake 4: Not Testing Simple Baseline

**Symptom:** Complex GNN with minimal improvement
**Fix:** Always test MLP on node features first

### Mistake 5: Using GNN on Euclidean Data

**Symptom:** CNN/RNN would work better
**Fix:** Use GNN only for irregular graph structure (not grids/sequences)

---

## Part 7: Summary

### Quick Reference

**When to use GNN:**
- Graph-structured data (molecules, social networks, citations)
- Irregular neighborhoods (not grid/sequence)
- Graph structure informative (test this!)

**Architecture selection:**
```
Start: GCN (simplest)
Large graph: GraphSAGE (scalable)
Inductive learning: GraphSAGE or GAT
Neighbor importance: GAT (attention)
```

**Key principles:**
- Message passing: Aggregate neighbors + Update node
- Permutation invariance: Use mean/sum/max (not LSTM)
- Test baseline: MLP first, GNN if structure helps
- Layers: 2-3 sufficient (more = over-smoothing)

**Implementation:**
- PyTorch Geometric: Standard library
- COO format: Edge index as 2×E tensor
- Batching: Merge graphs into one large graph

---

## Next Steps

After mastering this skill:
- `transformer-architecture-deepdive`: Understand attention (used in GAT)
- `architecture-design-principles`: Design principles for graph architectures
- Advanced GNNs: Graph Transformers, Equivariant GNNs

**Remember:** Not all graph data needs GNN. Test if graph structure actually helps! (Compare with MLP baseline)
