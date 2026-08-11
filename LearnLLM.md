# Learning LLMs: An End-to-End Visual & Mathematical Walkthrough

This guide traces exactly how a Transformer processes the English phrase **"The bank of England"** step-by-step. We will use real toy vectors, show the exact arithmetic, and explain *why* and *how* every module works.

---

# ─── THE PIPELINE AT A GLANCE ───

Here is the journey of the input phrase **"The bank of England"** as it flows through the model:

```
        Raw Input: "The bank of England"
                        │
                        ▼
  Step 1: [ Tokenizer ] ─────────────> Splits text into IDs: [101, 204, 105, 308]
                        │
                        ▼
  Step 2: [ Embedding Layer ] ───────> Translates IDs to raw vectors
                        │
                        ▼
  Step 3: [ Positional Encoding ] ───> Adds wave offset (order info)
                        │
  ======================▼====================== RESIDUAL STREAM BEGINS
  Step 4: [ Transformer Block 1 ]
          ├── Pre-LN LayerNorm ──────> Stabilizes coordinates
          ├── Causal Self-Attention ─> Directs words to look backward for context
          ├── Residual Addition 1 ───> Updates stream: x = x + Attention(x)
          ├── Pre-LN LayerNorm ──────> Stabilizes updated coordinates
          ├── Feed-Forward (MLP) ────> Applies factual and grammatical updates
          └── Residual Addition 2 ───> Updates stream: x = x + MLP(x)
  ======================│======================
                        ▼
  Step 5: [ LM Head & Softmax ] ─────> Turns final vectors into word probabilities
                        │
                        ▼
        Next Word Output: "is" (as in "The bank of England is...")
```

---

# ─── STEP-BY-STEP TRACE: "The bank of England" ───

To make the math simple, we will use a small vector dimension of **$d_{model} = 3$**.

## Step 1: Tokenization
Computers cannot read characters. The tokenizer breaks down the text and maps each word to a unique integer ID from a vocabulary list.

* Input: `"The bank of England"`
* Splitting: `["The", "bank", "of", "England"]`
* Vocabulary IDs:
  ```
  "The"     -> 101
  "bank"    -> 204
  "of"      -> 105
  "England" -> 308
  ```

---

## Step 2: Word Embeddings
We translate the integer IDs into coordinates in 3-dimensional space. Words with similar meanings are placed closer together in this space.

* **"The"**     (ID 101) $\rightarrow$ `[0.1,  0.2, 0.3]`
* **"bank"**    (ID 204) $\rightarrow$ `[1.0, -0.4, 3.0]` (Uncertain meaning: River bank or financial bank?)
* **"of"**      (ID 105) $\rightarrow$ `[0.5,  0.8, -0.1]`
* **"England"** (ID 308) $\rightarrow$ `[2.0,  1.0, 0.5]`

---

## Step 3: Positional Encoding (Adding Order)
Self-attention compares all words at once and has no concept of sequence order. Without positional information, `"bank of England"` and `"England of bank"` would yield identical vectors. We add a sine/cosine wave pattern to the coordinates to represent word position:

* **Wave offset at Position 0 ("The")**: `[0.0, 1.0, 0.0]`
* **Wave offset at Position 1 ("bank")**: `[0.8, 0.5, -0.2]`
* **Wave offset at Position 2 ("of")**: `[-0.5, 0.3, 0.7]`
* **Wave offset at Position 3 ("England")**: `[0.1, -0.9, 0.4]`

Let's calculate the starting coordinates of the **Residual Stream** by adding these offsets element-wise:

```
  "The"     = [0.1,  0.2, 0.3] + [ 0.0,  1.0,  0.0] = [0.1,  1.2, 0.3]
  "bank"    = [1.0, -0.4, 3.0] + [ 0.8,  0.5, -0.2] = [1.8,  0.1, 2.8]
  "of"      = [0.5,  0.8, -0.1] + [-0.5,  0.3,  0.7] = [0.0,  1.1, 0.6]
  "England" = [2.0,  1.0, 0.5] + [ 0.1, -0.9,  0.4] = [2.1,  0.1, 0.9]
```
These summed vectors enter the **Residual Stream**.

---

## Step 4: The Transformer Block (Layer 1)

Let's trace how the vector for **"bank"** (`[1.8, 0.1, 2.8]`) is processed through Layer 1.

### 4.1 Pre-Layer Normalization
Before the vector enters the Self-Attention module, it is normalized to keep the math stable.

1. **Calculate the Average (Mean) of the coordinates of "bank"**:
   $$\mu = \frac{1.8 + 0.1 + 2.8}{3} = \frac{4.7}{3} \approx 1.57$$
2. **Calculate the Variance (how spread out the coordinates are)**:
   $$\sigma^2 = \frac{(1.8 - 1.57)^2 + (0.1 - 1.57)^2 + (2.8 - 1.57)^2}{3} \approx \frac{0.05 + 2.16 + 1.51}{3} \approx 1.24$$
   $$\text{Standard Deviation } (\sigma) = \sqrt{1.24} \approx 1.11$$
3. **Rescale the values**: Subtract the mean and divide by the standard deviation.
   * Coordinate 0: $\frac{1.8 - 1.57}{1.11} \approx 0.21$
   * Coordinate 1: $\frac{0.1 - 1.57}{1.11} \approx -1.32$
   * Coordinate 2: $\frac{2.8 - 1.57}{1.11} \approx 1.11$

* **Normalized Vector for "bank"**: `[0.21, -1.32, 1.11]`

> [!NOTE]
> The original vector `[1.8, 0.1, 2.8]` is stored safely on the side to be used in the residual shortcut addition later.

---

### 4.2 Causal Self-Attention
The normalized vector `[0.21, -1.32, 1.11]` enters the attention block to gather context from surrounding words.

1. **Create Query, Key, and Value**: We project the vector using three separate learned linear matrices:
   * **Query ($Q$)**: *"What context am I looking for?"* $\rightarrow$ `[0.5, 0.1, 0.8]`
   * **Key ($K$)**: *"What attributes do I have?"* $\rightarrow$ `[0.2, -0.6, 0.4]`
   * **Value ($V$)**: *"What information do I contain?"* $\rightarrow$ `[1.0, 1.0, 0.0]`

2. **Compute Similarity Match**: The Query of `"bank"` is multiplied with the Keys of all preceding words.
   In a causal decoder model (like GPT), a word **cannot** look forward into the future. 
   Therefore, `"bank"` (Position 1) can only attend to `"The"` (Position 0) and itself `"bank"` (Position 1). It is blocked from looking at `"of"` or `"England"`.

   Let's assume the dot product calculation yields these raw matching scores:
   * Match score with `"The"` = $0.3$
   * Match score with `"bank"` = $1.2$
   * Match score with `"of"` = $-\infty$ (Causal masked out)
   * Match score with `"England"` = $-\infty$ (Causal masked out)

3. **Softmax**: Convert these scores into percentages that sum to 100%.
   * Attention Weight to `"The"` = **20%**
   * Attention Weight to `"bank"` = **80%**
   * Attention Weight to `"of"` and `"England"` = **0%**

4. **Aggregate Values**: Multiply these percentages by the Value vectors of the words:
   $$\text{Attention Output } f(x) = (0.2 \times V_{\text{The}}) + (0.8 \times V_{\text{bank}})$$
   Let's say this resulting update vector is:
   $$f(x) = [0.1, 1.8, -0.3]$$
   
   This update vector $f(x)$ contains new contextual details: **"Because 'The' is in front of me, I am likely a noun."**

---

### 4.3 Residual Addition 1 (Attention Shortcut)
Instead of replacing our original stream vector with the attention output, we add them together:

$$\text{Updated Stream Vector} = \text{Original Stream Vector } (x) + \text{Attention Output } (f(x))$$
$$\text{Updated Stream Vector} = [1.8, 0.1, 2.8] + [0.1, 1.8, -0.3] = [1.9, 1.9, 2.5]$$

> [!IMPORTANT]
> **Why do we do this?** 
> The original meaning of the word `"bank"` (river/finance) was `[1.8, 0.1, 2.8]`. Attention added context `[0.1, 1.8, -0.3]`. By adding them, we update the vector representation incrementally without losing or overwriting the original word's identity.

---

### 4.4 Feed-Forward Network (MLP)
The updated vector `[1.9, 1.9, 2.5]` travels down the stream and reaches the Feed-Forward Network.

1. **Pre-Normalization**: The vector is normalized again using LayerNorm to keep the scale balanced.
   * Normalized Vector: `[-0.6, -0.6, 1.2]`
2. **Factual Lookup**: The normalized vector enters the MLP. While attention handles relationships between words, the MLP acts as a factual encyclopedia. It looks at the word vector in isolation to check for grammatical rules and facts.
   * **The MLP detects**: *"This token is the noun 'bank' with 'The' in front of it."*
   * **Factual Update $f(x)$**: The MLP generates a knowledge update vector:
     $$f(x) = [0.4, -0.3, 0.9]$$

---

### 4.5 Residual Addition 2 (MLP Shortcut)
We add the factual lookup update back to our stream vector:

$$\text{Final Layer Output} = \text{Updated Stream Vector} + \text{MLP Output}$$
$$\text{Final Layer Output} = [1.9, 1.9, 2.5] + [0.4, -0.3, 0.9] = [2.3, 1.6, 3.4]$$

This finalized representation vector `[2.3, 1.6, 3.4]` represents the word `"bank"` within the context of `"The bank"`. It is now ready to flow into Layer 2.

---

## Step 5: LM Head & Softmax (Next Word Prediction)
After passing through all layers, the final vector for the last input token `"England"` (index 3) is extracted.

1. **Context Extraction**: The vector at position 3 has collected context from all previous positions (`"The"`, `"bank"`, `"of"`, and `"England"`). It represents the sentence: *"The financial institution located in England."*
2. **Projection (LM Head)**: The 3-dimensional vector is projected through a linear matrix to output scores for every word in the vocabulary (logits).
   Let's assume our vocabulary has 4 words: `["is", "river", "money", "runs"]`.
   * Score for `"is"` = $8.2$
   * Score for `"river"` = $0.1$
   * Score for `"money"` = $1.5$
   * Score for `"runs"` = $-2.0$
3. **Softmax**: Convert scores into probabilities:
   * `"is"` $\rightarrow$ **92%**
   * `"money"` $\rightarrow$ **7%**
   * `"river"` $\rightarrow$ **1%**
   * `"runs"` $\rightarrow$ **0%**

The model samples the highest probability word: **"is"**.
The next input prompt becomes: `"The bank of England is"`. The process repeats to predict the word after `"is"`.

---

# ─── MODERN LLM UPGRADES (LLaMA/Mistral Style) ───

Standard Transformers (like GPT-2) have evolved. Modern architectures like LLaMA, Mistral, and Gemma introduce optimized components to make training more stable, computation faster, and memory usage during inference much lower. Let's explore these upgrades step-by-step.

---

## 1. RMSNorm (Root Mean Square Normalization)

**Why it's used:** In standard LayerNorm (Step 4.1), we compute the mean ($\mu$) and variance ($\sigma^2$) to shift and rescale the vector. RMSNorm simplifies this by assuming that centering (subtracting the mean) is not necessary. It only scales by the Root Mean Square (RMS). This is computationally cheaper ($7\%$ to $10\%$ faster) while achieving identical training stability.

### Mathematical Formula
For a vector $x$:
$$\text{RMSNorm}(x)_i = \frac{x_i}{\text{RMS}(x)} \gamma_i$$
Where the RMS is:
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}$$
And $\gamma$ is a learnable scaling parameter (initialized to 1).

### Toy Calculation Trace
Let's normalize the vector for `"bank"` (`[1.8, 0.1, 2.8]`) with $d = 3$ and $\epsilon = 10^{-5}$:

1. **Calculate Mean of Squares**:
   $$\text{Mean of Squares} = \frac{1.8^2 + 0.1^2 + 2.8^2}{3} = \frac{3.24 + 0.01 + 7.84}{3} = \frac{11.09}{3} \approx 3.6967$$
2. **Calculate RMS**:
   $$\text{RMS}(x) = \sqrt{3.6967 + 10^{-5}} \approx \sqrt{3.69671} \approx 1.9227$$
3. **Rescale coordinates**:
   * Coordinate 0: $\frac{1.8}{1.9227} \approx 0.936$
   * Coordinate 1: $\frac{0.1}{1.9227} \approx 0.052$
   * Coordinate 2: $\frac{2.8}{1.9227} \approx 1.456$
   
* **RMSNorm Output**: `[0.936, 0.052, 1.456]` (assuming $\gamma = [1.0, 1.0, 1.0]$)

### PyTorch Code
```python
class RMSNorm(nn.Module):
    """A faster alternative to LayerNorm that only scales by the RMS of activations."""
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        # Calculate RMS along the last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma
```

---

## 2. SwiGLU Activations (Swish Gated Linear Unit)

**Why it's used:** In standard Feed-Forward Networks (Step 4.4), we use a ReLU or GELU activation function. SwiGLU replaces this with a gated structure. It computes two projections in parallel—one is activated with the Swish (SiLU) function and then multiplied element-wise by the other (the gate). This gating mechanism allows the network to dynamically filter information, leading to significantly better convergence.

### Mathematical Formula
$$\text{SwiGLU}(x) = \left(\text{Swish}(x W_g) \otimes (x W_u)\right) W_d$$
Where:
- $\text{Swish}(y) = y \cdot \text{sigmoid}(y) = \frac{y}{1 + e^{-y}}$ (also known as SiLU)
- $W_g$ is the gate projection weight matrix
- $W_u$ is the up projection weight matrix
- $W_d$ is the down projection weight matrix
- $\otimes$ is element-wise multiplication (Hadamard product)

### Toy Calculation Trace
Let input $x = [1.0, -0.5, 2.0]$. Suppose we project $x$ using $W_g$ and $W_u$ to get two vectors:
- Gate vector ($x W_g$): `[1.0, 2.0, -1.0]`
- Up-projected vector ($x W_u$): `[0.5, 1.5, 3.0]`

1. **Apply Swish to Gate**:
   * $\text{Swish}(1.0) = 1.0 \times \text{sigmoid}(1.0) \approx 1.0 \times 0.731 = 0.731$
   * $\text{Swish}(2.0) = 2.0 \times \text{sigmoid}(2.0) \approx 2.0 \times 0.881 = 1.762$
   * $\text{Swish}(-1.0) = -1.0 \times \text{sigmoid}(-1.0) \approx -1.0 \times 0.269 = -0.269$
   * Activated Gate: `[0.731, 1.762, -0.269]`
2. **Element-wise Multiplication ($\otimes$)**:
   $$\text{Gated output} = [0.731 \times 0.5, \ 1.762 \times 1.5, \ -0.269 \times 3.0] = [0.366, 2.643, -0.807]$$
3. **Down Projection**: This gated output vector is finally projected back to $d_{model}$ using the $W_d$ matrix.

### PyTorch Code
```python
class SwiGLUFeedForward(nn.Module):
    """Replaces standard MLP with a gated linear unit utilizing Swish (SiLU)."""
    def __init__(self, embed_dim, d_ff):
        super().__init__()
        self.w_g = nn.Linear(embed_dim, d_ff, bias=False)  # Gate projection
        self.w_u = nn.Linear(embed_dim, d_ff, bias=False)  # Up projection
        self.w_d = nn.Linear(d_ff, embed_dim, bias=False)  # Down projection
        self.silu = nn.SiLU()

    def forward(self, x):
        # Element-wise gate-multiplier logic
        return self.w_d(self.silu(self.w_g(x)) * self.w_u(x))
```

---

## 3. Grouped Query Attention (GQA)

**Why it's used:** Multi-Head Attention (MHA) creates separate Key ($K$) and Value ($V$) representations for each head. During generation (autoregressive decoding), saving these is known as the **KV Cache**, which grows extremely large and bottlenecks memory bandwidth. 
* **MQA (Multi-Query Attention)** uses a single $K$ and $V$ head shared across all $Q$ heads. This is fast but hurts model capacity.
* **GQA (Grouped Query Attention)** groups Query heads, and each group shares a single $K$ and $V$ head. This is the optimal trade-off: high performance with a fraction of the KV cache memory footprint.

```
Multi-Head (MHA)            Grouped Query (GQA)            Multi-Query (MQA)
   (1:1 KV heads)              (Grouped KV heads)            (Single shared KV)

 Q Q Q Q  K K K K            Q Q Q Q   K    K             Q Q Q Q      K
 │ │ │ │  │ │ │ │            ├───┤ ├───┤   │    │             ├────────┤  │
 ▼ ▼ ▼ ▼  ▼ ▼ ▼ ▼            ▼ ▼ ▼ ▼   ▼    ▼             ▼ ▼ ▼ ▼      ▼
 [Head 0] [Head 3]           [ Group 0 ] [ Group 1]       [ Shared KV Head ]
```

### Explanation and Grouping Mechanics
Let a model have $H_q = 8$ Query heads and $H_{kv} = 2$ Key-Value heads:
1. The 8 Query heads are divided into 2 groups (each containing $8 / 2 = 4$ Query heads).
2. Group 0 queries ($Q_0, Q_1, Q_2, Q_3$) share and attend to Key-Value head 0 ($K_0, V_0$).
3. Group 1 queries ($Q_4, Q_5, Q_6, Q_7$) share and attend to Key-Value head 1 ($K_1, V_1$).
4. The KV cache size is reduced by $75\%$ (from 8 down to 2 heads), enabling larger context windows and higher decoding speeds.

### PyTorch Code
```python
class GroupedQueryAttention(nn.Module):
    """Splits queries into groups where each group shares a single K/V head."""
    def __init__(self, embed_dim, num_q_heads, num_kv_heads):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        
        self.q_proj = nn.Linear(embed_dim, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape to [B, T, Heads, HeadDim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose to [B, Heads, T, HeadDim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Expand KV heads to match query heads in each group
        # repeat_interleave maps: [k0, k1] -> [k0, k0, k0, k0, k1, k1, k1, k1]
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Compute standard dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads back together
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(context)
```

---

# ─── COMPLETE RUNNABLE IMPLEMENTATION ───

You can run this complete character-level GPT script to see this entire tokenization, vector projection, block processing, and next-character prediction loop in action:

```python
import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    """Rescales token vectors to stabilize internal activations."""
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * ((x - mean) / torch.sqrt(var + self.eps)) + self.beta


class MultiHeadedAttention(nn.Module):
    """Projects tokens into Q, K, V to calculate attention matches."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_proj(context)


class PositionwiseFeedForward(nn.Module):
    """Processes each token vector individually (factual knowledge block)."""
    def __init__(self, embed_dim, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(embed_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.relu(self.w_1(x)))


class GPTBlock(nn.Module):
    """A single Transformer layer (Attention + MLP + residual additions)."""
    def __init__(self, embed_dim, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadedAttention(embed_dim, num_heads)
        self.feed_forward = PositionwiseFeedForward(embed_dim, d_ff)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x, mask):
        # 1. Attention + Residual skip
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        # 2. MLP + Residual skip
        x = x + self.feed_forward(self.norm2(x))
        return x


class SimpleCharTokenizer:
    """A character-level tokenizer for toy text generation."""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, string):
        return [self.char_to_id[ch] for ch in string]
        
    def decode(self, ids):
        return "".join([self.id_to_char[i] for i in ids])


class GPT(nn.Module):
    """A stacked Decoder-only LLM."""
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, d_ff=256, num_layers=2, max_seq_len=128):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([GPTBlock(embed_dim, num_heads, d_ff) for _ in range(num_layers)])
        self.norm = LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(0, seq_len, device=token_ids.device)
        x = self.token_embeddings(token_ids) + self.pos_embeddings(positions)
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(token_ids.device)
        for block in self.blocks:
            x = block(x, causal_mask)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


if __name__ == "__main__":
    # Create simple training corpus
    corpus = "The bank of England is a financial institution."
    tokenizer = SimpleCharTokenizer(corpus)
    
    print("--- 1. INITIALIZING TOKENIZER ---")
    print("Corpus:", corpus)
    print("Vocab size:", tokenizer.vocab_size)
    
    # Input phrase
    prompt = "The bank"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    print(f"\nPrompt: '{prompt}'")
    print(f"Tokenized input IDs: {input_ids.tolist()}")
    
    # Initialize the GPT model
    model = GPT(vocab_size=tokenizer.vocab_size)
    
    # Forward pass
    logits = model(input_ids)
    print("\n--- 2. RUNNING FORWARD PASS ---")
    print(f"Logits shape: {logits.shape} (Expected: [Batch=1, SeqLen=8, Vocab={tokenizer.vocab_size}])")
    
    # Pull next character prediction
    next_token_logits = logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    next_id = torch.argmax(probs).item()
    next_char = tokenizer.id_to_char[next_id]
    print(f"Predicted next character: '{next_char}'")
    
    # Autoregressive generation loop
    print("\n--- 3. AUTOREGRESSIVE GENERATION LOOP ---")
    curr_ids = input_ids
    generated = prompt
    for _ in range(12):
        logits = model(curr_ids)
        next_id = torch.argmax(logits[0, -1, :]).item()
        next_char = tokenizer.id_to_char[next_id]
        generated += next_char
        curr_ids = torch.cat([curr_ids, torch.tensor([[next_id]])], dim=1)
        
    print(f"Generated text: '{generated}'")
    print("\nVerification complete!")
```