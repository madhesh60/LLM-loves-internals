## The full plan — from self-attention to a working ChatGPT-like app

Here's every stage, in order. Each stage builds on the previous one. By the end you have something real to put on your resume.

---

## The full roadmap

```
✅ Done:  Tokenizer
✅ Done:  Embeddings
👉 Now:   Stage 1  — Self Attention
          Stage 2  — Multi-Head Attention
          Stage 3  — Feed Forward block
          Stage 4  — Residual Connections
          Stage 5  — Layer Normalization
          Stage 6  — One full Transformer block (combining 1-5)
          Stage 7  — Stack blocks into a full GPT model (MiniGPT)
          Stage 8  — Training loop
          Stage 9  — Text generation (the "chat" part)
          Stage 10 — FastAPI backend (make it an API)
          Stage 11 — Simple frontend (make it look like ChatGPT)
```

---

## Stage 1 — Self Attention

**What it does in real life:** every word looks at every other word in the sentence and decides how much to care about each one.

"The cat sat because **it** was tired" — the word "it" needs to figure out it means "cat." Self-attention does that.

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Three different "views" of each word
        self.Q = nn.Linear(embed_dim, embed_dim)  # what am I looking for?
        self.K = nn.Linear(embed_dim, embed_dim)  # what do I have?
        self.V = nn.Linear(embed_dim, embed_dim)  # what do I share?

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        # how much should each word pay attention to each other word
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])

        # turn scores into percentages that add to 100%
        weights = torch.softmax(scores, dim=-1)

        # mix word info based on those percentages
        return weights @ v
```

**Resume line:** Implemented scaled dot-product self-attention from scratch in PyTorch.

---

## Stage 2 — Multi-Head Attention

**What it does:** instead of one attention pass, do it 8 times in parallel — each "head" looks for a different kind of pattern (grammar, meaning, who did what, etc). Then combine all 8 results.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttention(embed_dim) for _ in range(num_heads)
        ])
        # combine all heads back into one
        self.output_proj = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x):
        # run all heads and stack results side by side
        head_outputs = [head(x) for head in self.heads]
        combined = torch.cat(head_outputs, dim=-1)
        return self.output_proj(combined)
```

**Resume line:** Built multi-head attention with parallel attention heads and learned output projection.

---

## Stage 3 — Feed Forward Block

**What it does:** after words talk to each other via attention, each word now does its own private thinking — processes its own vector individually.

```python
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # expand
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)   # shrink back
        )

    def forward(self, x):
        return self.net(x)
```

---

## Stage 4 — Residual Connections

**What it does:** As networks get deeper, they suffer from the "vanishing gradient" problem—information and gradients get lost as they pass through many layers. A residual connection (or skip connection) simply adds the input of a layer back to its output. 

Mathematically: 
$$\text{Output} = x + \text{Sublayer}(x)$$

This creates a "highway" for gradients to flow backward directly during training, making it much easier to train deep models without loss of signal.

```python
class ResidualConnection(nn.Module):
    def __init__(self, sublayer: nn.Module):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x):
        # Add the original input x back to the output of the sublayer
        return x + self.sublayer(x)
```

**Resume line:** Implemented residual (skip) connections to enable stable backpropagation and prevent vanishing gradients in deep networks.

---

## Stage 5 — Layer Normalization

**What it does:** Keeps the activation values (numbers) inside the network stable. Without normalization, the values can grow too large (exploding) or too small (vanishing) as they pass through layers, causing training to fail. 

Unlike Batch Normalization (which normalizes across the batch), Layer Normalization normalizes the features of *each individual sequence position/word* independently.

Mathematically, for a vector $x$:
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \gamma + \beta$$
Where:
* $\mu$ is the mean of the vector $x$
* $\sigma^2$ is the variance of the vector $x$
* $\epsilon$ is a tiny number to prevent division by zero
* $\gamma$ (scale) and $\beta$ (shift) are learnable parameters initialized to 1 and 0.

Let's write a custom implementation from scratch:

```python
class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scale (gamma) and shift (beta) parameters
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        # Calculate mean and variance along the embedding dimension (last dimension)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.gamma * x_normalized + self.beta
```

**Resume line:** Developed a custom Layer Normalization module from scratch to stabilize activations across embedding dimensions.

---

## Stage 6 — One Full Transformer Block

**What it does:** combines stages 1+2+3+4+5 into one repeatable unit. This is one "floor" of the building. GPT stacks many of these floors.

We place the Layer Normalization before each sublayer (Pre-LN style, which is standard in modern models like GPT-2/GPT-3) and apply residual connections.

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention    = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim)
        # Using our custom LayerNorm (or PyTorch's nn.LayerNorm)
        self.norm1        = LayerNorm(embed_dim)
        self.norm2        = LayerNorm(embed_dim)

    def forward(self, x):
        # Pre-LN: Apply norm first, then attention, then add residual (x)
        x = x + self.attention(self.norm1(x))
        # Pre-LN: Apply norm first, then feed forward, then add residual (x)
        x = x + self.feed_forward(self.norm2(x))
        return x
```

**Resume line:** Built a full Transformer block with multi-head attention, feed-forward layers, layer normalization, and residual connections.

---

## Stage 7 — Stack into a full GPT model

**What it does:** takes your embedding layer, stacks N transformer blocks on top, then adds an output head that guesses the next word.

```python
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, num_layers):
        super().__init__()

        # your existing embedding work plugged straight in
        self.token_emb  = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb    = nn.Embedding(max_seq_len, embed_dim)

        # stack many transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(embed_dim)

        # output head: turns final vector into next-word scores
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids):
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)

        x = self.token_emb(token_ids) + self.pos_emb(positions)
        x = self.blocks(x)
        x = self.norm(x)
        return self.output_head(x)   # shape: [B, T, vocab_size]
```

**Resume line:** Designed and implemented a GPT-style autoregressive language model from scratch with stacked Transformer blocks.

---

## Stage 8 — Training loop

**What it does:** feeds sentences in, compares what the model guessed vs what actually came next, and nudges every weight slightly in the right direction. Repeat millions of times.

```python
model     = MiniGPT(vocab_size=200, embed_dim=128, max_seq_len=64, num_heads=4, num_layers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn   = nn.CrossEntropyLoss()

def train_step(token_ids):
    # input  = all tokens except the last one
    # target = all tokens except the first one (shifted by 1)
    inputs  = token_ids[:, :-1]   # "hello world"
    targets = token_ids[:, 1:]    # "world <next>"

    logits = model(inputs)        # model's guesses

    # reshape for loss calculation
    logits  = logits.reshape(-1, 200)
    targets = targets.reshape(-1)

    loss = loss_fn(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

**Resume line:** Implemented autoregressive training with cross-entropy loss and Adam optimizer.

---

## Stage 9 — Text generation

**What it does:** give the model a starting word, it guesses the next word, you feed that back in, it guesses again — like autocomplete on steroids.

```python
def generate(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()
    ids = tokenizer.encode([prompt])   # your SPTokenizer from before

    for _ in range(max_new_tokens):
        logits    = model(ids)
        next_logit = logits[:, -1, :]              # only care about last position
        probs     = torch.softmax(next_logit, dim=-1)
        next_id   = torch.multinomial(probs, 1)    # sample from distribution
        ids       = torch.cat([ids, next_id], dim=1)

        if next_id.item() == tokenizer.sp.eos_id():  # stop at end-of-sentence
            break

    return tokenizer.sp.decode(ids[0].tolist())

print(generate(model, tokenizer, "hello"))
```

**Resume line:** Implemented autoregressive token-by-token text generation with temperature sampling.

---

## Stage 10 — FastAPI backend

**What it does:** wraps your model in an API so anyone can call it from a browser, app, or anywhere — exactly like how OpenAI's API works.

```python
# pip install fastapi uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.post("/generate")
def generate_text(req: PromptRequest):
    output = generate(model, tokenizer, req.prompt, req.max_tokens)
    return {"response": output}

# run with: uvicorn app:app --reload
```

Now your model is a real API. Hit `POST /generate` with a prompt, get text back. Same structure as OpenAI.

**Resume line:** Deployed language model as a REST API using FastAPI with JSON request/response handling.

---

## Stage 11 — Simple frontend

**What it does:** a minimal chat UI in plain HTML + JavaScript — type a message, hit send, see the model reply. Looks like ChatGPT.

```html
<!-- save as index.html, open in browser -->
<!DOCTYPE html>
<html>
<body style="max-width:600px; margin:auto; font-family:sans-serif">
  <h2>MiniGPT</h2>
  <div id="chat" style="height:400px; overflow-y:auto; border:1px solid #ccc; padding:10px"></div>
  <input id="input" style="width:80%" placeholder="Type a message..." />
  <button onclick="send()">Send</button>

  <script>
    async function send() {
      const prompt = document.getElementById("input").value
      document.getElementById("chat").innerHTML += `<p><b>You:</b> ${prompt}</p>`

      const res = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({prompt, max_tokens: 50})
      })
      const data = await res.json()
      document.getElementById("chat").innerHTML += `<p><b>Bot:</b> ${data.response}</p>`
      document.getElementById("input").value = ""
    }
  </script>
</body>
</html>
```

**Resume line:** Built an end-to-end chat interface connecting a custom LLM backend to a browser-based frontend.

---

## Your full resume bullet points when done

```
• Implemented Byte Pair Encoding tokenizer from scratch trained on UltraChat 200k
• Built token and positional embedding layers using raw PyTorch weight matrices
• Implemented scaled dot-product self-attention and multi-head attention from scratch
• Designed full GPT-style Transformer architecture with stacked blocks, layer norm,
  and residual connections
• Trained autoregressive language model with cross-entropy loss and Adam optimizer
• Deployed model as a REST API using FastAPI
• Built end-to-end chat interface connecting custom LLM to a browser frontend
```

---

**One line summary: self-attention → multi-head attention → feed forward → residual connections → layer normalization → stack into transformer blocks → add output head → train it → generate text → wrap in FastAPI → add a chat UI, and that entire pipeline is a real ChatGPT-style application you built from scratch.**