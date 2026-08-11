import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta


class MultiHeadedAttention(nn.Module):
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
    def __init__(self, embed_dim, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(embed_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.relu(self.w_1(x)))


class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadedAttention(embed_dim, num_heads)
        self.feed_forward = PositionwiseFeedForward(embed_dim, d_ff)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x, mask):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class SimpleCharTokenizer:
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
    corpus = "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence."
    tokenizer = SimpleCharTokenizer(corpus)
    
    print("Corpus:", corpus)
    print("Vocab size:", tokenizer.vocab_size)
    
    prompt = "Artificial"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    print(f"\nPrompt: '{prompt}'")
    print(f"Tokenized input: {input_ids.tolist()}")
    
    model = GPT(vocab_size=tokenizer.vocab_size)
    
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}") # Expect [1, seq_len, vocab_size]
    
    next_token_logits = logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    next_id = torch.argmax(probs).item()
    next_char = tokenizer.id_to_char[next_id]
    print(f"Predicted next character: '{next_char}'")
    
    # Generate 10 characters
    curr_ids = input_ids
    generated = prompt
    for _ in range(10):
        logits = model(curr_ids)
        next_id = torch.argmax(logits[0, -1, :]).item()
        next_char = tokenizer.id_to_char[next_id]
        generated += next_char
        curr_ids = torch.cat([curr_ids, torch.tensor([[next_id]])], dim=1)
        
    print(f"Generated text: '{generated}'")
    print("\nComplete test passed successfully!")
