import torch
import torch.nn as nn
import torch.nn.functional as F

# net
batch_size = 64
block_size = 256
n_emb = 384
n_heads = 6
n_layers = 6 # transformer
dropout = 0.2

# training
steps = 5000
lr = 3e-4
estimate_interval = 100
batch_estimate = 300
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(device)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
chars_size = len(chars)

stoi = { c:i for i,c in enumerate(chars) }
itos = { i:c for i,c in enumerate(chars) }
encode = lambda str: [stoi[c] for c in str]
decode = lambda inds: ''.join(itos[i] for i in inds)

data = torch.tensor(encode(text), dtype=torch.long)

torch.manual_seed(1337)

# split data on training and valuation
ind = int(0.9*len(data))
data_tr = data[:ind]
data_val = data[ind:]

def get_batch(split):
    data = data_tr if split == 'train' else data_val
    inds = torch.randint(0, len(data) - block_size, (batch_size,))
    
    x = torch.stack([data[i:i+block_size] for i in inds])
    y = torch.stack([data[i+1:i+block_size+1] for i in inds])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(batch_estimate)
        for i in range(batch_estimate):
            x_b, y_b = get_batch(split)
            logits, loss = model(x_b, y_b)
            losses[i] = loss
        out[split] = losses.mean().item()
        
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.keys =     nn.Linear(n_emb, head_size, bias=False)
        self.queries =  nn.Linear(n_emb, head_size, bias=False)
        self.values =   nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape 
        k = self.keys(x)
        q = self.queries(x)

        weights = q @ k.transpose(-2,-1) * C**-0.5
        # tril = torch.tril(torch.ones((block_size,block_size)))
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        v = self.values(x)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1) # create n_emb (concat all heads)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb, n_emb),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_heads, n_emb):
        super().__init__()
        head_size = n_emb // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwrd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwrd(self.ln2(x))
        return x
    
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(chars_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(*[Block(n_heads=n_heads, n_emb=n_emb) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_emb) # final layer norm
        self.lm_head = nn.Linear(n_emb, chars_size)
        
    def forward(self, ind, targets=None):
        B,T = ind.shape
        tok_emb = self.token_embedding_table(ind) # [batch_size, block_size, n_emb]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # [block_size, n_emb]
        x = tok_emb + pos_emb    # [batch_size, block_size, n_emb]
        x = self.blocks(x)       # [batch_size, block_size, n_emb]
        x = self.ln_f(x)         # [batch_size, block_size, n_emb]
        logits = self.lm_head(x) # [batch_size, block_size, chars_size]
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, ind, num_tokens):
        for _ in range(num_tokens):
            ind_con = ind[:, -block_size:]
            logits, loss = self(ind_con) # forward(x)
            logits = logits[:, -1, :] # get the last character [batch_size, n_emb]
            probs = F.softmax(logits, dim=-1)
            ind_next = torch.multinomial(probs, num_samples=1)
            ind = torch.cat((ind, ind_next), dim=1)
        
        return ind

model = BigramLanguageModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(steps):
    if i % estimate_interval == 0:
        losses = estimate_loss()
        print('step:', i, 'train loss:', losses['train'], 'val loss:', losses['val'])
    
    x_b, y_b = get_batch('train')
    logits, loss = model.forward(x_b, y_b)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# generate from the model
num_tokens = 1000
context = torch.zeros((1,1), dtype=torch.long, device=device)
result = model.generate(context, num_tokens=num_tokens)
print(decode(result[0].tolist()))
