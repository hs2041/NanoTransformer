# GPT2 custom model

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as f

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        config.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# Params set as per GPT2 hugging face model
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), #position embedding
            h = nn.ModuleList([BLock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = false)
    
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arrange(0, T, dtype = torch.log, device = idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


# Trickes to improve speed
model = torch.compile(model)

torch.autocast(device_type = device, dtype = torch.bloat16)

 # Kernel fusion
 # Reduce round trips to the memory on GPU during kernel computation

 # Flash attention: Optimal usage of GPU memory (SRAM, HBM) for the attention computation step
#  y = F.scaled_dot_product_attention(q, k, v, is_causal = True)

# use good numbers, i.e., number that are divisible by powers of 2

# Optimize learning
    # Variable learning rate (higher in the beginning, decays over a cosine function)
    # LInear gradual batch size increase
    # Use weight decays (fused paramter in AdamW makes things faster)
    # Use gradient accumulation to handle larger batch sizes

# Distributed data parallelism
  # Each GPU proceses different parts of the data
  # Then a accumulation process aggregates the results from all the GPUs

# from torch.distributed import init_process_group, destroy_process_group
# ddp_rank = int(os.environ['RANK'])   
# master_process = ddp_rank == 0 # printing, logging, checkpointing
  # Similar to MPI

# DDP does gradient synchronisation by itself, use no_sync() to disable it


# Real time debugging tactic
import code; code.interact(local= locals)

# Wait for GPU to finish computation
torch.synchronize()

# Evaluation using helloswag
