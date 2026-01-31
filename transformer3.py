import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2") #124M param model
sd_hf = model_hf.state_dict()

for k, v in sd_hf.items():
    print(k, v.shape)