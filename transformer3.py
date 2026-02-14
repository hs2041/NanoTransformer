# Load GPT 2 model and it's weights and run inference promtps

import torch

import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import matplotlib as plt
from transformers import pipeline, set_seed

model_hf = GPT2LMHeadModel.from_pretrained("gpt2") #124M param model
sd_hf = model_hf.state_dict()

# 50247 tokens in gpt2, Each token is represented by an embedding of vector (size 768)
# for k, v in sd_hf.items():
#     print(k, v.shape)

generator = pipeline('text-generation', model = 'gpt2')
set_seed(42)
generated_txt = generator("But the problem is", max_length = 30, num_return_sequences = 5, truncation = True)

for generated in generated_txt:
    print(generated)
# print(generated_txt)