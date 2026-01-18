import torch

# Let's build GPT by Andrej Karpathy
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of dataset: ", len(text))

