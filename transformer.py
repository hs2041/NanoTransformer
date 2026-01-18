import torch

# Let's build GPT by Andrej Karpathy
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of dataset: ", len(text))

# List of unique characters in the text

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] # takes a string and outputs a list of integer
decode = lambda l: ''.join([itos[i] for i in l]) # coverts integers to strings

print(encode("hii there"))
print(decode(encode("hii there")))