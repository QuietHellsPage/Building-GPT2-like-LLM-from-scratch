"""
This dirty file has code with experiments and tests I make while creating the components of the model
"""
import torch
from src.components.gptmodel import GPTModel


def generate(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        
        last_logits = logits[:, -1, :]
        probas = torch.softmax(last_logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

        return idx

"""
Testing models
"""

import tiktoken

CONFIG = {
    "vocab_size": 50257,
    "context_len": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

tokenizer = tiktoken.get_encoding("gpt2")
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"

# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))

# batch = torch.stack(batch, dim=0)
# print(batch)

# torch.manual_seed(1)

# model = GPTModel(CONFIG)
# logits = model(batch)
# print(logits.shape)

# total_params = sum(p.numel() for p in model.parameters())

# total_bytes = total_params * 4
# size = total_bytes  / (1024 ** 2)
# print(size)


start_context = "Hello, I am"

encoded = tokenizer.encode(start_context)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)

model = GPTModel(CONFIG)

model.eval()

out = generate(model, encoded_tensor, 3, 1024)

decoded_text = tokenizer.decode(out.squeeze(0).tolist())

print(decoded_text)