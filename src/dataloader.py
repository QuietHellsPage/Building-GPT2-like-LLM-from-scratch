"""

Raw data processing pipeline.

Steps:

1. Load raw text
2. Tokenize text
3. Create dataset using sliding window technique
4. Embed dataset

"""

import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset
from src.constants import VOCAB_SIZE, OUTPUT_DIM, CONTEXT_LENGTH, BATCH_SIZE, MAX_LENGTH


class GPTDataset(Dataset):
    """
    Dataset wrapper to create input_ids ant target_ids from scratch
    """

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(
    text, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0
):
    """
    Function that processes raw text into tokenized DataLoader
    """
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(text, tokenizer, max_length, stride)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


"""
Building Token embedder and positional encoder
"""

token_embeddings_layer = torch.nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
pos_encoder = torch.nn.Embedding(CONTEXT_LENGTH, OUTPUT_DIM)


"""
Testing on sample
"""
with open("data/the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()


dataloader = create_dataloader(raw_text, BATCH_SIZE, MAX_LENGTH, stride=MAX_LENGTH)

for batch in dataloader:
    x, _ = batch

    token_embeddings = token_embeddings_layer(x)
    pos_embeddings = pos_encoder(torch.arange(MAX_LENGTH))

    input_embeddings = token_embeddings + pos_embeddings

    break


print(input_embeddings.shape)
