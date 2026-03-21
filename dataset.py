import torch 
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Convert Text to token 
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        #total possible sequences
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        #get chunk
        chunk = self.tokens[idx:idx + self.block_size+1]

        #split into input and target
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y

