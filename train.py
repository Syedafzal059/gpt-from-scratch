from pickletools import optimize
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.tokenizer import CharTokenizer
from model.gpt import GPTModel
from dataset import GPTDataset

batch_size = 16
block_size = 32
embed_dim = 64
num_heads = 4
num_layers = 2
learning_rate = 3e-4
epochs = 200


# Sample text (replace later with real dataset)
text = "hello Afzal this is a simple dataset for training a gpt model from scratch"
tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size

# Dataset + Loader
dataset = GPTDataset(text, tokenizer, block_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Model 
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPTModel(vocab_size, embed_dim, num_heads, num_layers, block_size).to(device)
model.to(device)


#Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


#Training Loop
for epoch in range(epochs):
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        #Forward Pass
        logits = model(x)  #(B, T, vocab_size)

        #Reshape for loss
        B, T, V = logits.shape
        logits = logits.view(B*T, V)
        y = y.view(B*T)

        #Compute Loss
        loss = criterion(logits, y)

        #Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
