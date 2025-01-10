import torch
import torchaudio
from torch.utils.data import DataLoader



BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        predicition = model(input)
        loss = loss_fn(predicition, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"loss: {loss.item()}")