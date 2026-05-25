import os

os.environ["TT_VISIBLE_DEVICES"] = "0"
os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"


import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torchvision
import torchvision.models as models
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

# sudo /home/delaunap/.tenstorrent-venv/bin/tt-smi -r

import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.perf_counter()
    loss = None

    def set_loss(l):
        nonlocal loss
        loss = l

    yield set_loss

    elapsed = time.perf_counter() - start
    if loss is not None:
        loss = f"loss: {loss}"
    else:
        loss = ""

    print(f"{name}: {elapsed:.4f}s {loss}")


def get_dataloader():
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.1307], std=[0.3081])  # MNIST mean/std
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False)


def get_model():
    class MNISTLinear(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, bias=True):
            super(MNISTLinear, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Flatten(),           # flatten the 1x28x28 image to 784
                nn.Linear(input_size, hidden_size, bias=bias),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size, bias=bias),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size, bias=bias),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits
    
    return MNISTLinear(28 * 28, 32 * 32, 10)


def get_device():
    # Standard PyTorch — the only difference is the device

    if os.getenv("DEVICE", "cpu") == "tt":
        return torch_xla.device()
    else:
        return torch.device("cpu")


def train():
    device = get_device()
    model = get_model().to(dtype=torch.bfloat16).to(device=device)
    loader = get_dataloader()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for inputs, targets in loader:
        with timer("step") as t:
            inputs, targets = inputs.to(device, dtype=torch.bfloat16), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            optimizer.step()
            torch_xla.sync(wait=True)
            t(loss.item())


with timer("all"):
    train()