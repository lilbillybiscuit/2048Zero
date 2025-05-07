import torch
import os
from torch import nn
from typing import List, Tuple, Any, Dict, Optional, Union
import torch.nn.functional as F
import numpy as np

# Default device - but don't auto-set it here
# This causes all workers to try using CUDA
# Instead, each model instance should set its own device based on constructor parameter
default_device = 'cpu'


class ZeroResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ZeroResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1) # input is filters x n x m
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)# -> filters x 1 x 1
        self.linear1 = nn.Linear(filters, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 2*filters)

    def forward(self, x):
        # input x is batch x filters x n x m
        identity = x
        out = self.relu(self.conv1(x)) # -> batch x filters x n x m
        out = self.conv2(out) # -> batch x filters x n x m

        # squeeze and excitation
        y = self.avgpool1(out) # -> batch x filters x 1 x 1
        y = self.linear1(y.view(y.size(0), -1)) # -> batch x filters -> 32
        y = self.relu(y) # -> batch x 32
        y = self.linear2(y) # -> batch x 2 * filters

        W,B = y.chunk(2, dim=1) # each ones is batch x filters
        W = torch.sigmoid(W) # -> batch x filters

        W = W.view(-1, W.size(1), 1, 1) # -> batch x filters x 1 x 1
        B = B.view(-1, B.size(1), 1, 1) # -> batch x filters x 1 x 1
        out = identity + W * out + B # -> batch x filters x n x m (should broadcast operation)
        out = self.relu(out)
        return out



class ZeroNetworkMain(nn.Module):
    def __init__(self, n: int, m: int, k: int, filters=128, blocks=10, infer_device='cpu'):
        super(ZeroNetworkMain, self).__init__()
        self.n = n
        self.m = m
        self.k = k
        # Store architecture parameters for serialization
        self.filters = filters
        self.blocks = blocks
        
        # input = k * n * m, where k is max exponent + 1 (for 0), n and m are dimensions
        self.conv1 = nn.Conv2d(in_channels=k, out_channels=filters, kernel_size=3, padding=1) # -> F x n x m
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)

        # Create residual blocks
        self.blocks_list = nn.ModuleList([ZeroResidualBlock(filters) for _ in range(blocks)])

        # policy head
        self.policy_conv1 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1) # -> F x n x m
        self.policy_conv2 = nn.Conv2d(in_channels=filters, out_channels=40, kernel_size=3, padding=1) # -> 40 x n x m
        self.flatten = nn.Flatten()
        self.policy_linear = nn.Linear(40*n*m, 4)
        self.softmax = nn.Softmax(dim=1)
        self.infer_device = infer_device

        # value head
        self.value_conv1 = nn.Conv2d(in_channels=filters, out_channels=16, kernel_size=3, padding=1)
        self.value_conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.value_linear1 = nn.Linear(8*n*m, 1)
        self.infer_device = infer_device

    def to(self, device: Union[str, torch.device]) -> 'ZeroNetwork':
        """Move the model to the specified device."""
        super().to(device)
        self.infer_device = device
        return self

    def forward(self, x):

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        for block in self.blocks_list:
            out = block(out)

        # policy head
        policy_out = self.relu(self.policy_conv1(out)) # -> batch x filters x n x m
        policy_out = self.policy_conv2(policy_out) # -> batch x 40 x n x m
        policy_out = self.flatten(policy_out) # -> batch x 40*n*m
        policy_out = self.policy_linear(policy_out) # -> batch x 4
        policy_out = self.softmax(policy_out)

        # value head
        value_out = self.relu(self.value_conv1(out)) # -> batch x 16 x n x m
        value_out = self.value_conv2(value_out) # -> batch x 8 x n x m
        value_out = self.flatten(value_out) # -> batch x 8*n*m
        value_out = self.relu(value_out) # -> batch x 8*n*m
        value_out = torch.tanh(self.value_linear1(value_out)) # -> batch x 1

        return policy_out, value_out

    def to_onehot(self, board: np.ndarray, device=None) -> torch.Tensor:
        # board is batch x k x n x m
        device = device or self.infer_device
        onehot = torch.tensor(board, dtype=torch.long, device=device)
        oh = F.one_hot(onehot, num_classes=self.k).permute(0,3,1,2).float()
        return oh

    def infer(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        batch_size = len(board)
        p, v= self.forward(self.to_onehot(board, device=self.infer_device))
        p = p.detach().cpu().reshape(batch_size, 4).numpy()
        v = v.detach().cpu().reshape(batch_size).numpy()
        return p, v

    def save(self, path: str):
        torch.save(self.state_dict(), path, pickle_protocol=4)
        print(f"Model saved to {path}")

class ZeroNetworkMini(ZeroNetworkMain):
    def __init__(self, n: int, m: int, k: int, filters=128, dense_out=64, blocks = None, infer_device='cpu'):
        super(ZeroNetworkMini, self).__init__(n, m, k, filters=filters, infer_device=infer_device)
        self.n = n
        self.m = m
        self.k = k
        self.filters = filters
        self.dense_out = dense_out

        self.conva1 = nn.Conv2d(in_channels=k, out_channels=filters, kernel_size=(3, 3), padding='same')
        self.conva2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(3, 3), padding='same')

        self.convb1 = nn.Conv2d(in_channels=k, out_channels=filters, kernel_size=(3, 3), padding='same')
        self.convb2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(3, 3), padding='same')

        self.flat = nn.Flatten()

        flattened_size = filters * n * m

        self.densea = nn.Linear(in_features=flattened_size, out_features=dense_out)
        self.denseb = nn.Linear(in_features=flattened_size, out_features=dense_out)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.policy_linear = nn.Linear(in_features=dense_out, out_features=4)

        self.value_linear = nn.Linear(in_features=dense_out, out_features=1)

        self.infer_device = infer_device
        self.to(infer_device)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.relu(self.conva1(x))
        a = self.relu(self.conva2(a))
        a_flat = self.flat(a)
        a_dense = self.relu(self.densea(a_flat))

        b = self.relu(self.convb1(x))
        b = self.relu(self.convb2(b))
        b_flat = self.flat(b)
        b_dense = self.relu(self.denseb(b_flat))

        merged = a_dense * b_dense

        policy_logits = self.policy_linear(merged)
        policy_out = self.softmax(policy_logits)

        value_out = self.value_linear(merged)
        value_out = self.tanh(value_out)

        return policy_out, value_out


ZeroNetwork = ZeroNetworkMain