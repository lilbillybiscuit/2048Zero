import torch
import os
from torch import nn

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



class ZeroNetwork(nn.Module):
    def __init__(self, n, m, k, filters=128, blocks=10):
        super(ZeroNetwork, self).__init__()
        # input = k * n * m, where k is 2^k (max tile), n and m are dimensions
        self.conv1 = nn.Conv2d(in_channels=k, out_channels=filters, kernel_size=3, padding=1) # -> F x n x m
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([ZeroResidualBlock(filters) for _ in range(blocks)])

        # policy head
        self.policy_conv1 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1) # -> F x n x m
        self.policy_conv2 = nn.Conv2d(in_channels=filters, out_channels=40, kernel_size=3, padding=1) # -> 40 x n x m
        self.flatten = nn.Flatten()
        self.policy_linear = nn.Linear(40*n*m, 4)
        self.softmax = nn.Softmax(dim=1)

        # value head
        self.value_conv1 = nn.Conv2d(in_channels=filters, out_channels=16, kernel_size=3, padding=1)
        self.value_conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.value_linear1 = nn.Linear(8*n*m, 1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        for block in self.blocks:
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
        value_out = self.value_linear1(value_out) # -> batch x 1

        return policy_out, value_out

