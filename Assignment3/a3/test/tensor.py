import torch
import torch.nn as nn
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(input.shape)
print(output.shape)
print(m, input, output)
print(output.size())
# torch.Size([128, 30])