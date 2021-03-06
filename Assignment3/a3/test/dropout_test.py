import torch
import torch.nn as nn
inp = torch.tensor([1.0, 2.0, 3, 4, 5])

outplace_dropout = nn.Dropout(p=0.4)
print(inp)
output = outplace_dropout(inp)
print(output)
print(inp) # Notice that the input doesn't get changed here

print("--")
inplace_droput = nn.Dropout(p=0.4, inplace=True)
inplace_droput(inp)
print(inp) # Notice that the input is changed now