import torch
import torch.nn.functional as F
# example with padding_idx
weights = torch.rand(10, 3)
weights[0, :].zero_()
embedding_matrix = weights
input = torch.tensor([[0,2,0,5]])

result = F.embedding(input, embedding_matrix, padding_idx=0)

print(result)
# tensor([[[ 0.0000,  0.0000,  0.0000],
#             [ 0.5609,  0.5384,  0.8720],
#             [ 0.0000,  0.0000,  0.0000],
#             [ 0.6262,  0.2438,  0.7471]]])