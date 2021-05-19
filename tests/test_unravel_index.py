from fix_torch import torch

x = torch.unravel_index(torch.tensor([22, 41, 37]), (7,6))
print(x.shape)