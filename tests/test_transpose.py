import torch
import fix_torch


x = torch.randn(5,6,7)
print(torch.t(x).shape)
print(x.t().shape)
