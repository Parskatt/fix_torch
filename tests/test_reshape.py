import torch
import fix_torch


x = torch.randn(6,4,5,6)


y = x.reshape(1,2,3,...)
print(y.shape)
z = x.reshape(...,3,10)
print(z.shape)
