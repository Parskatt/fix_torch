import torch
import numpy as np
import copy
from typing import Tuple
from collections.abc import Iterable

__all__ = []


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int, device=indices.device)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)

# Introduce unravel_index function (credit to https://github.com/pytorch/pytorch/issues/35674#issuecomment-739492875)
torch.unravel_index = unravel_index

#Overload matrix trace
torch.trace = lambda x: torch.einsum('...ii->...',x)

#Overload matrix multiplication 
torch.Tensor.__matmul__ = lambda self,other: torch.einsum('...ab,...bc->...ac',self,other)



def match_ind(shape1,shape2):
    p1 = np.cumprod(shape1)
    p2 = np.cumprod(shape2)
    return np.argwhere(p1==p2[-1])[0].item()
torch_reshape = copy.deepcopy(torch.Tensor.reshape)

def fixed_reshape(self, shape, *_):
    if not isinstance(shape, Iterable):
        shape = (shape,)+_ 
    shape = shape
    shape1 = self.shape
    shape2 = shape
    if shape2[-1] is Ellipsis:
        ind = match_ind(shape1,shape2[:-1])
        shape = shape2[:-1]+shape1[ind+1:]
    elif shape2[0] is Ellipsis:
        ind = match_ind(shape1[::-1],shape2[::-1][:-1])
        shape = shape1[:-(ind+1)]+shape2[1:]
    return torch_reshape(self,shape)

# Overload the pytorch reshape functions
torch.Tensor.reshape = fixed_reshape
torch.reshape = fixed_reshape

# Overload the pytorch transpose functions, note that I have not figured out how to overwrite .T yet
torch.t = lambda x: x.permute(*range(len(x.shape)-2),-1,-2)
torch.Tensor.t = lambda self: torch.t(self)

to_cpy = copy.deepcopy(torch.nn.Module.to)
def new_to(self,device,*args,**kwargs):
    self.device = device
    return to_cpy(self,device,*args,**kwargs)
torch.nn.Module.to = new_to
