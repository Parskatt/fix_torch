import torch
import types
import numpy as np
import copy

__all__ = ['unravel_index']


def unravel_index(indices,shape):
    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)

torch.trace = lambda x: torch.einsum('...ii->...',x)
a=copy.deepcopy(torch.Tensor.reshape)

torch.Tensor.__matmul__ = lambda self,other: torch.einsum('...ab,...bc->...ac',self,other)

def match_ind(shape1,shape2):
    p1 = np.cumprod(shape1)
    p2 = np.cumprod(shape2)
    return np.argwhere(p1==p2[-1]).item()

old_reshape = copy.deepcopy(torch.Tensor.reshape)

def new_reshape(self,shape,*_):
    if not isinstance(shape,tuple):
        shape = (shape,)+_ 
    shape = shape
    shape1 = self.shape
    shape2 = shape
    cum_prod = np.cumprod(shape1)
    if shape2[-1] is Ellipsis:
        ind = match_ind(shape1,shape2[:-1])
        shape = shape2[:-1]+shape1[ind+1:]
    elif shape2[0] is Ellipsis:
        ind = match_ind(shape1[::-1],shape2[::-1][:-1])
        shape = shape1[:-(ind+1)]+shape2[1:]
    return old_reshape(self,shape)


torch.Tensor.reshape = new_reshape 