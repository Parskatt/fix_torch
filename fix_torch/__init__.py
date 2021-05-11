import torch

def unravel_index(indices,shape):
    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


torch.trace = lambda x: torch.einsum('...ii->...',x)
torch.Tensor.__matmul__ = lambda self,other: torch.einsum('...ab,...bc->...ac',self,other)