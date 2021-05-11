# fix_torch

Quality of Life improvements for pytorch


Basically a combination of monkey-patches and additional functionality which is missing from pytorch.
Use at your own *peril* :o 


## Examples

```python
import fix_torch
import torch

x = torch.randn(13,127,12,3)
y = torch.randn(13,127,3,17)
v = torch.randn(8,4,4)

z = x@y # do bmm without torch being annoying about it
tr = torch.trace(v) # do trace on batched matrices

w = x.reshape(...,4,3,3)

```
