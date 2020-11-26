import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import Tensor



# Implementation of the softplussigmoid activation function, following
# how pytorch implements activation functions

def softplussigmoid(x: Tensor, inplace: bool=False) -> Tensor:
    """
    Applies the SoftPlusSigmoid unit function element-wise
    """
    result = F.softplus(x) * torch.sigmoid(x)
    return result

class SoftPlusSigmoid(nn.Module):
    """
    Applies the softPlusSigmoid function element-wise:
    SoftPlusSigmoid = SoftPlus(x) * Sigmoid(x)
    Args:
        x: input tensor
        inplace: optionally do the operation in-place
    Shape:
        - x: (N, *), where * means any number of addtional dimensions
        - Output: (N, *), same shape as input
    """
    def __init__(self, inplace: bool = False):
        #self.soft = super(nn.Softplus, self).__init__()
        #self.sig = super(Sigmoid, self).__init__()
        super(SoftPlusSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) * F.sigmoid(x)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


# Implementation of our custom loss function
# loosely follows how pytorch implements loss functions, but not entirely
def make_mask_weights(shape, dtype, device):
    """
    Used to create a mask weights tensor, specifically for 2d images. These mask
    weights are used in our computation of the weights for our loss function
    I believe this is the fastest way to create the array. Alternative would be
    to use an O(n^2) algorithm, which is probably slower
    """
    # check shape
    # if len(shape) != 2:
        # raise ValueError(f"Mask shape ({shape}) isn't 2 dimensional")
    n = np.fromfunction(lambda i, j: np.maximum(np.abs(i-shape[1]/2), np.abs(j-shape[2]/2)), (shape[1], shape[2]), dtype=float)
    # stack em up
    n = np.tile(n, (shape[0], 1, 1))
    return torch.tensor(n, dtype=dtype, device=device)

def reg_loss(mask, mask_weights, reduction='none'):
    """
    Computes the element/pixel wise loss using the mask weights passed in.
    Currently supports either no reduction, or mean reduction
    """
    # check dimensions match and yell if they don't
    if mask.shape != mask_weights.shape:
        raise ValueError(f"Shape of mask {mask.shape} doesn't match shape of mask weights {mask_weights.shape}")
    if reduction == 'mean':
        return (mask*mask_weights).mean()
    elif reduction == 'none':
        return mask*mask_weights
    else:
        raise ValueError(f"Reduction operation {reduction} not supported")


