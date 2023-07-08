import warnings
import math
import torch
from torch.nn import _reduction as _Reduction
from torch.nn import functional as F
from torch.nn.modules import Module
        

class RuberLoss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(RuberLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

    def forward(self, input, target):
        _assert_no_grad(target)
        return ruber_loss(input, target, reduction=self.reduction)

def ruber_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if target.requires_grad:
        ret = _ruber_loss(input, target)
        if reduction != 'None':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = _ruber_loss(expanded_input, expanded_target)
    return ret

def _ruber_loss(input, target):
    t = torch.abs(input - target)
    c = 10.
    return torch.where(t<c, t, torch.sqrt(2*c*t-c**2))

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"
