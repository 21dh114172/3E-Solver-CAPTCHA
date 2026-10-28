"""Independent coordinate-padding contribution; visual convolutions stay unchanged.

Research reconstruction under the one-based valid-index convention 1..D.
This is not author code. p counts coordinate samples on EACH side and is
independent of nn.Conv2d.padding (the visual zero-padding width).
"""
import math
import torch
from torch import nn


POSITIONAL_MODES = ('legacy', 'linear', 'corrected', 'explicit')


class LearnedCoordinatePadding(nn.Module):
    """Learned side weights, summed across replicated input coordinate channels.

No convolution, normalization, activation, or bias is added here. Existing
layers own those operations. The two side-weight tensors have shape [O,I,p].
"""
    def __init__(self, in_channels, out_channels, p=2):
        super().__init__()
        if not isinstance(p, int) or isinstance(p, bool) or p < 1:
            raise ValueError('coordinate_padding must be a positive integer')
        self.p = p
        self.left = nn.Parameter(torch.empty(out_channels, in_channels, p))
        self.right = nn.Parameter(torch.empty(out_channels, in_channels, p))
        nn.init.normal_(self.left, std=0.02 / math.sqrt(in_channels * p))
        nn.init.normal_(self.right, std=0.02 / math.sqrt(in_channels * p))

    def forward(self, features, input_size, axis, kernel, stride, padding,
                mode='corrected'):
        if mode not in ('linear', 'corrected', 'explicit'):
            raise ValueError(f'Unsupported coordinate-padding mode: {mode}')
        # l is the index immediately BEFORE the central visual window.
        # Visual window: l+1,...,l+k; side indices: l-m and l+k+m.
        D = input_size
        l = torch.arange(features.shape[axis], device=features.device,
                         dtype=features.dtype) * stride - padding
        m = torch.arange(1, self.p + 1, device=features.device,
                         dtype=features.dtype)
        il, ir = l[:, None] - m, l[:, None] + kernel + m
        raw_l, raw_r = il / D, ir / D
        cl = torch.where(il < 1, -torch.ones_like(raw_l), raw_l)
        cr = torch.where(ir > D, torch.ones_like(raw_r), raw_r)
        wl = self.left.to(dtype=features.dtype).sum(dim=1)
        wr = self.right.to(dtype=features.dtype).sum(dim=1)
        if mode == 'explicit':
            # Direct dot product with every substituted coordinate sample.
            pos = torch.einsum('op,np->on', wl, cl)
            pos = pos + torch.einsum('op,np->on', wr, cr)
        else:
            slope = (wl + wr).sum(dim=-1)
            intercept = (kernel * wr.sum(dim=-1)
                         + ((wr - wl) * m).sum(dim=-1)) / D
            pos = slope[:, None] * (l / D)[None, :] + intercept[:, None]
            if mode == 'corrected':
                pos = pos + wl @ (cl - raw_l).T + wr @ (cr - raw_r).T
        return pos[None, :, :, None] if axis == 2 else pos[None, :, None, :]


def set_positional_mode(module, mode):
    """Switch ablations without replacing parameters or rebuilding optimizers."""
    if mode not in POSITIONAL_MODES:
        raise ValueError(f'Unknown positional mode: {mode}')
    for child in module.modules():
        if hasattr(child, 'positional_mode'):
            child.positional_mode = mode
    return module
