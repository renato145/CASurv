__all__ = ['make_cuts']


from fastai.basics import *
from ..basics import *


def make_cuts(x, cuts, equidistant=True, min_=0, max_=None):
    x = tensor(x)
    min_ = ifnone(min_, min(x))
    max_ = ifnone(max_, max(x))
    if equidistant: cuts = torch.linspace(min_, max_, cuts)
    else:
        x = x.sort()[0]
        n = len(x)
        cuts -= 1
        idxs = tensor([n // cuts]*cuts)
        t = torch.zeros_like(idxs)
        t[:n-idxs.sum()] = 1
        idxs = (idxs + t).cumsum(0) - 1
        cuts = torch.cat([tensor([min_]), x[idxs]])

    return cuts