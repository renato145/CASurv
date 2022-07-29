__all__ = ['get_combinations', 'MarginRank', 'CAMarginRank']


from fastai.basics import *


def get_combinations(x, y, remove_matchs=False):
    idxs = torch.combinations(torch.arange(len(x)))
    x1,x2 = x[idxs[:,0]],x[idxs[:,1]]
    y1,y2 = y[idxs[:,0]],y[idxs[:,1]]
    if remove_matchs:
        mask = y1!=y2
        x1,x2,y1,y2 = x1[mask],x2[mask],y1[mask],y2[mask]
    return x1,x2,y1,y2


def _get_lbls(y1, y2): return torch.where(y1 > y2, 1, -1).float()


class MarginRank(Module):
    'Obtain pair combinations and applies `nn.MarginRankingLoss`'
    __repr__ = basic_repr()
    def __init__(self):
        self.loss =  nn.MarginRankingLoss()

    def forward(self, output, target):
        x1,x2,y1,y2 = get_combinations(output.squeeze(), target)
        return self.loss(x1, x2, _get_lbls(y1, y2))


def _get_ca_lbls(y1, y2):
    lbls = torch.zeros(len(y1))
    lbls[(y1.surv >  y2.surv) & (y2.cens == 0)] = +1
    lbls[(y1.surv <= y2.surv) & (y1.cens == 0)] = -1
    return lbls.to(y1.device)


class CAMarginRank(MarginRank):
    'Censor Aware version of `MarginRank`, target is expected to be an instance of `TensorSurvival`'
    def forward(self, output, target):
        x1,x2,y1,y2 = get_combinations(output.squeeze(), target)
        return self.loss(x1, x2, _get_ca_lbls(y1, y2))