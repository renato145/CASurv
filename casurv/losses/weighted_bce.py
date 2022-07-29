__all__ = ['WeightedBCEWithLogits']


from fastai.basics import *


class WeightedBCEWithLogits(Module):
    def __init__(self, pos_weights=None, neg_weights=None, loss_scale=1.0, thresh=0.5):
        "`loss_scale` multiply the final loss, as the use of weights may produce very low values"
        if (pos_weights is None) or (neg_weights is None):
            pos_weights,neg_weights = 1,1
        else:
            pos_weights = nn.Parameter(tensor(pos_weights), requires_grad=False)
            neg_weights = nn.Parameter(tensor(neg_weights), requires_grad=False)
            assert pos_weights.shape == neg_weights.shape
        store_attr()

    __repr__ = basic_repr('loss_scale')

    def forward(self, output, target):
        # Original code before optimize it:
        # preds = input.sigmoid()
        # loss = (    -preds.log().mul(  target).mul(pos_weights) -
        #          (1-preds).log().mul(1-target).mul(neg_weights)).mean()
        preds = F.logsigmoid(output)
        loss = (            -preds.mul(  target).mul(self.pos_weights) -
                 preds.sub(output).mul(1-target).mul(self.neg_weights)).mean()
        return loss * self.loss_scale

    def decodes(self, x): return x > self.thresh
    def activation(self, x): return x.sigmoid()