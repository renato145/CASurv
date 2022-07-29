__all__ = ['RegressionSurvLoss', 'ELRRegressionSurvLoss']


from fastai.vision.all import *
from ...basics import *
from ..core import *
from ..data import *


class RegressionSurvLoss(Module):
    '''Censor aware loss for survival regression
    Params:
    - loss_func: main loss function.
    - rank_func: if given is used to calculate a ranking loss.
    - rank_penalty: factor to multiply to the result of `rank_func`.
    '''
    def __init__(self, loss_func=nn.MSELoss(), rank_func=CAMarginRank(), rank_penalty=0):
        "If possible sets `loss_func.reduction` to `'none'`"
        store_attr()
        if hasattr(self.loss_func, 'reduction'): setattr(self.loss_func, 'reduction', 'none')
        self.main_loss_weights = None

    __repr__ = basic_repr(['loss_func', 'rank_func', 'rank_penalty'])
    @property
    def do_rank(self): return (self.rank_func is not None) and (self.rank_penalty > 0)

    def get_main_loss(self, output, target):
        surv = target.surv
        cens = target.cens
        no_cens_idxs = torch.where((output > surv) & (cens == 1))[0]
        loss = self.loss_func(output.flatten(), surv.flatten())
        loss[no_cens_idxs] = 0
        return loss

    def forward(self, output, target):
        loss = self.get_main_loss(output, target)
        if self.main_loss_weights is not None: loss = loss * self.main_loss_weights.to(loss.device)
        self.main_loss = loss.mean()
        self.rank_loss = self.rank_func(output, target).mul(self.rank_penalty) if self.do_rank else 0
        return self.main_loss + self.rank_loss


class ELRRegressionSurvLoss(RegressionSurvLoss):
    "`RegressionSurvLoss` with ELR regularizer"
    @delegates(RegressionSurvLoss.__init__)
    def __init__(self, elr_cb, **kwargs):
        super().__init__(**kwargs)
        self.elr_cb = elr_cb

    def forward(self, output, target):
        loss = super().forward(output, target)
        self.elr_reg = self.elr_cb.elr_reg
        return loss + self.elr_reg


@patch
@delegates(RegressionSurvLoss)
def setup_regression_surv_loss(self: Learner, elr=False, elr_plus=False, elr_alpha=1.0, elr_beta=0.3,
                               elr_gamma=0.997, model2=None, **kwargs):
    # loss
    if elr:
        # elr callback
        if elr_plus:
            elr_cb = ELRPlusCallback(alpha=elr_alpha, beta=elr_beta, gamma=elr_gamma, c=1)
            self.model = ELRPlusModel(self.model, model2)
            # optimizer splitter
            if self.splitter != trainable_params: elr_set_splitter(self)
        else: elr_cb = ELRCallback(alpha=elr_alpha, beta=elr_beta, c=1)
        self.add_cb(elr_cb)
        self.loss_func = ELRRegressionSurvLoss(elr_cb, **kwargs)
    else: self.loss_func = RegressionSurvLoss(**kwargs)

    # metrics
    metrics = 'main_loss'
    if getattr(self.loss_func, 'do_rank', False): metrics += ',rank_loss'
    if elr                                      : metrics += ',elr_reg'
    self._metrics += LossMetrics(metrics)

    return self

Learner.setup_regression_surv_loss.__doc__ = RegressionSurvLoss.__doc__