__all__ = ['DiscreteSurvLoss', 'SurvDiscreteMetric', 'c_index_discrete', 'PatientWiseCIndexDiscrete']


from fastai.vision.all import *
from ...basics import *
from ..core import *
from ..data import *

def ce_loss(hazards, surv, lbls, cens, alpha=0.4, eps=1e-7):
    lbls = lbls.view(-1, 1).long() # ground truth bin, 1,2,...,k
    cens = cens.view(-1, 1).float() # censorship status, 0 or 1
    surv_padded = torch.cat([torch.ones_like(cens), surv], 1) # all patients are alive from (-inf, 0)
    reg = -(1 - cens) * (  torch.gather(surv_padded, 1, lbls).clamp_min(eps).log()
                         + torch.gather(hazards    , 1, lbls).clamp_min(eps).log())
    ce_l = (    - cens  *      torch.gather(surv, 1, lbls).clamp_min(eps).log()
            -(1 - cens) * (1 - torch.gather(surv, 1, lbls).clamp_min(eps)).log())
    loss = (1-alpha) * ce_l + alpha * reg
    return loss.mean()

def nll_loss(hazards, surv, lbls, cens, eps=1e-7):
    lbls = lbls.view(-1, 1).long() # ground truth bin, 1,2,...,k
    cens = cens.view(-1, 1).float() # censorship status, 0 or 1
    surv_padded = torch.cat([torch.ones_like(cens), surv], 1) # all patients are alive from (-inf, 0)
    uncensored_loss = -(1 - cens) * (  torch.gather(surv_padded, 1, lbls).clamp_min(eps).log()
                                     + torch.gather(hazards    , 1, lbls).clamp_min(eps).log())
    censored_loss = -cens * torch.gather(surv_padded, 1, lbls+1).clamp_min(eps).log()
    loss = censored_loss + uncensored_loss
    return loss.mean()


class DiscreteSurvLoss(Module):
    '''From paper:
       [Bias in Cross-Entropy-Based Training of Deep Survival Networks](https://ieeexplore.ieee.org/document/9028113)

       Implementation based on https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py

       n_bins: number of bins
       q_bins: bins limits (should have `c + 1` elements),
               bins are defined as `[lower,upper)` (includes lower limit but not upper limit)
       use_ce: use cross-entropy, else use negative log-likelihood (check paper)
       alpha: how much to weigh uncensored patients (only for use_ce=True)
    '''
    def __init__(self, n_bins, q_bins, use_ce=False, do_sigmoid=True, alpha=0): store_attr()
    __repr__ = basic_repr('n_bins,use_ce,do_sigmoid,alpha')

    @classmethod
    def from_ds(cls, ds, n_bins, use_ce=False, do_sigmoid=True, alpha=0, tls_idx=-1):
        uncensored_times = [surv.item() for surv,cens in ds.tls[tls_idx] if cens==0]
        q_bins = pd.qcut(uncensored_times, n_bins, retbins=True, labels=False)[1]
        q_bins[0] = 0
        q_bins[-1] = np.inf
        return cls(n_bins, q_bins, use_ce, do_sigmoid, alpha)

    def get_discrete_lbls(self, target):
        disc_lbls = pd.cut(to_np(target.surv), bins=self.q_bins, labels=False, right=False, include_lowest=True)
        return tensor(disc_lbls).to(target.device)

    def forward(self, output, target):
        hazards = self.get_hazard(output)
        surv = torch.cumprod(1 - hazards, dim=1)
        lbls = self.get_discrete_lbls(target)
        if self.use_ce: return  ce_loss(output, surv, lbls, target.cens, self.alpha)
        else          : return nll_loss(output, surv, lbls, target.cens)

    def get_hazard(self, x): return x.sigmoid() if self.do_sigmoid else x
    def decodes(self, x):
        "Decodes model output into risk score"
        hazards = self.get_hazard(x)
        surv = torch.cumprod(1 - hazards, dim=1)
        return -surv.sum(1)


@patch
@delegates(DiscreteSurvLoss.from_ds)
def setup_discrete_survival_loss(self: Learner, n_bins, **kwargs):
    self.loss_func = DiscreteSurvLoss.from_ds(self.dls.train_ds, n_bins=n_bins, **kwargs)
    return self

Learner.setup_discrete_survival_loss.__doc__ = DiscreteSurvLoss.__doc__


class SurvDiscreteMetric(AccumMetric):
    "Gets risk score from model output before computing metrics"
    def accum_values(self, preds, targs, learn=None):
        "Store targs and preds"
        to_d = learn.to_detach if learn is not None else to_detach
        preds,targs = to_d(preds),to_d(targs)
        preds = learn.loss_func.decodes(preds)
        if self.flatten: preds,targs = flatten_check(preds,targs)
        self.preds.append(preds)
        self.targs.append(targs)


from ..metrics import _c_index, c_index

c_index_discrete = SurvDiscreteMetric(_c_index, flatten=False)
c_index_discrete.__doc__ = f'{c_index.__doc__} (for survival discrete models)\n{SurvDiscreteMetric.__doc__}'


from ..metrics import SurvivalPatientWiseMetric, PatientWiseCIndex

def PatientWiseCIndexDiscrete(patient_ids):
    return SurvivalPatientWiseMetric(c_index, patient_ids, SurvDiscreteMetric)

PatientWiseCIndexDiscrete.__doc__ = (f'{PatientWiseCIndex.__doc__} (for survival discrete models)\n'
                                     f'{SurvDiscreteMetric.__doc__}')