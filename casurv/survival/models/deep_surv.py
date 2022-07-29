__all__ = ['DeepSurv', 'DeepSurvLoss', 'DeepSurvSplitter', 'get_survival_estimation', 'get_survival_time_metrics']


from fastai.vision.all import *
from ...basics import *
from ..core import *
from ..data import *


class DeepSurv(Module):
    "DeepSurv model wrapper"
    def __init__(self, base_model): store_attr()
    def forward(self, x): return self.base_model(x)
    def hazard_ratio(self, x):
        "A greater hazard signifies a greater risk of death"
        return self(x).exp()


class DeepSurvLoss(Module):
    '''From paper:
       'DeepSurv: Personalized Treatment Recommender System UsingA Cox Proportional Hazards Deep Neural Network'
       (https://arxiv.org/abs/1606.00931)

       This loss penalizes the output if its not sorted according to the target.
       Implementation based on https://github.com/havakv/pycox/blob/master/pycox/models/loss.py#L407
    '''
    def forward(self, output, target):
        # 1. get indexes in descending order based on target survival times
        idxs = target.surv.argsort(descending=True)
        # 2. get predictions and censoring indicators based on `idxs`
        risk = output[idxs].view(-1)
        E = target.E[idxs].view(-1)
        # 3. calc loss
        gamma = risk.max() # avoid nan caused by exp() on big numbers
        log_risk = risk.sub(gamma).exp().cumsum(0).add(1e-7).log().add(gamma)
        loss = -risk.sub(log_risk).mul(E).sum().div(E.sum().clamp_min(1))
        return loss

    def decodes(self, x):
        "Decodes into hazard ratio instead of risk"
        return x.exp()


class DeepSurvSplitter:
    "Custom learn splitter for `DeepSurv` models."
    def __init__(self, split_fn): store_attr()
    def __call__(self, m): return self.split_fn(m.base_model)

@patch
def setup_deepsurv(self: Learner, model_cls=DeepSurv, loss_func=DeepSurvLoss()):
    "Modifies learn instance model, loss_fn and splitter."
    self.loss_func = loss_func
    if model_cls is not None:
        self.model = model_cls(self.model)
        self.splitter = DeepSurvSplitter(self.splitter)
    return self


import pysurvival
from scipy.integrate import trapz
from pysurvival.models._coxph import _baseline_functions
from pysurvival.utils._functions import _get_time_buckets

def get_survival_estimation(learn, train_ds_idx=0, target_ds_idx=1):
    "Estimates survival for models that output a risk scores (follows pysurvival library)"
    preds_train,lbls_train = learn.get_preds(ds_idx=train_ds_idx)
    preds_valid,lbls_valid = learn.get_preds(ds_idx=target_ds_idx)

    T_train = to_np(lbls_train.surv)
    E_train = (1 - to_np(lbls_train.cens))
    times = np.unique(T_train[E_train.astype(bool)])
    time_buckets = _get_time_buckets(times)
    buckets = [o[0] for o in time_buckets] + [time_buckets[-1][1]]
    phi_train = np.exp(to_np(learn.loss_func.decodes(preds_train)))
    order_train = np.argsort(-T_train)
    baselines = _baseline_functions(phi_train[order_train], T_train[order_train], E_train[order_train])
    baseline_hazard = np.array( baselines[1] )
    baseline_survival = np.array( baselines[2] )

    T_valid = to_np(lbls_valid.surv)
    E_valid = (1 - to_np(lbls_valid.cens))
    phi_valid = np.exp(to_np(learn.loss_func.decodes(preds_valid)))
    survival = np.power(baseline_survival, phi_valid.reshape(-1,1))
    expectation = trapz(survival, buckets)
    return {'expectation': expectation, 'target': T_valid, 'censored': to_np(lbls_valid.cens)}


from ..metrics import *

@delegates(get_survival_estimation)
def get_survival_time_metrics(learn, patient_ids=None, **kwargs):
    "Returns survival time metrics (MAE) and the survival estimations"
    data = get_survival_estimation(learn)
    output = tensor(data['expectation'])
    target = TensorSurvival(np.stack([data['target'], data['censored']], 1))
    uncensored_mask = target.cens==0
    res = {'mae': mae_cens(output, target).item(),
           'mae_only_uncensored': mae_cens(output[uncensored_mask], target[uncensored_mask]).item()}
    if patient_ids is not None:
        patient_ids = tensor(patient_ids)
        fn = PatientWiseMaeCens(patient_ids)
        res['pw_mae'] = fn(output, target).item()
        fn = PatientWiseMaeCens(patient_ids[uncensored_mask])
        res['pw_mae_only_uncensored'] = fn(output[uncensored_mask], target[uncensored_mask]).item()
    return res,data