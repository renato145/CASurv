__all__ = ['mae_surv', 'SurvivalPatientWiseMetric', 'PatientWiseMaeSurv', 'mae_cens', 'PatientWiseMaeCens', 'c_index',
           'PatientWiseCIndex']


from fastai.basics import *
from ..basics import *
from .core import *
from .data import *


def mae_surv(output, target, do_mean=True):
    "Simple mae error ignoring censored information"
    inp,targ = flatten_check(output, target.surv)
    out = torch.abs(inp - targ)
    return out.mean() if do_mean else out


def _pw_fix(output, target):
    '''This function expects `preds` and `targ` belonging to the same patient.
    This helps fixing predictions and target for patient-wise calculations when patients have
    different survival targets.
    Per each patient a padding is calculated using the maximum survival target, then we add this
    padding to the predictions. Doing so will make all the predictions start on the same time, so
    the same (maximum) target can be used.'''
    surv_max = target.max(dim=0)[0]
    new_output = (surv_max.surv - target.surv) + output
    return new_output.mean(),surv_max


def SurvivalPatientWiseMetric(func, patient_ids, cls=AccumMetric, **kwargs):
    '''It first calculates the mean target and prediction for each patient_id.
    It fixes cases of different survivals for the same patient (check documentation)
    `func` should not reduce its output
    Returns nan when the number of `len(ids) != len(output)`
    '''
    assert issubclass(cls, AccumMetric)
    try: ids = tensor(patient_ids)
    except: ids = tensor(pd.Series(patient_ids).astype('category').cat.codes.tolist())
    unique_ids = ids.unique()
    def _func(output, target, ids, unique_ids):
        if len(target) != len(ids): return np.nan
        new_output,new_target = zip(*[_pw_fix(output[ids==i],target[ids==i]) for i in ids.unique()])
        new_output = tensor(new_output)
        new_target = torch.stack(new_target, dim=0)
        return func(new_output, new_target)

    _func.__name__ = 'pw_' + (func.func.__name__ if hasattr(func, 'func') else  func.__name__)
    return cls(_func, ids=ids, unique_ids=unique_ids, flatten=False, **kwargs)


def PatientWiseMaeSurv(patient_ids):
    "Patient-wise version of `mae_surv`, it first calculates the mean survival for each patient_id"
    return SurvivalPatientWiseMetric(mae_surv, patient_ids)


def mae_cens(output, target, do_mean=True):
    "mae error considering censored information"
    surv,cens = target.surv,target.cens
    ind = output < surv
    out = (surv - output) * ind
    out += (1 - cens) * (output - surv) * ~ind
    return out.mean() if do_mean else out


def PatientWiseMaeCens(patient_ids):
    "Patient-wise version of `mae_cens`, it first calculates the mean survival for each patient_id"
    return SurvivalPatientWiseMetric(mae_cens, patient_ids)


from ._lifelines_cindex import concordance_index

def _c_index(output, target): return concordance_index(target.surv, output, 1-target.cens)
_c_index.__name__ = 'c_index'
c_index = AccumMetric(_c_index, flatten=False)
c_index.__doc__ = "Concordance index using lifelines (https://github.com/CamDavidsonPilon/lifelines)"


def PatientWiseCIndex(patient_ids):
    "Patient-wise version of `c_index`, it first calculates the mean survival for each patient_id"
    return SurvivalPatientWiseMetric(c_index, patient_ids)