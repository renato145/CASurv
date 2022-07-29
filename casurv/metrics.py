__all__ = ['PatientWiseMetric']


from fastai.basics import *


@delegates(AccumMetric)
def PatientWiseMetric(func, patient_ids, cls=AccumMetric, **kwargs):
    '''It first calculate the mean of the metric by patient and then the mean of means
    `func` should not reduce its output
    Returns nan when the number of `len(ids) != len(output)`
    '''
    assert issubclass(cls, AccumMetric)
    try: ids = tensor(patient_ids)
    except: ids = tensor(pd.Series(patient_ids).astype('category').cat.codes.tolist())
    unique_ids = ids.unique()
    def _func(output, target, ids, unique_ids):
        res = func(output, target)
        if len(res) != len(ids): return np.nan
        return tensor([res[ids==i].mean() for i in unique_ids]).mean()

    _func.__name__ = func.func.__name__ if hasattr(func, 'func') else  func.__name__
    return cls(_func, ids=ids, unique_ids=unique_ids, **kwargs)