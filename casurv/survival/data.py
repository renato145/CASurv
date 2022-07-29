__all__ = ['SurvivalLabel', 'TensorSurvival', 'SurvivalSetup', 'SurvivalBlock']


from fastai.vision.all import *
from ..basics import *


class SurvivalLabel:
    def __init__(self, surv, cens, is_hazard=False): store_attr()
    __repr__ = basic_repr('surv,cens,is_hazard')
    def show(self, ctx=None, color='black', **kwargs):
        cens = int(self.cens)
        assert cens in [0, 1], f'Invalid censored label: {cens}'
        msg = f'{self.surv:.2f}'
        if int(self.cens) == 1: msg += ' (censored)'
        return show_title(msg, ctx=ctx, color=color, **kwargs)

    def __eq__(self, other): return (self.surv==other.surv) and (self.cens==other.cens)


class TensorSurvival(TensorBase):
    @property
    def surv(self): return self[0] if self.ndim == 1 else self[:,0]
    @property
    def cens(self): return self[1] if self.ndim == 1 else self[:,1]
    @property
    def E(self): return 1 - self.cens

TensorSurvival.register_func(Tensor.__getitem__)


class SurvivalSetup(ItemTransform):
    def __init__(self, is_hazard=False):
        self.c = 1
        store_attr()
    def encodes(self, o): return TensorSurvival(o).float()
    def decodes(self, o):
        return (SurvivalLabel(o.item(), 0, is_hazard=self.is_hazard) if o.numel()==1 else
                SurvivalLabel(*[o_.item() for o_ in o], is_hazard=self.is_hazard))


def SurvivalBlock(is_hazard=False):
    "`TransformBlock` for survival problems: (survival_time, censored)"
    return TransformBlock(type_tfms=SurvivalSetup(is_hazard=is_hazard))


@typedispatch
def show_results(x:TensorImage, y:TensorSurvival, samples, outs, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, add_vert=1, figsize=figsize)
    lbls = samples.itemgot(-1)
    preds = outs.itemgot(-1)
    censored_idxs = lbls.enumerate().filter(lambda o: o[1].cens==1).map(lambda o: o[0])
    uncensored_mask = lbls.map(lambda o: o.cens==0)
    is_hazard = first(preds).is_hazard

    def _get_ranks(x, inverse=False):
        x = np.asarray(x[uncensored_mask].attrgot('surv'))
        if inverse: x = -x
        # 2xargsort = rank
        return L((np.argsort(x).argsort() + 1).tolist()).map(lambda o: f'# {o}')
    def _insert_uncensored(x):
        for i in censored_idxs: x.insert(i, '-')
        return x

    rank_lbls = _insert_uncensored(_get_ranks(lbls))
    rank_preds = _insert_uncensored(_get_ranks(preds, inverse=is_hazard))
    lbls = [f'{a} | {b}' for a,b in zip(rank_lbls,rank_preds)]
    colors = ['black' if a=='-' else 'green' if a==b else 'red' for a,b in zip(rank_lbls,rank_preds)]
    ctxs = show_results[object](x, y, samples, outs, ctxs=ctxs, max_n=max_n, **kwargs)
    for ctx,lbl,color in zip(ctxs,lbls,colors): show_title(lbl, ctx=ctx, color=color)
    return ctxs

@typedispatch
def show_results(x:object, y:TensorSurvival, samples, outs, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    return show_results[TensorImage,TensorSurvival](x, y, samples, outs, ctxs=ctxs, max_n=max_n, nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)