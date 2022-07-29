__all__ = ['get_bin_weights', 'SigMix', 'from_ds', 'SigMixMetadata', 'from_ds', 'sigmix_metadata_splitter',
           'SigMixLoss', 'SigMixCALoss', 'ELRSigMixCallback', 'ELRSigMixLoss', 'ELRSigMixCALoss',
           'ELRPlusSigMixCallback', 'ELRPlusSigMixLoss', 'ELRPlusSigMixCALoss']


from fastai.vision.all import *
from ...basics import *
from ..core import *
from ..data import *


@delegates(make_cuts)
def get_bin_weights(x, bins, equidistant=True, **kwargs):
    cuts =  make_cuts(x, bins+1, equidistant=equidistant, **kwargs)
    weights = cuts[1:] - cuts[:-1]
    return weights


class SigMix(Module):
    "From paper: Post-hoc Overall Survival Time Prediction from Brain MRI"
    def __init__(self, base_model, bin_weights:FloatTensor):
        store_attr()
        self.base = bin_weights.sum()

    @property
    def n_bins(self): return self.bin_weights.size(0)
    @property
    def logits_layer(self): return self.base_model

    def get_bins(self, x):
        '''For high activations to represent risk, we substract them from a base value.
        Otherwise high activations will represent a higher survival.'''
        d = x.device
        return self.bin_weights.to(d) - self.base_model(x).sigmoid().mul(self.bin_weights.to(d))

    def predict_with_bins(self, x, show_bins=False, y_true=None):
        "Returns predictions and computed bins"
        bins = self.get_bins(x)
        preds = bins.sum(dim=1)
        if show_bins:
            df = pd.DataFrame(bins)
            subset = df.columns
            df['surv'] = preds.data.cpu()

            if y_true is not None:
                target = y_true.data.cpu()
                df['y_true'] = target.surv
                df['cens'] = target.cens.long()
                df['mae'] = np.abs(df['surv'] - df['y_true'])

            d = df.style.background_gradient(subset='surv', axis=0, vmin=0, vmax=self.base, cmap='RdYlGn')

            for i,col in enumerate(subset):
                d = d.bar(subset=[col], axis=0, vmin=0, vmax=self.bin_weights[i].item())

            if y_true is not None:
                d = (d.background_gradient(subset='y_true', axis=0, vmin=0, vmax=self.base, cmap='RdYlGn')
                      .background_gradient(subset='mae', axis=0, vmin=0, cmap='RdYlGn_r'))

            display(d)

        return preds,bins

    def forward(self, x): return self.predict_with_bins(x)[0]


@patch(cls_method=True)
@delegates(get_bin_weights, but='min_,max_')
def from_ds(cls: SigMix, base_model, ds, bins, max_surv_mul=1.0, tls_idx=-1, **kwargs):
    '''Gets bins from `ds` and invert the order so the last bin indicates more risk.
    To create the bins we need a maximum number of survival, this can be infered from the
    dataset and use `max_surv_mul` to allow for a bigger window in the following way:
    `max_surv = max(survival_times) * max_surv_mul`
    '''
    survival_times = [o.item() for o,_ in ds.tls[tls_idx]]
    max_surv = max(survival_times) * max_surv_mul
    bin_weights = get_bin_weights(survival_times, bins, max_=max_surv, **kwargs).flip(0)
    return cls(base_model, bin_weights)


@patch
@delegates(SigMix.from_ds)
def setup_sigmix(self: Learner, bins, **kwargs):
    '''Gets bins from train_ds and invert the order so the last bin indicates more risk.
    To create the bins we need a maximum number of survival, this can be infered from the
    dataset and use `max_surv_mul` to allow for a bigger window in the following way:
    `max_surv = max(survival_times) * max_surv_mul`
    '''
    self.model = SigMix.from_ds(self.model, self.dls.train_ds, bins=bins, **kwargs)
    return self


class SigMixMetadata(SigMix):
    "`SigMix` with inclusion of metadata"
    def __init__(self, body_cnn, body_metadata, head, bin_weights:FloatTensor):
        store_attr()
        self.base = bin_weights.sum()

    @property
    def logits_layer(self): return self.head

    def get_bins(self, x):
        '''For high activations to represent risk, we substract them from a base value.
        Otherwise high activations will represent a higher survival.'''
        x,metadata = x
        d = x.device
        img_fts = self.body_cnn(x)
        met_fts = self.body_metadata(metadata)
        fts = torch.cat([img_fts, met_fts], dim=1)
        return self.bin_weights.to(d) - self.head(fts).sigmoid().mul(self.bin_weights.to(d))

    def predict_with_bins(self, x, metadata, show_bins=False, y_true=None):
        "Returns predictions and computed bins"
        return super().predict_with_bins((x,metadata), show_bins=show_bins, y_true=y_true)

    def forward(self, x, metadata): return self.predict_with_bins(x, metadata)[0]


@patch(cls_method=True)
@delegates(get_bin_weights, but='min_,max_')
def from_ds(cls: SigMixMetadata, body_cnn, body_metadata, head, ds, bins, max_surv_mul=1.0, tls_idx=-1, **kwargs):
    '''Gets bins from `ds` and invert the order so the last bin indicates more risk.
    To create the bins we need a maximum number of survival, this can be infered from the
    dataset and use `max_surv_mul` to allow for a bigger window in the following way:
    `max_surv = max(survival_times) * max_surv_mul`
    '''
    survival_times = [o.item() for o,_ in ds.tls[tls_idx]]
    max_surv = max(survival_times) * max_surv_mul
    bin_weights = get_bin_weights(survival_times, bins, max_=max_surv, **kwargs).flip(0)
    return cls(body_cnn, body_metadata, head, bin_weights)


def sigmix_metadata_splitter(m):
    "To use for `Learner` model split"
    return L(params(m.body_cnn), params(m.body_metadata)+params(m.head))


class SigMixLoss(Module):
    '''From paper: Post-hoc Overall Survival Time Prediction from Brain MRI

    Params:
    - rank_func: if given is used to calculate a ranking loss.
    - penalty: factor to multiply to the result of the `SigMixLoss` penalty.
    - main_penalty: factor to multiply to the result of the main `loss_func`.
    - rank_penalty: factor to multiply to the result of `rank_func`.
    - log: applies a log function to the result of the main `loss_func` to reduce its impact
      (log is applied after `main_penalty`).

    Notes:
    - On isbi paper we used `penalty`=10_000
    '''
    def __init__(self, logits_layer, loss_func=nn.MSELoss(), rank_func=None, penalty=100, main_penalty=1,
                 rank_penalty=1, log=False):
        "If possible sets `loss_func.reduction` to `'none'`"
        store_attr(but='logits_layer')
        self.hook = hook_output(logits_layer, detach=False)
        if hasattr(self.loss_func, 'reduction'): setattr(self.loss_func, 'reduction', 'none')
        self.main_loss_weights = None

    __repr__ = basic_repr(['loss_func', 'rank_func', 'penalty', 'main_penalty', 'rank_penalty', 'log'])
    @property
    def do_sigmix(self): return self.penalty > 0
    @property
    def do_rank(self): return (self.rank_func is not None) and (self.rank_penalty > 0)

    def get_main_loss(self, output, target):
        surv = target.surv
        return self.loss_func(output.flatten(), surv.flatten())

    def forward(self, output, target):
        logits = self.hook.stored.sigmoid()
        loss = self.get_main_loss(output, target)
        if self.main_loss_weights is not None: loss = loss * self.main_loss_weights.to(loss.device)
        loss = loss.mean().mul(self.main_penalty)
        if self.log: loss = loss.add(1e-7).log()
        self.main_loss = loss
        self.sm_penalty = ((logits[:,1:] - logits[:,:-1]).relu().sum(dim=1).mul(self.penalty).mean()
                           if self.do_sigmix else 0)
        self.rank_loss = self.rank_func(output, target).mul(self.rank_penalty) if self.do_rank else 0
        return self.main_loss + self.sm_penalty + self.rank_loss

    def __del__(self)->None: self.hook.remove()


class SigMixCALoss(SigMixLoss):
    def get_main_loss(self, output, target):
        surv = target.surv
        cens = target.cens
        no_cens_idxs = torch.where((output > surv) & (cens == 1))[0]
        loss = self.loss_func(output.flatten(), surv.flatten())
        loss[no_cens_idxs] = 0
        return loss

SigMixCALoss.__doc__ =  'Censored aware version of `SigMixLoss` (ignores loss when output > target)'\
                       f'. {SigMixLoss.__doc__}'


@patch
@delegates(SigMixLoss, but=['logits_layer'])
def setup_sigmix_loss(self: Learner, censor_aware=True, logits_layer=None, **kwargs):
    logits_layer = ifnone(logits_layer, getattr(self.model, 'logits_layer', None))
    assert logits_layer is not None

    # loss
    f = SigMixCALoss if censor_aware else SigMixLoss
    self.loss_func = f(logits_layer, **kwargs)

    # metrics
    metrics = 'main_loss'
    if getattr(self.loss_func, 'do_sigmix', False): metrics += ',sm_penalty'
    if getattr(self.loss_func, 'do_rank'  , False): metrics += ',rank_loss'
    self._metrics += LossMetrics(metrics)

    return self

Learner.setup_sigmix_loss.__doc__ = SigMixLoss.__doc__


@delegates()
class ELRSigMixCallback(ELRCallback):
    "SigMix version of `ELRCallback`"
    def __init__(self, logits_layer, **kwargs):
        super().__init__(**kwargs)
        self.hook = hook_output(logits_layer, detach=False)

    def after_pred(self):
        if self.training:
            idxs = self.get_batch_idxs()
            # Get the prediction from the `logits_layer`
            y_pred = self.hook.stored.sigmoid().clamp(1e-4, 1.0-1e-4)
            y_pred_ = y_pred.data.detach().cpu()

            self.ensemble_predictions[idxs] = (     self.beta  * self.ensemble_predictions[idxs] +
                                               (1 - self.beta) * y_pred_)
            dot_prod = (self.ensemble_predictions[idxs].to(y_pred.device) * y_pred).sum(dim=1)
            elr_reg = (( 1 - dot_prod/self.c ).log()).mean()
            self.elr_reg = self.alpha*elr_reg

    def __del__(self)->None: self.hook.remove()


class ELRSigMixLoss(SigMixLoss):
    "`SigMixLoss` with ELR regularizer"
    @delegates(SigMixLoss.__init__)
    def __init__(self, logits_layer, elr_cb, **kwargs):
        super().__init__(logits_layer, **kwargs)
        self.elr_cb = elr_cb

    def forward(self, output, target):
        loss = super().forward(output, target)
        self.elr_reg = self.elr_cb.elr_reg
        return loss + self.elr_reg


class ELRSigMixCALoss(SigMixCALoss):
    "`SigMixCALoss` with ELR regularizer"
    @delegates(SigMixCALoss.__init__)
    def __init__(self, logits_layer, elr_cb, **kwargs):
        super().__init__(logits_layer, **kwargs)
        self.elr_cb = elr_cb

    def forward(self, output, target):
        loss = super().forward(output, target)
        self.elr_reg = self.elr_cb.elr_reg
        return loss + self.elr_reg


@patch
@delegates(ELRSigMixLoss, but=['logits_layer', 'elr_cb'])
def setup_sigmix_elr_loss(self: Learner, censor_aware=True, logits_layer=None, elr_alpha=1.0, elr_beta=0.3, **kwargs):
    "Setups SigMixLoss with ELR regularizer"
    logits_layer = ifnone(logits_layer, getattr(self.model, 'logits_layer', None))
    assert logits_layer is not None

    # callback
    elr_cb = ELRSigMixCallback(logits_layer, alpha=elr_alpha, beta=elr_beta, c=self.model.n_bins)
    self.add_cb(elr_cb)

    # loss
    f = ELRSigMixCALoss if censor_aware else ELRSigMixLoss
    self.loss_func = f(logits_layer, elr_cb, **kwargs)

    # metrics
    metrics = 'main_loss'
    if getattr(self.loss_func, 'do_sigmix', False): metrics += ',sm_penalty'
    if getattr(self.loss_func, 'do_rank'  , False): metrics += ',rank_loss'
    metrics += ',elr_reg'
    self._metrics += LossMetrics(metrics)

    return self


@delegates()
class ELRPlusSigMixCallback(ELRPlusCallback):
    "SigMix version of `ELRPlusCallback`"
    def __init__(self, logits_layer_m1, logits_layer_m2, logits_layer_ema1, logits_layer_ema2, **kwargs):
        super().__init__(**kwargs)
        self.hook_m1   = hook_output(logits_layer_m1, detach=False)
        self.hook_m2   = hook_output(logits_layer_m2, detach=False)
        self.hook_ema1 = hook_output(logits_layer_ema1)
        self.hook_ema2 = hook_output(logits_layer_ema2)

    def before_train(self)   : self.learn.loss_func.train()
    def before_validate(self): self.learn.loss_func.eval()

    def after_pred(self):
        if self.training:
            idxs = self.get_batch_idxs()
            current_m = 'm1' if self.m1_active else 'm2'
            m_hook   = getattr(self, 'hook_m1'   if self.m1_active else 'hook_m2')
            ema = getattr(self.learn.model, 'ema2' if self.m1_active else 'ema1')
            ema_hook = getattr(self, 'hook_ema2' if self.m1_active else 'hook_ema1')

            # Update self.ensemble_predictions
            with torch.no_grad(): _ = ema(*self.xb)
            y_pred_ = ema_hook.stored.data.sigmoid().cpu()
            self.ensemble_predictions[current_m][idxs] = (
                self.beta  * self.ensemble_predictions[current_m][idxs] +
                (1 - self.beta) *  y_pred_
            )

            # Get the prediction from the `logits_layer`
            y_pred = m_hook.stored.sigmoid().clamp(1e-4, 1.0-1e-4)
            dot_prod = (self.ensemble_predictions[current_m][idxs].to(y_pred.device) * y_pred).sum(dim=1)
            elr_reg = (( 1 - dot_prod/self.c ).log()).mean()
            self.elr_reg = self.alpha*elr_reg

    def after_epoch(self):
        self.learn.model.switch()
        self.learn.loss_func.switch()

    def __del__(self)->None:
        self.hook_m1.remove()
        self.hook_m2.remove()
        self.hook_ema1.remove()
        self.hook_ema2.remove()


class ELRPlusSigMixLoss(Module):
    "`SigMixLoss` with ELR+ regularizer"
    @delegates(SigMixLoss.__init__)
    def __init__(self, logits_layer_m1, logits_layer_m2, elr_cb, **kwargs):
        self.loss1 = SigMixLoss(logits_layer_m1, **kwargs)
        self.loss2 = SigMixLoss(logits_layer_m2, **kwargs)
        self.elr_cb = elr_cb
        self.m1_active = True

    __repr__ = basic_repr(['loss1', 'loss2', 'elr_cb'])

    def forward(self, output, target):
        loss_func = self.loss1 if self.m1_active else self.loss2
        loss = loss_func(output, target)
        for o in ['main_loss','sm_penalty','rank_loss']: setattr(self, o, getattr(loss_func, o))
        self.elr_reg = self.elr_cb.elr_reg
        return loss + self.elr_reg

    def switch(self, m1_active=None):
        "Switch active loss"
        self.m1_active = (not self.m1_active) if m1_active is None else m1_active


class ELRPlusSigMixCALoss(Module):
    "`SigMixCALoss` with ELR+ regularizer"
    @delegates(SigMixCALoss.__init__)
    def __init__(self, logits_layer_m1, logits_layer_m2, elr_cb, **kwargs):
        self.loss1 = SigMixCALoss(logits_layer_m1, **kwargs)
        self.loss2 = SigMixCALoss(logits_layer_m2, **kwargs)
        self.elr_cb = elr_cb
        self.m1_active = True

    __repr__ = basic_repr(['loss1', 'loss2', 'elr_cb'])

    def forward(self, output, target):
        loss_func = self.loss1 if self.m1_active else self.loss2
        loss = loss_func(output, target)
        for o in ['main_loss','sm_penalty','rank_loss']: setattr(self, o, getattr(loss_func, o))
        self.elr_reg = self.elr_cb.elr_reg
        return loss + self.elr_reg

    def switch(self, m1_active=None):
        "Switch active loss"
        self.m1_active = (not self.m1_active) if m1_active is None else m1_active


@patch
@delegates(ELRPlusSigMixLoss, but=['logits_layer', 'elr_cb'])
def setup_sigmix_elrplus_loss(self: Learner, model2, censor_aware=True, elr_alpha=1.0, elr_beta=0.3,
                              elr_gamma=0.997, **kwargs):
    "Setups SigMixLoss with ELR+ regularizer"
    # ELRPlusModel duplicates both models for ema calcs
    self.model = ELRPlusModel(self.model, model2)
    self.model.n_bins = self.model.m1.n_bins
    logits_layer_m1 = self.model.m1.logits_layer
    logits_layer_m2 = self.model.m2.logits_layer
    logits_layer_ema1 = self.model.ema1.logits_layer
    logits_layer_ema2 = self.model.ema2.logits_layer

    # callback
    elr_cb = ELRPlusSigMixCallback(logits_layer_m1, logits_layer_m2, logits_layer_ema1, logits_layer_ema2,
                                   alpha=elr_alpha, beta=elr_beta, gamma=elr_gamma, c=self.model.n_bins)
    self.add_cb(elr_cb)

    # loss
    f = ELRPlusSigMixCALoss if censor_aware else ELRPlusSigMixLoss
    self.loss_func = f(logits_layer_m1, logits_layer_m2, elr_cb, **kwargs)

    # optimizer splitter
    def get_logits_layer(m): return m.logits_layer
    if self.splitter != trainable_params: elr_set_splitter(self, get_logits_layer)

    # metrics
    metrics = 'main_loss'
    if getattr(self.loss_func.loss1, 'do_sigmix', False): metrics += ',sm_penalty'
    if getattr(self.loss_func.loss1, 'do_rank'  , False): metrics += ',rank_loss'
    metrics += ',elr_reg'
    self._metrics += LossMetrics(metrics)

    return self