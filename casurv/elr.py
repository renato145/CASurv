__all__ = ['ELRCallback', 'ELRPlusModel', 'ELRPlusCallback', 'elr_set_splitter']


from fastai.vision.all import *


def _get_batch_idxs(self): return self.dl._DataLoader__idxs[self.iter*self.dl.bs:self.iter*self.dl.bs+self.dl.bs]

class ELRCallback(Callback):
    '''Early-Learning Regularization (https://arxiv.org/abs/2007.00151)
    This `Callback` stores and keep track of the ensemble predictions'''
    def __init__(self, alpha=1.0, beta=0.3, c=None):
        '''
        - alpha: regularization parameter
        - beta: temporal ensembling momentum
        - c: number of classes
        '''
        store_attr()
        self.ensemble_predictions = None

    __repr__ = basic_repr('alpha,beta,c')

    def before_fit(self):
        if self.ensemble_predictions is None:
            n = len(self.learn.dls.train.dataset)
            c = self.learn.dls.c if self.c is None else self.c
            self.ensemble_predictions = torch.zeros(n,c)

    def after_pred(self):
        "Calculate loss"
        if self.training:
            idxs = self.get_batch_idxs()
            # From paper code:
            y_pred = self.pred.softmax(dim=1).clamp(1e-4, 1.0-1e-4)
            y_pred_ = y_pred.data.detach().cpu()
            self.ensemble_predictions[idxs] = (     self.beta  * self.ensemble_predictions[idxs] +
                                               (1 - self.beta) * ( y_pred_ / (y_pred_).sum(dim=1, keepdim=True)))
            elr_reg = (( 1 - (self.ensemble_predictions[idxs].to(y_pred.device) * y_pred).sum(dim=1) )
                       .log()).mean()
            self.elr_reg = self.alpha*elr_reg

ELRCallback.get_batch_idxs = _get_batch_idxs


class ELRPlusModel(Module):
    "Wrapper to use 2 models and switch between them"
    def __init__(self, m1, m2):
        self.m1,self.m2 = m1,m2
        self.ema1,self.ema2 = deepcopy(m1),deepcopy(m2)
        for param in self.ema1.parameters(): param.detach_()
        for param in self.ema2.parameters(): param.detach_()
        self.m1_active = True
        self.eval_mode = True
        self.steps = {'m1': 0, 'm2': 0}

    def forward(self, *args):
        if self.training: return self.m1(*args) if self.m1_active else self.m2(*args)
        else            : return (self.m1(*args) + self.m2(*args)) / 2

    def switch(self, m1_active=None):
        "Switch active model"
        self.m1_active = (not self.m1_active) if m1_active is None else m1_active

    def update_ema(self, gamma):
        "Update weight averages"
        current_m = 'm1' if self.m1_active else 'm2'
        gamma = min(1 - 1 / (self.steps[current_m] + 1), gamma)
        m = getattr(self, current_m)
        ema = getattr(self, 'ema2' if self.m1_active else 'ema1')

        for ema_param, param in zip(ema.parameters(), m.parameters()):
            ema_param.data.mul_(gamma).add_(param.data, alpha = 1 - gamma)

        self.steps[current_m] += 1


class ELRPlusCallback(Callback):
    '''Early-Learning Regularization ELR+ (https://arxiv.org/abs/2007.00151)'''
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.997, c=None):
        '''
        - alpha: regularization parameter
        - beta: temporal ensembling momentum
        - gamma: weight averaging momentum
        - c: number of classes
        '''
        store_attr()
        self.ensemble_predictions = None

    __repr__ = basic_repr('alpha,beta,c')

    @property
    def m1_active(self): return self.learn.model.m1_active

    def before_fit(self):
        self.learn.model.switch(True)
        if self.ensemble_predictions is None:
            n = len(self.learn.dls.train.dataset)
            c = self.learn.dls.c if self.c is None else self.c
            self.ensemble_predictions = {'m1': torch.zeros(n,c), 'm2': torch.zeros(n,c)}

    def after_pred(self):
        "Calculate loss"
        if self.training:
            idxs = self.get_batch_idxs()
            current_m = 'm1' if self.m1_active else 'm2'
            ema = getattr(self.learn.model, 'ema2' if self.m1_active else 'ema1')

            # Update self.ensemble_predictions
            with torch.no_grad(): y_pred_ = ema(*self.xb).data.detach().softmax(dim=1).cpu()
            self.ensemble_predictions[current_m][idxs] = (
                self.beta  * self.ensemble_predictions[current_m][idxs] +
                (1 - self.beta) * ( y_pred_ / (y_pred_).sum(dim=1, keepdim=True))
            )

            # Get Loss:
            y_pred = self.pred.softmax(dim=1).clamp(1e-4, 1.0-1e-4)
            elr_reg = (( 1 - (self.ensemble_predictions[current_m][idxs].to(y_pred.device) * y_pred).sum(dim=1) )
                       .log()).mean()
            self.elr_reg = self.alpha*elr_reg

    def after_step(self): self.learn.model.update_ema(self.gamma)
    def after_epoch(self): self.learn.model.switch()

ELRPlusCallback.get_batch_idxs = _get_batch_idxs


def elr_set_splitter(learn, model_selector=noop):
    "Set up proper splitter for ELR+"
    orig_splitter = deepcopy(learn.splitter)
    def splitter(m): return L(L(*a,*b) for a,b in
                              zip(orig_splitter(model_selector(m.m1)), orig_splitter(model_selector(m.m2))))
    learn.splitter = splitter
    learn.create_opt()