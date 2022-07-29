__all__ = ['lse_pool1d', 'LSELBAPool1d', 'lse_pool', 'LSELBAPool', 'lse_pool3d', 'LSELBAPool3d', 'create_cnn_metadata']


from fastai.vision.all import *
from fastai.vision.learner import _default_meta


def lse_pool1d(x:Tensor, r:float=1.0)->Tensor:
    n = x.size(-1)
    theta = tensor(n).float().to(x.device).log()
    return x.mul(r).logsumexp((-1)).sub(theta).div(r)


class LSELBAPool1d(Module):
    def __init__(self, r0:float=0.0):
        "Log-Sum-Exp Pooling with Lower-bounded Adaptation (LSE-LBA Pool) (from https://arxiv.org/abs/1803.07703)"
        self.r0 = r0
        self.beta = nn.Parameter(tensor([0.]))
        self.pool = lse_pool1d

    @property
    def r(self)->float:
        with torch.no_grad(): r = self.beta.exp().add(self.r0).item()
        return r

    def __repr__(self):
        return f'{self.__class__.__name__} (r0={self.r0:.2f}, beta={self.beta.item():.4f} -> r={self.r:.4f})'

    def forward(self, x):
        r = self.beta.exp().add(self.r0)
        return self.pool(x, r)


def lse_pool(x:Tensor, r:float=1.0)->Tensor:
    h,w = x.shape[-2:]
    theta = tensor(h*w).float().to(x.device).log()
    return x.mul(r).logsumexp((-2,-1)).sub(theta).div(r)


class LSELBAPool(LSELBAPool1d):
    def __init__(self, r0:float=0.0):
        super().__init__(r0)
        self.pool = lse_pool

LSELBAPool.__doc__ = LSELBAPool1d.__doc__


def lse_pool3d(x:Tensor, r:float=1.0)->Tensor:
    h,w,d = x.shape[-3:]
    theta = tensor(h*w*d).float().to(x.device).log()
    return x.mul(r).logsumexp((-3,-2,-1)).sub(theta).div(r)


class LSELBAPool3d(LSELBAPool1d):
    def __init__(self, r0:float=0.0):
        super().__init__(r0)
        self.pool = lse_pool3d

LSELBAPool.__doc__ = LSELBAPool1d.__doc__


@delegates(create_head)
def create_cnn_metadata(arch, n_out, meta_fts, meta_layers=(512,128), pretrained=True, cut=None, n_in=3,
                        init=nn.init.kaiming_normal_, custom_head=None, concat_pool=True, **kwargs):
    '''Creates custom architecture that takes as input images and metadata (continuous values)
    Returns: body_cnn,body_metadata,head
    '''
    meta = model_meta.get(arch, _default_meta)
    body_cnn = create_body(arch, n_in, pretrained, ifnone(cut, meta['cut']))

    lin_ftrs = [meta_fts] + listify(meta_layers)
    bns = [True]*len(lin_ftrs[2:]) + [False]
    layers = []
    for ni,no,bn in zip(lin_ftrs[:-1], lin_ftrs[1:], bns):
        layers += LinBnDrop(ni, no, bn=bn, act=nn.ReLU(inplace=True), lin_first=True)
    body_metadata = nn.Sequential(*layers)

    nf = num_features_model(nn.Sequential(*body_cnn.children()))
    if concat_pool: nf *= 2
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    body_cnn = nn.Sequential(body_cnn, *layers)

    if custom_head is None:
        nf += meta_layers[-1]
        head = nn.Sequential(*flatten_model(create_head(nf, n_out, concat_pool=False, **kwargs))[2:])
    else: head = custom_head

    return body_cnn,body_metadata,head