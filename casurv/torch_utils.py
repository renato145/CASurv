__all__ = ['str2arch', 'IgnoreIdModel', 'AvgIdModel', 'SetAvgIdModel', 'replace_layers', 'replace_layers_types',
           'modify_layers', 'modify_layers_types']


from fastai.vision.all import *


def str2arch(model_name):
    "Get model arch from the string `name`"
    arch = models
    for o in model_name.split('.'): arch = getattr(arch, o)
    return arch


class IgnoreIdModel(Module):
    "Wraps a model to ignore patient IDs on batches"
    _default='model'
    def __init__(self, model, id_idx=-1): store_attr()
    def forward(self, *x):
        idx = len(x) + self.id_idx if self.id_idx < 0 else self.id_idx
        x = [_x for i,_x in enumerate(x) if i != idx]
        return self.model(*x)


@patch
def setup_ignore_id(self: Learner, id_idx=-1):
    "Wraps the model with `IgnoreIdModel` to ignore patient IDs on batches"
    self.model = IgnoreIdModel(self.model, id_idx=id_idx)
    return self


class AvgIdModel(Module):
    '''Wraps a model to output average results based on given `ids` (`ids` should be ints).
    The `forward` methods expects 2 inputs: `ids` and `x`. And outputs, per each element in `x`,
    the average result for its corresponding id.'''
    def __init__(self, model, dim=-1, avg_enable=True): store_attr()
    def forward(self, ids, *x):
        ids = ids.long()
        out = self.model(*x)
        if not self.avg_enable: return out
        d = {id.item(): out[id==ids].mean() for id in ids.unique()}
        out = torch.stack([d[id.item()] for id in ids], dim=self.dim)
        return out


class SetAvgIdModel(Callback):
    "Sets `AvgIdModel` only while training"
    def after_create(self): self.model.avg_enable = False
    def before_train(self): self.model.avg_enable = True
    def after_train (self): self.model.avg_enable = False

    @classmethod
    def setup(cls, learn):
        "Setups `AvgIdModel` on the `Learner` model and adds this callback to make it work only while training"
        learn.model = AvgIdModel(learn.model)
        learn.add_cb(cls())
        learn.model.avg_enable = False
        return learn


def replace_layers(m, new_layer_fn, filter_fn=noop):
    "Recursively replace layers with a `new_layer_fn` according to a `filter_fn`"
    is_sequential = isinstance(m, nn.Sequential)
    it = enumerate(m.children()) if is_sequential else m.named_children()
    for name,layer in it:
        if filter_fn(layer):
            if is_sequential: m[name] = new_layer_fn()
            else            : setattr(m, name, new_layer_fn())
        elif has_children(layer):
            replace_layers(layer, new_layer_fn, filter_fn)

    return m


def replace_layers_types(m, new_layer_fn, replace_types):
    "Recursively replace layers with a `new_layer_fn` according to types `replace_types`"
    replace_types = listify(replace_types)
    def filter_types(layer): return any(isinstance(layer,o) for o in replace_types)
    return replace_layers(m, new_layer_fn, filter_types)


def modify_layers(m, modify_fn, filter_fn):
    "Recursively modify layers with a `modify_fn` according to a `filter_fn`"
    is_sequential = isinstance(m, nn.Sequential)
    it = enumerate(m.children()) if is_sequential else m.named_children()
    for name,layer in it:
        if filter_fn(layer):
            if is_sequential: m[name] = modify_fn(layer)
            else            : setattr(m, name, modify_fn(layer))
        elif has_children(layer):
            modify_layers(layer, modify_fn, filter_fn)

    return m


def modify_layers_types(m, modify_fn, replace_types):
    "Recursively modify layers with a `modify_fn` according to types `replace_types`"
    replace_types = listify(replace_types)
    def filter_types(layer): return any(isinstance(layer,o) for o in replace_types)
    return modify_layers(m, modify_fn, filter_types)