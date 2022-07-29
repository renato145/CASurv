__all__ = ['PatientIdSetup', 'PatientIdBlock', 'MetadataCatSetup', 'MetadataCatBlock', 'MetadataContSetup',
           'MetadataContBlock']


from fastai.basics import *


class PatientIdSetup(DisplayedTransform):
    def __init__(self, id_col='pid', show_label=None):
        show_label = ifnone(show_label, id_col)
        store_attr()
    def encodes(self, o:pd.Series): return tensor([o[self.id_col].astype(int)]).long()
    def decodes(self, o): return TitledStr(f'{self.show_label}: {o.item()}')


def PatientIdBlock(id_col='pid', show_label=None):
    "Transforms a `pd.Series` of numbers into tensors"
    return TransformBlock(type_tfms=PatientIdSetup(id_col, show_label))


class MetadataCatSetup(DisplayedTransform):
    def __init__(self, names, show_names=None):
        self.names = listify(names)
        self.show_names = listify(ifnone(show_names, names))
    def encodes(self, o:pd.Series): return tensor(o[self.names].astype(int)).long()
    def decodes(self, o):
        return TitledTuple(f'{n}: {o_.item()}' for n,o_ in zip(self.names,o) if n in self.show_names)


def MetadataCatBlock(names, show_names=None):
    "Transforms a `pd.Series` of categorical values into tensors"
    return TransformBlock(type_tfms=MetadataCatSetup(names, show_names))


class MetadataContSetup(DisplayedTransform):
    def __init__(self, names, show_names=None):
        self.names = listify(names)
        self.show_names = listify(ifnone(show_names, names))
    def encodes(self, o:pd.Series): return tensor(o[self.names].astype(float)).float()
    def decodes(self, o):
        return TitledTuple(f'{n}: {o_.item():.4f}' for n,o_ in zip(self.names,o) if n in self.show_names)


def MetadataContBlock(names, show_names=None):
    "Transforms a `pd.Series` of continuous values into tensors"
    return TransformBlock(type_tfms=MetadataContSetup(names, show_names))