__all__ = ['get_nlst_df', 'get_nlst_regression_dls', 'get_nlst_dls']


from fastai.vision.all import *
from fastai.tabular.all import *
from ...basics import *
from ..data import *


def get_nlst_df(path, csv_name):
    return pd.read_csv(path / csv_name)


@delegates(TfmdDL.__init__)
def get_nlst_regression_dls(path, csv_name, img_sz=224, batch_tfms=None, return_df=False,
                 imagenet_normalize=True, **kwargs):
    df = get_nlst_df(path, csv_name)

    df = df.query('data_split != "test"').reset_index(drop=True)
    df['is_valid'] = df.data_split == 'valid'


    def label_func(x): return x.surv
    def get_fp(x): return path / x.fname
    item_tfms = Resize(img_sz)
    batch_tfms = listify(batch_tfms)
    if imagenet_normalize: batch_tfms.append(Normalize.from_stats(*imagenet_stats))

    blocks = [ImageBlock, RegressionBlock]
    get_x = [get_fp]

    db = DataBlock(
        blocks=blocks,
        get_x=get_x,
        get_y=label_func,
        splitter=ColSplitter('is_valid'),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )

    dls = db.dataloaders(df, **kwargs)
    return (dls, df) if return_df else  dls


@delegates(TfmdDL.__init__)
def get_nlst_dls(path=None, csv_name='data_splits.csv', img_sz=224, batch_tfms=None, return_df=False,
                 include_metadata=False, load_ids=False, is_hazard=False, imagenet_normalize=True,
                 metadata_cont=True, metadata_cat=True, **kwargs):
    path = Path(path)
    df = get_nlst_df(path, csv_name)

    df = df.query('data_split != "test"').reset_index(drop=True)
    df['is_valid'] = df.data_split == 'valid'

    def label_func(x): return (x.surv, x.censored)
    def get_id(x): return x.patient_id
    def get_fp(x): return path / x.fname
    item_tfms = Resize(img_sz)
    batch_tfms = listify(batch_tfms)
    if imagenet_normalize: batch_tfms.append(Normalize.from_stats(*imagenet_stats))

    blocks = [ImageBlock]
    get_x = [get_fp]

    if include_metadata:
        cont_names = ['age']
        cat_names = ['gender']
        to = TabularPandas(df, cat_names=cat_names, cont_names=cont_names,
                           procs=[Categorify, FillMissing, Normalize])
        df = to.items
        if metadata_cat:
            blocks.append(MetadataCatBlock(cat_names))
            get_x.append(noop)
        if metadata_cont:
            blocks.append(MetadataContBlock(cont_names))
            get_x.append(noop)

    if load_ids:
        blocks.append(PatientIdBlock('patient_id', 'pid'))
        get_x.append(get_id)

    blocks = tuple(blocks + [SurvivalBlock(is_hazard=is_hazard)])

    db = DataBlock(
        blocks=blocks,
        get_x=get_x,
        get_y=label_func,
        splitter=ColSplitter('is_valid'),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )

    dls = db.dataloaders(df, **kwargs)
    return (dls, df) if return_df else  dls
