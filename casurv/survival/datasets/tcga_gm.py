__all__ = ['get_tcga_gm_df', 'get_tcga_gm_dls']


from fastai.vision.all import *
from fastai.tabular.all import *
from ...basics import *
from ..data import *


def get_tcga_gm_df(path):
    "Survival time is given in months"
    path = Path(path)
    cols = ['TCGA ID', 'censored', 'Survival months']
    n_cols = ['id', 'censored', 'survival']
    df = pd.read_csv(path).drop(['indexes'], axis=1)
    df.rename(dict(zip(cols, n_cols)), axis=1, inplace=True)
    return df.set_index('id')


@delegates(TfmdDL.__init__)
def get_tcga_gm_dls(path, img_sz=None, batch_tfms=None, return_df=False, include_metadata=False,
                    load_ids=False, imagenet_normalize=True, images_dir='images', **kwargs):
    path = Path(path)
    metadata_df = get_tcga_gm_df(path / 'all_dataset.csv')
    df = pd.DataFrame({'fp': get_image_files(path / images_dir)})
    df['id'] = df.fp.apply(lambda x: x.name[:12])
    df = df.merge(metadata_df, 'left', left_on='id', right_index=True)
    df['is_valid'] = df.fp.apply(lambda x: x.parent.name == 'test')
    if load_ids:
        map_id = {o:i for i,o in enumerate(df.id.unique().tolist())}
        df['id_'] = df.id.map(map_id)

    def label_func(x): return (x.survival, x.censored)
    def get_id(x): return x.id_
    def get_fp(x): return x.fp
    item_tfms = Resize(img_sz) if (img_sz is not None) and (img_sz != 1024) else None
    batch_tfms = listify(batch_tfms)
    if imagenet_normalize: batch_tfms.append(Normalize.from_stats(*imagenet_stats))

    blocks = [ImageBlock]
    get_x = [get_fp]

    if include_metadata:
        cont_names = df.columns[4:-2 if load_ids else -1].tolist()
        to = TabularPandas(df, cont_names=cont_names, procs=[FillMissing, Normalize])
        df = to.items
        blocks.append(MetadataContBlock(cont_names, 'codeletion'))
        get_x.append(noop)

    if load_ids:
        blocks.append(PatientIdBlock('id_', 'pid'))
        get_x.append(get_id)

    blocks = tuple(blocks + [SurvivalBlock])

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
