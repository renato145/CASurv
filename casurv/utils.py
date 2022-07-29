__all__ = []


from fastai.basics import *


@patch
def remove_unique_cols(df: pd.DataFrame, inplace=False):
    '''Remove columns with unique values on a DataFrame, useful to check training results and
    ignore not changing parameters'''
    if not inplace: df = df.copy()
    df.drop([k for k,v in df.nunique().items() if v==1], axis=1, inplace=True)
    return df