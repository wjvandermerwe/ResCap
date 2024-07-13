import os
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
import plotnine as pn
from SurvSet.data import SurvLoader
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector
from sklearndf.pipeline import PipelineDF
from sklearndf.transformation import OneHotEncoderDF, ColumnTransformerDF, SimpleImputerDF, StandardScalerDF
import pickle

from config import save_dataset


class SurvivalDataset(Dataset):
    def __init__(self, data: DataFrame, enc_df: ColumnTransformerDF, event_col: str = 'event', time_col: str = 'time'):
        self.data = data
        self.enc_df = enc_df
        self.event_col = event_col
        self.time_col = time_col

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        features = self.enc_df.transform(row.to_frame().T).astype(np.float32)
        event = row[self.event_col]
        time = row[self.time_col]
        return {
            'features': torch.tensor(features.values),
            'event': torch.tensor(event),
            'time': torch.tensor(time)
        }


def summarize_dataset(df: DataFrame, ds_name: str, sel_fac: callable, sel_num: callable, sel_ohe: callable) -> dict:
    summary = {
        "ds_name": ds_name,
        "td": False,
        "n": len(df),
        "n_fac": len(sel_fac(df)),
        "n_ohe": len(sel_ohe(df)),
        "n_num": len(sel_num(df))
    }
    return summary


def load_datasets() -> None:
    # Set up feature transformer pipeline
    loader = SurvLoader()
    ds_lst = loader.df_ds[~loader.df_ds['is_td']]['ds'].to_list()
    n_ds = len(ds_lst)

    for i, ds in enumerate(ds_lst):
        print('Dataset %s (%i of %i)' % (ds, i + 1, n_ds))
        df, ref = loader.load_dataset(ds).values()
        df_train, df_test = train_test_split(df, stratify=df['event'], random_state=1, test_size=0.3)

        train_filename = f"{ds}_train"
        test_filename = f"{ds}_test"

        save_dataset(df_train, train_filename, "datasets")
        save_dataset(df_test, test_filename, "datasets")

def get_column_transformer(df: DataFrame) -> ColumnTransformerDF:
    enc_fac = PipelineDF(steps=[('ohe', OneHotEncoderDF(sparse=False, drop=None, handle_unknown='ignore'))])
    enc_num = PipelineDF(steps=[('impute', SimpleImputerDF(strategy='median')), ('scale', StandardScalerDF())])
    sel_fac = make_column_selector(pattern='^fac\\_')
    sel_num = make_column_selector(pattern='^num\\_')
    sel_ohe = make_column_selector(pattern='^ohe\\_')

    if any(df.columns.str.startswith('fac_')) and any(df.columns.str.startswith('num_')):
        enc_df = ColumnTransformerDF(transformers=[('ohe', enc_fac, sel_fac), ('s', enc_num, sel_num)])
    elif any(df.columns.str.startswith('fac_')):
        enc_df = ColumnTransformerDF(transformers=[('ohe', enc_fac, sel_fac)])
    elif any(df.columns.str.startswith('num_')):
        enc_df = ColumnTransformerDF(transformers=[('s', enc_num, sel_num)])
    elif any(df.columns.str.startswith('ohe_')):
        enc_df = ColumnTransformerDF(transformers=[('ohe', enc_fac, sel_ohe)])
    else:
        raise ValueError("Dataset does not contain the expected 'fac_', 'num_', or 'ohe_' columns.")

    return enc_df
def get_torch_loaders(train, test, enc_df) -> (DataLoader, DataLoader):
    train_dataset = SurvivalDataset(train, enc_df)
    test_dataset = SurvivalDataset(test, enc_df)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return (train_loader, test_loader)

def show_dataset_metrics(holder_cindex: np.ndarray, ds_lst: list) -> None:
    df_cindex = pd.DataFrame(holder_cindex, columns=['cindex', 'lb', 'ub'])
    df_cindex.insert(0, 'ds', ds_lst)
    ds_ord = df_cindex.sort_values('cindex')['ds'].values
    df_cindex['ds'] = pd.Categorical(df_cindex['ds'], ds_ord)

    gg_cindex = (pn.ggplot(df_cindex, pn.aes(y='cindex', x='ds')) +
                 pn.theme_bw() + pn.coord_flip() +
                 pn.geom_point(size=2) +
                 pn.geom_linerange(pn.aes(ymin='lb', ymax='ub')) +
                 pn.labs(y='Concordance') +
                 pn.geom_hline(yintercept=0.5, linetype='--', color='red') +
                 pn.theme(axis_title_y=pn.element_blank()))
    print(gg_cindex)
