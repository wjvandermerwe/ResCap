import numpy as np
import pandas as pd
from pandas import DataFrame
from SurvSet.data import SurvLoader
from sklearn.model_selection import train_test_split
from utils.config import save_dataset


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


def extract_datasets() -> None:
    loader = SurvLoader()
    ds_lst = loader.df_ds['ds'].to_list()
    is_td_lst = loader.df_ds['is_td'].to_list()
    n_ds = len(ds_lst)

    for i, ds in enumerate(ds_lst):
        print(f'Dataset {ds} ({i + 1} of {n_ds})')

        df, ref = loader.load_dataset(ds).values()

        # Handle time binning
        try:
            df['time_bin'] = pd.qcut(df['time'], q=10, labels=False, duplicates='drop')
        except ValueError as e:
            print(f"Warning: {e}. Adjusting binning.")
            df['time_bin'] = pd.qcut(df['time'], q=5, labels=False, duplicates='drop')

        # Check for time-varying features
        if is_td_lst[i]:
            df['censor'] = np.where(df['time2'].isna(), 0, 1)
            df['stratify_label'] = (df['event'].astype(str) + '_' +
                                    df['censor'].astype(str) + '_' +
                                    df['time_bin'].astype(str))
        else:
            df['stratify_label'] = df['event'].astype(str) + '_' + df['time_bin'].astype(str)

        # Check stratification label distribution
        label_counts = df['stratify_label'].value_counts()

        # Remove sparse labels if necessary
        if any(label_counts < 2):
            print('Warning: Some stratification labels have fewer than 2 instances.')
            df = df[df['stratify_label'].isin(label_counts[label_counts >= 2].index)]
            # Recalculate stratification labels
            if is_td_lst[i]:
                df['stratify_label'] = (df['event'].astype(str) + '_' +
                                        df['censor'].astype(str) + '_' +
                                        df['time_bin'].astype(str))
            else:
                df['stratify_label'] = df['event'].astype(str) + '_' + df['time_bin'].astype(str)

        # Ensure there are enough samples for stratified split
        n_classes = len(df['stratify_label'].unique())
        total_samples = len(df)

        if total_samples < n_classes:
            print(f"Warning: Not enough samples for stratified split in dataset {ds}. Using random split.")
            df_train, df_test = train_test_split(df, random_state=1, test_size=0.3)
        else:
            # Adjust test size based on number of classes
            min_test_size = max(n_classes, 2)  # Ensure at least 2 samples per class in the test set
            test_size = min(0.3, (total_samples - min_test_size) / (2 * min_test_size))
            test_size = max(test_size, 0.3)  # Ensure test size is not too small

            try:
                df_train, df_test = train_test_split(df, stratify=df['stratify_label'], random_state=1,
                                                     test_size=test_size)
            except ValueError as e:
                print(f"Error in stratified split: {e}. Using random split.")
                df_train, df_test = train_test_split(df, random_state=1, test_size=0.3)  # Fallback

        # Drop the splitting columns
        if is_td_lst[i]:
            df_train = df_train.drop(['time_bin', 'stratify_label', 'censor'], axis=1)
            df_test = df_test.drop(['time_bin', 'stratify_label', 'censor'], axis=1)
        else:
            df_train = df_train.drop(['time_bin', 'stratify_label'], axis=1)
            df_test = df_test.drop(['time_bin', 'stratify_label'], axis=1)

        save_dataset(df_train, ds, "../outputs/datasets")
        save_dataset(df_test, ds, "../outputs/test_sets")
