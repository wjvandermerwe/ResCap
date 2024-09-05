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




