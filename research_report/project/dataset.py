import os
import numpy as np
import pandas as pd
from SurvSet.data import SurvLoader
from sksurv.util import Surv
from sklearn.compose import make_column_selector
from sklearndf.pipeline import PipelineDF
from sklearndf.transformation import OneHotEncoderDF, ColumnTransformerDF, SimpleImputerDF, StandardScalerDF

# (i) Set up feature transformer pipeline
enc_fac = PipelineDF(steps=[('ohe', OneHotEncoderDF(sparse=False, drop=None, handle_unknown='ignore'))])
sel_fac = make_column_selector(pattern='^fac\\_')
enc_num = PipelineDF(steps=[('impute', SimpleImputerDF(strategy='median')), ('scale', StandardScalerDF())])
sel_num = make_column_selector(pattern='^num\\_')
# Combine both
enc_df = ColumnTransformerDF(transformers=[('ohe', enc_fac, sel_fac),('s', enc_num, sel_num)])
alpha = 0.1
senc = Surv()
loader = SurvLoader()
ds_lst = loader.df_ds[~loader.df_ds['is_td']]['ds'].to_list()
print(ds_lst)