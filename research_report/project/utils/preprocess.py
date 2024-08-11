import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearndf.pipeline import PipelineDF
from sklearndf.transformation import OneHotEncoderDF, ColumnTransformerDF, SimpleImputerDF, StandardScalerDF

def impute_missing_values(ds_train):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    exclude_columns = ['event', 'time']
    numerical_features = [col for col in ds_train.select_dtypes(include=[np.number]).columns if col not in exclude_columns]
    imp.fit(ds_train[numerical_features])
    ds_train[numerical_features] = imp.transform(ds_train[numerical_features])
    return ds_train

# def get_column_transformer(df: DataFrame) -> ColumnTransformerDF:
#     enc_fac = PipelineDF(steps=[('ohe', OneHotEncoderDF(sparse=False, drop=None, handle_unknown='ignore'))])
#     enc_num = PipelineDF(steps=[('impute', SimpleImputerDF(strategy='median')), ('scale', StandardScalerDF())])
#     sel_fac = make_column_selector(pattern='^fac\\_')
#     sel_num = make_column_selector(pattern='^num\\_')
#     sel_ohe = make_column_selector(pattern='^ohe\\_')
#
#     # if any(df.columns.str.startswith('fac_')) and any(df.columns.str.startswith('num_')):
#         # enc_df = ColumnTransformerDF(transformers=[('fac', enc_fac, sel_fac), ('s', enc_num, sel_num)])
#     if any(df.columns.str.startswith('fac_')):
#         enc_df = ColumnTransformerDF(transformers=[('ohe', enc_fac, sel_fac)], remainder='passthrough')
#     elif any(df.columns.str.startswith('num_')):
#         enc_df = ColumnTransformerDF(transformers=[('s', enc_num, sel_num)],  remainder='passthrough')
#     elif any(df.columns.str.startswith('ohe_')):
#         enc_df = ColumnTransformerDF(transformers=[('ohe', enc_fac, sel_ohe)], remainder='passthrough')
#     else:
#         raise ValueError("Dataset does not contain the expected 'fac_', 'num_', or 'ohe_' columns.")
#
#     return enc_df


def get_column_transformer(df: pd.DataFrame) -> ColumnTransformer:
    # Define pipelines for encoding and scaling
    enc_fac = Pipeline(steps=[('ohe', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'))])
    enc_num = Pipeline(steps=[('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])

    # Define column selectors based on patterns
    sel_fac = make_column_selector(pattern='^fac\\_')
    sel_num = make_column_selector(pattern='^num\\_')
    sel_ohe = make_column_selector(pattern='^ohe\\_')

    # Check for presence of different types of columns and create ColumnTransformer
    if any(df.columns.str.startswith('fac_')):
        enc_df = ColumnTransformer(
            transformers=[('ohe', enc_fac, sel_fac)],
            remainder='passthrough'
        )
    elif any(df.columns.str.startswith('num_')):
        enc_df = ColumnTransformer(
            transformers=[('s', enc_num, sel_num)],
            remainder='passthrough'
        )
    elif any(df.columns.str.startswith('ohe_')):
        enc_df = ColumnTransformer(
            transformers=[('ohe', enc_fac, sel_ohe)],
            remainder='passthrough'
        )
    else:
        raise ValueError("Dataset does not contain the expected 'fac_', 'num_', or 'ohe_' columns.")

    original_columns = df.columns

    # Define a function to get the feature names after transformation
    def get_feature_names(transformer, columns):
        if hasattr(transformer, 'get_feature_names_out'):
            return transformer.get_feature_names_out(columns)
        return columns

    # Apply transformations and get new feature names
    enc_df = enc_df.fit(df)

    transformed_df = pd.DataFrame(
        enc_df.transform(df),
        columns=get_feature_names(enc_df, original_columns),
        index=df.index
    )

    return transformed_df