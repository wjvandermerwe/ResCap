import pandas as pd
from sklearn.compose import make_column_selector
from sklearndf.pipeline import PipelineDF
from sklearndf.transformation import OneHotEncoderDF, ColumnTransformerDF, SimpleImputerDF, StandardScalerDF

def preprocess(df: pd.DataFrame):

    enc_fac = PipelineDF(steps=[('ohe', OneHotEncoderDF(sparse=False, drop='first', handle_unknown='ignore'))])
    enc_num = PipelineDF(steps=[('imp', SimpleImputerDF(strategy='median')), ('scale', StandardScalerDF())])

    sel_fac = make_column_selector(pattern='^fac\\_')
    sel_num = make_column_selector(pattern='^num\\_')

    enc_df = ColumnTransformerDF(transformers=[('ohe', enc_fac, sel_fac), ('imp', enc_num, sel_num)], remainder='passthrough')

    enc_df = enc_df.fit(df)
    transformed_array = enc_df.transform(df)

    def get_feature_names(transformer, columns):
        if hasattr(transformer, 'get_feature_names_out'):
            return transformer.get_feature_names_out(columns)
        return columns


    feature_names = get_feature_names(enc_df, df.columns)

    transformed_df = pd.DataFrame(
        transformed_array,
        columns=feature_names,
        index=df.index
    )

    return transformed_df

