import pandas as pd
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearndf.pipeline import PipelineDF
from sklearndf.transformation import OneHotEncoderDF, ColumnTransformerDF, SimpleImputerDF, StandardScalerDF


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Define the pipelines for different types of columns
    enc_fac = Pipeline(steps=[('ohe', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'))])
    enc_num = Pipeline(steps=[('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())])

    # Selectors for categorical and numerical columns
    sel_fac = make_column_selector(pattern='^fac\\_')
    sel_num = make_column_selector(pattern='^num\\_')

    # Print selected columns for debugging
    print(f"Categorical columns selected: {sel_fac(df)}")
    print(f"Numerical columns selected: {sel_num(df)}")

    # Column transformer
    enc_df = ColumnTransformer(transformers=[
        ('ohe', enc_fac, sel_fac),
        ('imp', enc_num, sel_num)
    ], remainder='passthrough')

    # Fit the transformer and transform the data
    try:
        enc_df.fit(df)
    except Exception as e:
        print(f"Error during fit: {e}")
        raise

    # Check the number of transformers and prefixes
    transformers = enc_df.transformers_
    print(f"Number of transformers: {len(transformers)}")
    print(f"Transformers: {transformers}")

    transformed_array = enc_df.transform(df)

    # Create the DataFrame with the original feature names
    transformed_df = pd.DataFrame(transformed_array, columns=enc_df.get_feature_names_out(), index=df.index)

    # Directly update the column names by removing the part before '__'
    transformed_df.columns = [name.split('__')[-1] if '__' in name else name for name in transformed_df.columns]

    return transformed_df
