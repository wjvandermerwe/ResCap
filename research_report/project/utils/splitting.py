import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df: pd.DataFrame, is_td: bool) -> tuple:
    """
    Perform a stratified train-test split on the dataframe.

    Args:
    - df (pd.DataFrame): The dataframe to split.
    - is_td (bool): Indicator for time-dependent features.

    Returns:
    - tuple: DataFrames for training and testing sets.
    """
    # Try with a lower number of quantiles if the original stratification fails
    try_quantiles = [10, 5, 3]

    for q in try_quantiles:
        # Create the stratify label temporarily
        if is_td:
            df['stratify_label'] = (df['event'].astype(str) + '_' +
                                    np.where(df['time2'].isna(), '0', '1').astype(str) + '_' +
                                    pd.qcut(df['time'], q=q, labels=False, duplicates='drop').astype(str))
        else:
            df['stratify_label'] = df['event'].astype(str) + '_' + pd.qcut(df['time'], q=q, labels=False, duplicates='drop').astype(str)

        # Stratified split
        try:
            df_train, df_test = train_test_split(df, stratify=df['stratify_label'], random_state=1, test_size=0.3)

            # Drop the temporary stratify label
            df_train = df_train.drop(['stratify_label'], axis=1)
            df_test = df_test.drop(['stratify_label'], axis=1)

            return df_train, df_test
        except ValueError as e:
            print(f"ValueError with {q} quantiles: {e}")
            continue

    # If all attempts fail, perform a simple train-test split without stratification
    print("All stratified splits failed, performing a random split.")
    df_train, df_test = train_test_split(df, random_state=1, test_size=0.3)

    return df_train, df_test