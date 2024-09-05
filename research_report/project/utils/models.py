import os

import pandas as pd
from lifelines import CoxPHFitter
from sklearn.inspection import permutation_importance
from sksurv.ensemble import RandomSurvivalForest
import numpy as np
from SurvivalEVAL.Evaluator import  ScikitSurvivalEvaluator, LifelinesEvaluator
from matplotlib import pyplot as plt
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import joblib
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant


def get_data_splits(data, test_data):
    train_event_times = data['time'].values
    train_event_indicators = data['event'].values.astype(bool)  # Ensure boolean type
    test_event_times = test_data['time'].values
    test_event_indicators = test_data['event'].values.astype(bool)
    return train_event_times, train_event_indicators, test_event_times, test_event_indicators

def run_cox_model_varying_penalizer(data, test_data, lasso_penalizers):
    """
    Fit a Cox Proportional Hazards model with varying Lasso regularization strengths and return various outputs.

    Args:
    - data (DataFrame): The training dataset.
    - test_data (DataFrame): The test dataset.
    - lasso_penalizers (list): A list of L1 regularization strengths to test.

    Returns:
    - dict: A dictionary where each key is a penalizer value and each value is a tuple containing:
        - model (CoxPHFitter): The fitted Cox model.
        - median (Series): Predicted median survival times.
        - hazard (Series): Partial hazard predictions.
        - coefficients (DataFrame): The model coefficients.
    """
    data_pre = data.drop(columns=['pid'], errors='ignore')
    results = {}

    for penalizer in lasso_penalizers:
        # Fit the Cox Proportional Hazards model with Lasso regularization
        model = CoxPHFitter(penalizer=penalizer)
        model.fit(data_pre, duration_col='time', event_col='event')

        # Compute survival function, median survival, and partial hazards
        median = model.predict_median(data_pre)
        hazard = model.predict_partial_hazard(data_pre)
        coefficients = model.params_

        # Store the results
        results[penalizer] = (model, median, hazard, coefficients)

    return results


def run_rsf_model_varying_estimators(data, test_data, n_estimators_list, random_state=42):
    """
    Fit a Random Survival Forest model with varying numbers of trees and return various outputs.

    Args:
    - data (DataFrame): The training dataset.
    - test_data (DataFrame): The test dataset.
    - n_estimators_list (list): A list of values for `n_estimators` (number of trees).
    - random_state (int): Random state for reproducibility.

    Returns:
    - dict: A dictionary where each key is a number of trees and each value is a tuple containing:
        - rsf (RandomSurvivalForest): The fitted RSF model.
        - hazard (list of Series): Cumulative hazard predictions.
        - feature_importance (DataFrame): Feature importances.
    """
    test_pre = test_data.drop(columns=['pid', 'event', 'time'], errors='ignore')
    data_pre = data.drop(columns=['pid', 'event', 'time'], errors='ignore')

    train_event_times, train_event_indicators, test_event_times, test_event_indicators = get_data_splits(data,
                                                                                                         test_data)

    # Prepare the structured array needed for RSF model input
    y_train = np.array(list(zip(train_event_indicators, train_event_times)),
                       dtype=[('event', 'bool'), ('time', 'float')])
    y_test = np.array(list(zip(test_event_indicators, test_event_times)), dtype=[('event', 'bool'), ('time', 'float')])

    results = {}

    for n_estimators in n_estimators_list:
        # Initialize and fit the Random Survival Forest model
        rsf = RandomSurvivalForest(
            n_estimators=n_estimators, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=random_state,

        )
        rsf.fit(X=data_pre, y=y_train)

        # Predict cumulative hazard function
        hazard = rsf.predict_cumulative_hazard_function(data_pre)

        # Compute permutation importance to assess feature importance
        result = permutation_importance(rsf, test_pre, y_test, n_repeats=15, random_state=random_state)
        feature_importance = pd.DataFrame(
            {
                k: result[k]
                for k in ("importances_mean", "importances_std")
            },
            index=test_pre.columns,
        ).sort_values(by="importances_mean", ascending=False)

        # Store the results
        results[n_estimators] = (rsf, hazard, feature_importance)

    return results


def plot_rsf_varying_estimators(results):
    """
    Plot the results from varying the number of trees in the Random Survival Forest model.

    Args:
    - results (dict): A dictionary where each key is a number of trees and each value is a tuple containing:
                      - rsf (RandomSurvivalForest): The fitted RSF model.
                      - hazard (list of Series): Cumulative hazard predictions.
                      - feature_importance (DataFrame): Feature importances.

    Returns:
    - None: Displays the plots for performance metrics and feature importances.
    """
    n_estimators_list = list(results.keys())

    # Placeholder for collecting metrics and feature importances
    mean_importances = pd.DataFrame(index=results[n_estimators_list[0]][2].index)
    std_importances = pd.DataFrame(index=results[n_estimators_list[0]][2].index)

    for n_estimators, (_, _, feature_importance) in results.items():
        mean_importances[n_estimators] = feature_importance['importances_mean']
        std_importances[n_estimators] = feature_importance['importances_std']

    # Plot feature importances for the top features
    plot_top_feature_importances(mean_importances, std_importances)


def plot_top_feature_importances(mean_importances, std_importances, n_highlight=10):
    """
    Plot the top feature importances across different numbers of trees.

    Args:
    - mean_importances (DataFrame): DataFrame with mean importances across different numbers of trees.
    - std_importances (DataFrame): DataFrame with standard deviations of importances.
    - n_highlight (int): Number of top features to highlight in the plot.

    Returns:
    - None: Displays the plot.
    """
    # Identify the top features based on the maximum number of trees
    top_features = mean_importances.iloc[:, -1].sort_values(ascending=False).head(n_highlight).index

    plt.figure(figsize=(10, 6))

    # Plot each top feature's importance across the different numbers of trees
    for feature in top_features:
        plt.errorbar(mean_importances.columns, mean_importances.loc[feature], yerr=std_importances.loc[feature],
                     label=feature, marker='o', linestyle='-')

    plt.xscale('log')
    plt.xlabel("Number of Trees (Log Scale)")
    plt.ylabel("Feature Importance")
    plt.title("Top Feature Importances vs. Number of Trees")
    plt.legend(loc='best')
    plt.grid(True, which="both", ls="--")
    plt.show()

def run_cox_model(data, test_data, lasso_penalizer=0.1):
    data_pre = data.drop(columns=['pid'], errors='ignore')
    # Extract training and test event times and indicators

    # Fit the Cox Proportional Hazards model with Lasso regularization
    model = CoxPHFitter(penalizer=lasso_penalizer)
    model.fit(data_pre, duration_col='time', event_col='event')

    # Compute survival function, median survival, and partial hazards
    cox_survival_function = model.predict_survival_function(data_pre)
    median = model.predict_median(data_pre)
    hazard = model.predict_partial_hazard(data_pre)
    return model, cox_survival_function, median, hazard


def preprocess_and_fit_cox_model(df: pd.DataFrame, df_test: pd.DataFrame, duration_col: str, event_col: str, penalizer: float = 0.1, step_size: float = 0.5):
    # 1. Check and Drop Low Variance Columns
    low_variance_threshold = 1e-4
    low_variance_columns = [col for col in df.columns if col.startswith('fac_') and df[col].var() < low_variance_threshold]
    if low_variance_columns:
        print(f"Dropping low variance columns: {low_variance_columns}")
        df = df.drop(columns=low_variance_columns)
        df_test = df_test.drop(columns=low_variance_columns)

    # 2. Check for Complete Separation
    events = df[event_col].astype(bool)
    separation_columns = []
    for col in df.columns:
        if col.startswith('fac_'):
            var_event = df.loc[events, col].var()
            var_no_event = df.loc[~events, col].var()
            if var_event < low_variance_threshold or var_no_event < low_variance_threshold:
                separation_columns.append(col)
                print(f"Column {col} may cause complete separation. Variance when event is present: {var_event}, when not present: {var_no_event}")

    if separation_columns:
        print(f"Dropping columns with potential complete separation: {separation_columns}")
        df = df.drop(columns=separation_columns)
        df_test = df_test.drop(columns=separation_columns)


    # 3. Check for High Collinearity using VIF
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_columns:
        X = add_constant(df[numerical_columns])
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        # Drop high VIF columns, but exclude 'const' and 'event'
        high_vif_columns = [feature for feature in vif_data[vif_data["VIF"] > 10]["feature"].tolist() if feature not in ['const', event_col]]
        if high_vif_columns:
            print(f"Dropping high VIF columns: {high_vif_columns}")
            df = df.drop(columns=high_vif_columns)
            df_test = df_test.drop(columns=high_vif_columns)


    # 4. Fit the Cox Proportional Hazards Model with Penalizer and Step Size
    model = CoxPHFitter(penalizer=penalizer)
    try:
        model.fit(df, duration_col=duration_col, event_col=event_col)
        cox_survival_function = model.predict_survival_function(df_test)
        median = model.predict_median(df_test)
        hazard = model.predict_partial_hazard(df_test)
        return model, cox_survival_function, median, hazard, df, df_test
        print("Model fit successfully.")
    except Exception as e:
        print(f"Model fitting failed: {e}")


def run_rsf_model(data, test_data, random_state=42):
    # Drop 'pid' column from test data if present
    test_pre = test_data.drop(columns=['pid', 'event', 'time'], errors='ignore')
    data_pre = data.drop(columns=['pid', 'event', 'time'], errors='ignore')

    # Prepare training and testing event times and indicators
    train_event_times, train_event_indicators, test_event_times, test_event_indicators = get_data_splits(data,
                                                                                                         test_data)

    # Ensure consistency in data
    assert len(test_pre) == len(test_event_times) == len(test_event_indicators), \
        "Mismatch between test data and event times/indicators!"

    # Prepare the structured array needed for RSF model input
    y_train = np.array(list(zip(train_event_indicators, train_event_times)),
                       dtype=[('event', 'bool'), ('time', 'float')])
    y_test = np.array(list(zip(test_event_indicators, test_event_times)),
                      dtype=[('event', 'bool'), ('time', 'float')])

    # Initialize and fit the Random Survival Forest model
    rsf = RandomSurvivalForest(
        n_estimators=100, min_samples_split=20, min_samples_leaf=20, n_jobs=-1, random_state=random_state, verbose=True
    )
    rsf.fit(X=data_pre, y=y_train)

    # Predict survival function and cumulative hazard function
    rsf_survival_function = rsf.predict_survival_function(test_pre, return_array=True)
    clean_rsf_survival_function = rsf.predict_survival_function(test_pre)
    hazard = rsf.predict_cumulative_hazard_function(test_pre, return_array=True)

    # Compute permutation importance to assess feature importance
    result = permutation_importance(rsf, test_pre, y_test, n_repeats=15, random_state=random_state)
    feature_importance = pd.DataFrame(
        {
            k: result[k]
            for k in ("importances_mean", "importances_std")
        },
        index=test_pre.columns,
    ).sort_values(by="importances_mean", ascending=False)

    return rsf, rsf_survival_function, hazard, feature_importance, clean_rsf_survival_function


from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV
from sksurv.metrics import concordance_index_censored
import numpy as np
import pandas as pd

def run_rsf_model_param_tuning(data, test_data, random_state=42):
    # Drop 'pid' column from test data if present
    test_pre = test_data.drop(columns=['pid','event', 'time'], errors='ignore')
    data_pre = data.drop(columns=['pid', 'event', 'time'], errors='ignore')

    # Prepare training and testing event times and indicators
    train_event_times, train_event_indicators, test_event_times, test_event_indicators = get_data_splits(data, test_data)

    # Prepare the structured array needed for RSF model input
    y_train = np.array(list(zip(train_event_indicators, train_event_times)),
                       dtype=[('event', 'bool'), ('time', 'float')])
    y_test = np.array(list(zip(test_event_indicators, test_event_times)), dtype=[('event', 'bool'), ('time', 'float')])

    # Set up the Random Survival Forest model
    rsf = RandomSurvivalForest(random_state=random_state)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [5, 10, 15],
        'max_features': ['sqrt', 'log2', 0.5]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rsf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X=data_pre, y=y_train)

    # Extract the best model and fit on the full training data
    best_rsf = grid_search.best_estimator_
    best_rsf.fit(X=data_pre, y=y_train)

    # Predict survival function and cumulative hazard function
    rsf_survival_function = best_rsf.predict_survival_function(test_pre, return_array=True)
    hazard = best_rsf.predict_cumulative_hazard_function(test_pre, return_array=True)
    clean_rsf_survival_function = best_rsf.predict_survival_function(test_pre)

    # Compute permutation importance to assess feature importance
    result = permutation_importance(best_rsf, test_pre, y_test, n_repeats=15, random_state=random_state)
    feature_importance = pd.DataFrame(
        {
            k: result[k]
            for k in ("importances_mean", "importances_std")
        },
        index=test_pre.columns,
    ).sort_values(by="importances_mean", ascending=False)

    return best_rsf, rsf_survival_function, hazard, feature_importance, clean_rsf_survival_function



def save_metrics(metrics, output_dir, dataset_name):
    """
    Save the metrics to a CSV file.

    :param metrics: Dictionary containing the calculated metrics.
    :param output_dir: Directory to save the output file.
    :param dataset_name: Name of the dataset (used in file name).
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert metrics dictionary to DataFrame for easy saving
    metrics_df = pd.DataFrame([metrics])

    # Save the metrics to a CSV file
    metrics_file = f"{output_dir}/{dataset_name}_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)

    print(f"Metrics saved to {metrics_file}")


def run_metrics(cox_survival_function, rsf_survival_function, train_event_times, train_event_indicators, test_event_times, test_event_indicators):

    cox_model_evals = LifelinesEvaluator(cox_survival_function,test_event_times, test_event_indicators, train_event_times, train_event_indicators)
    rsf_model_evals = ScikitSurvivalEvaluator(rsf_survival_function,test_event_times, test_event_indicators, train_event_times, train_event_indicators)

    metrics = {
        "concordance_cox": cox_model_evals.concordance()[0],
        "concordance_rsf": rsf_model_evals.concordance()[0],
        "brier_cox": cox_model_evals.brier_score(),
        "brier_rsf": rsf_model_evals.brier_score(),
        "ibs_cox": cox_model_evals.integrated_brier_score(),
        "ibs_rsf": rsf_model_evals.integrated_brier_score(),
        "cox_one_cal": cox_model_evals.one_calibration(np.median(test_event_times)),
        "rsf_one_cal": rsf_model_evals.one_calibration(np.median(test_event_times)),
        "cox_d_cal": cox_model_evals.d_calibration(),
        "rsf_d_cal": rsf_model_evals.d_calibration(),
        "mae_cox": cox_model_evals.mae(),
        "mae_rsf": rsf_model_evals.mae(),
        "rmse_cox": cox_model_evals.rmse(),
        "rmse_rsf": rsf_model_evals.rmse(),
    }

    return metrics


def plot_calibration_curve(cox_1, cox_2, rsf_1, rsf_2):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot calibration curves
    ax.plot(cox_1, cox_2, marker='o', linestyle='-', color='blue', label='Cox Model')
    ax.plot(rsf_1, rsf_2, marker='x', linestyle='-', color='red', label='RSF Model')

    # Add diagonal line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

    # Add labels and legend
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Probability')
    ax.set_title('Calibration Curves')
    ax.legend()

    # Annotate one-calibration scores if needed
    # Example (you can adjust the coordinates and text as necessary)
    ax.annotate('Cox 1-Cal: 0.0', xy=(0.5, 0.5), xytext=(0.55, 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, color='blue')
    ax.annotate('RSF 1-Cal: 0.0', xy=(0.5, 0.3), xytext=(0.55, 0.4),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, color='red')

    # Show plot
    plt.show()

def plot_d_calibration(cox_counts, rsf_counts, num_bins=10):
    bin_centers = np.arange(1, num_bins + 1)  # Bin centers (1-based index)

    plt.figure(figsize=(14, 6))

    # Plot counts for Cox model
    plt.subplot(1, 2, 1)
    plt.bar(bin_centers - 0.2, cox_counts, width=0.4, color='blue', label='Cox Model')
    plt.xlabel('Bin')
    plt.ylabel('Counts')
    plt.title('Cox Model D-Calibration Counts')
    plt.xticks(bin_centers)
    plt.legend()

    # Plot counts for RSF model
    plt.subplot(1, 2, 2)
    plt.bar(bin_centers + 0.2, rsf_counts, width=0.4, color='red', label='RSF Model')
    plt.xlabel('Bin')
    plt.ylabel('Counts')
    plt.title('RSF Model D-Calibration Counts')
    plt.xticks(bin_centers)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_rsf_feature_importances(feature_importance):
    """
    Plot the feature importances from a Random Survival Forest model.

    Args:
    - feature_importance (pd.DataFrame): A DataFrame containing feature importances and their standard deviations.

    Returns:
    - None: Displays a plot of the feature importances.
    """
    # Ensure the DataFrame contains 'importances_mean' and 'importances_std'
    importances = feature_importance['importances_mean']
    std_errors = feature_importance['importances_std']

    fig, ax = plt.subplots(figsize=(8, len(importances) / 2))
    importances.plot(kind='barh', ax=ax, xerr=std_errors, color='green', ecolor='black', capsize=4)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importances from Random Survival Forest Model')
    plt.show()

def plot_cox_box_coefficients(cox_model):
    coefs = cox_model.params_
    std_errors = cox_model.standard_errors_

    fig, ax = plt.subplots(figsize=(8, len(coefs) / 2))
    coefs.plot(kind='barh', ax=ax, xerr=std_errors, color='blue', ecolor='black', capsize=4)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Coefficients from Cox Proportional Hazards Model')
    plt.show()


def log_plot_cox(coefs, n_highlight=10):
    """
    Plot the coefficients from the Cox model with Lasso regularization on a logarithmic scale.

    Args:
    - coefs (pd.DataFrame): A DataFrame where rows are features and columns are Lasso penalizer values.
    - n_highlight (int): Number of top features to highlight in the plot.

    Returns:
    - None: Displays a plot of the coefficients on a logarithmic scale.
    """
    _, ax = plt.subplots(figsize=(9, 6))
    alphas = coefs.columns

    # Plot the coefficients for each feature across alphas
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    # Highlight the top n_highlight coefficients at the minimum alpha
    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("Lasso Penalizer (Log Scale)")
    ax.set_ylabel("Coefficient")
    ax.set_title("Cox Model Coefficients with Lasso Regularization")
    plt.show()

def log_plot_rsf(feature_importances, n_highlight=10):
    """
    Plot the feature importances from the RSF model on a logarithmic scale.

    Args:
    - feature_importances (pd.DataFrame): A DataFrame where rows are features and columns are n_estimators values.
    - n_highlight (int): Number of top features to highlight in the plot.

    Returns:
    - None: Displays a plot of the feature importances on a logarithmic scale.
    """
    _, ax = plt.subplots(figsize=(9, 6))
    n_estimators_list = feature_importances.columns

    # Plot the feature importances for each feature across different numbers of trees
    for row in feature_importances.itertuples():
        ax.semilogx(n_estimators_list, row[1:], ".-", label=row.Index)

    # Highlight the top n_highlight feature importances at the maximum number of trees
    n_estimators_max = n_estimators_list.max()
    top_importances = feature_importances.loc[:, n_estimators_max].sort_values(ascending=False).head(n_highlight)
    for name in top_importances.index:
        importance = feature_importances.loc[name, n_estimators_max]
        plt.text(n_estimators_max, importance, name + "   ", horizontalalignment="left", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("Number of Trees (Log Scale)")
    ax.set_ylabel("Feature Importance")
    ax.set_title("RSF Model Feature Importances Across Different Trees")
    plt.show()

def plot_importance_log_line(importances, n_highlight=10, title='Feature Importance (Log Scale)'):
    """
    Plot the feature importance scores on a logarithmic scale as a line plot.

    Args:
    - importances (pd.DataFrame): A DataFrame where rows are features and the column is the mean importance score.
    - n_highlight (int): Number of top features to highlight in the plot.
    - title (str): Title of the plot.

    Returns:
    - None: Displays a line plot of the importance scores on a logarithmic scale.
    """
    # Sort by importance and select the top n_highlight features
    sorted_importances = importances.sort_values(by='importances_mean', ascending=False).head(n_highlight)

    # Plotting the importance scores as a line plot with a log scale on the y-axis
    plt.figure(figsize=(10, 6))

    # Iterate over each feature and plot its importance score
    for feature in sorted_importances.index:
        plt.plot(sorted_importances.columns, sorted_importances.loc[feature], label=feature, marker='o', linestyle='-')

    plt.yscale('log')
    plt.xlabel("Features")
    plt.ylabel("Feature Importance (Log Scale)")
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend(loc='best')
    plt.show()

def plot_importance_log_scale(importances, n_highlight=10, title='Feature Importance (Log Scale)'):
    """
    Plot the feature importance scores on a logarithmic scale.

    Args:
    - importances (pd.DataFrame or pd.Series): A DataFrame or Series where rows are features and columns are
                                               importance scores (e.g., mean and std for RSF).
    - n_highlight (int): Number of top features to highlight in the plot.
    - title (str): Title of the plot.

    Returns:
    - None: Displays a plot of the importance scores on a logarithmic scale.
    """
    # Ensure importances is a DataFrame
    if isinstance(importances, pd.Series):
        importances = importances.to_frame(name='importance')

    # Sort by importance
    sorted_importances = importances.sort_values(by='importances_mean', ascending=False).head(n_highlight)

    # Plotting the importance scores with a log scale on the y-axis
    plt.figure(figsize=(10, 6))
    sorted_importances['importances_mean'].plot(kind='barh',
                                                xerr=sorted_importances['importances_std'],
                                                color='blue', ecolor='black', capsize=4)

    plt.xscale('log')
    plt.xlabel("Feature Importance (Log Scale)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_yaxis()  # To display the highest importance at the top
    plt.show()

def plot_lasso_effects(df, feature_columns, duration_col='time', event_col='event', alphas=None):
    """
    Plot the effect of Lasso regularization on the coefficients of a Cox Proportional Hazards model.

    Args:
    - df (pd.DataFrame): The dataset containing the features, duration, and event columns.
    - feature_columns (list): List of feature column names to include in the model.
    - duration_col (str): The name of the column that contains the duration data.
    - event_col (str): The name of the column that contains the event data.
    - alphas (array-like): A list or array of alpha (regularization strength) values to use.

    Returns:
    - None: Displays a plot showing the effect of Lasso regularization on the coefficients.
    """
    if alphas is None:
        alphas = np.logspace(-4, 0, 50)  # Default alpha values

    coefficients = pd.DataFrame(index=feature_columns)

    # Fit the model for each alpha and store the coefficients
    for alpha in alphas:
        cph = CoxPHFitter(penalizer=alpha)
        cph.fit(df[feature_columns + [duration_col, event_col]], duration_col=duration_col, event_col=event_col)
        coefficients[alpha] = cph.params_

    # Transpose for plotting (alphas as columns)
    coefficients = coefficients.T

    # Plot the coefficients to see the effect of Lasso regularization
    plt.figure(figsize=(10, 6))
    for column in coefficients.columns:
        plt.plot(coefficients.index, coefficients[column], label=column)

    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Coefficient Value')
    plt.title('Effect of Lasso Regularization on Coefficients')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def cross_validate_and_plot(models, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {name: [] for name in models.keys()}

    # Concatenate X and y to ensure that they are split together correctly
    data = pd.concat([X, y], axis=1)

    for train_index, val_index in kf.split(data):
        # Split the data into training and validation sets
        train_data, val_data = data.iloc[train_index], data.iloc[val_index]

        # Separate X and y for train and validation sets
        X_train, y_train = train_data.drop(columns=['time', 'event']), train_data[['time', 'event']]
        X_val, y_val = val_data.drop(columns=['time', 'event']), val_data[['time', 'event']]

        for name, model_func in models.items():
            if name == 'CoxPH':
                # Prepare data for the CoxPH model
                train_data_cox = pd.concat([X_train, y_train], axis=1)
                val_data_cox = pd.concat([X_val, y_val], axis=1)

                # Run the Cox model
                cox_model, _, _, hazard, _, _ = model_func(train_data_cox, val_data_cox)

                # Ensure hazard predictions have the correct index
                preds = pd.Series(hazard.values.flatten(), index=X_val.index)

            elif name == 'RSF':
                # Prepare y_train and y_val as structured arrays for RSF model
                y_train_structured = np.array(list(zip(y_train['event'], y_train['time'])),
                                              dtype=[('event', 'bool'), ('time', 'float')])
                y_val_structured = np.array(list(zip(y_val['event'], y_val['time'])),
                                            dtype=[('event', 'bool'), ('time', 'float')])

                # Run the RSF model
                rsf_model, rsf_survival_function, _, _ = model_func(pd.concat([X_train, y_train], axis=1), pd.concat([X_val, y_val], axis=1))

                # Predict cumulative hazard for the validation set
                rsf_hazard = rsf_model.predict_cumulative_hazard_function(X_val, return_array=True)
                preds = -pd.Series(rsf_hazard.mean(axis=1), index=X_val.index)  # Negative for concordance_index

            # Calculate the concordance index
            c_index = concordance_index(y_val['time'], preds, y_val['event'])
            results[name].append(c_index)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df['Fold'] = np.arange(1, n_splits + 1)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for model_name in models.keys():
        plt.errorbar(results_df['Fold'], results_df[model_name], yerr=np.std(results_df[model_name]), label=model_name, fmt='-o')

    plt.title('Cross-Validated C-index for Survival Models')
    plt.xlabel('Fold')
    plt.ylabel('C-index')
    plt.legend()
    plt.grid(True)
    plt.show()
