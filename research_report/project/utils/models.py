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

def run_cox_model(data, test_data):
    # Extract training and test event times and indicators
    train_event_times = data['time'].values
    train_event_indicators = data['event'].values
    test_event_times = test_data['time'].values
    test_event_indicators = test_data['event'].values

    # Fit the Cox Proportional Hazards model
    model = CoxPHFitter()
    model.fit(data, duration_col='time', event_col='event')

    # Print model summary for review
    # summary = model.print_summary()

    # Compute survival function, median survival, and partial hazards
    cox_survival_function = model.predict_survival_function(data)
    median = model.predict_median(data)
    hazard = model.predict_partial_hazard(data)
    residuals_info = [
        {
            'model_name': 'Cox Model',
            'residual_data': model.compute_residuals(data, 'schoenfeld'),
            'residual_type': 'Schoenfeld'
        },
        {
            'model_name': 'Cox Model',
            'residual_data': model.compute_residuals(data, 'martingale'),
            'residual_type': 'Martingale'
        },
    ]
    # Check model assumptions - proportional hazards
    assumptions = model.check_assumptions(data)
    return model, (cox_survival_function,train_event_times,train_event_indicators,test_event_times,test_event_indicators), median, hazard, assumptions, residuals_info


def save_rsf_model_output(rsf_survival_function, hazard, feature_importance, output_dir, dataset_name):
    """
    Save the RSF model outputs to the specified directory.

    :param rsf_survival_function: List of survival functions
    :param hazard: List of cumulative hazard functions
    :param feature_importance: DataFrame of feature importance
    :param output_dir: Directory to save the output files
    :param dataset_name: Name of the dataset (used in file names)
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the survival function
    survival_file = f"{output_dir}/{dataset_name}_rsf_survival_function.pkl"
    joblib.dump(rsf_survival_function, survival_file)

    # Save the hazard function
    hazard_file = f"{output_dir}/{dataset_name}_rsf_hazard_function.pkl"
    joblib.dump(hazard, hazard_file)

    # Save the feature importance as a CSV file
    feature_importance_file = f"{output_dir}/{dataset_name}_feature_importance.csv"
    feature_importance.to_csv(feature_importance_file, index=True)

    print(f"Outputs saved to {output_dir}")


def run_rsf_model(data, test_data, random_state=42):
    # Drop 'pid' column from test data if present
    test_pre = test_data.drop(columns=['pid'], errors='ignore')
    data_pre = data.drop(columns=['pid'], errors='ignore')
    # Prepare training and testing event times and indicators
    train_event_times = data_pre['time'].values
    train_event_indicators = data_pre['event'].values.astype(bool)  # Ensure boolean type
    test_event_times = test_pre['time'].values
    test_event_indicators = test_pre['event'].values.astype(bool)

    # Prepare the structured array needed for RSF model input
    y_train = np.array(list(zip(train_event_indicators, train_event_times)),
                       dtype=[('event', 'bool'), ('time', 'float')])
    y_test = np.array(list(zip(test_event_indicators, test_event_times)), dtype=[('event', 'bool'), ('time', 'float')])

    # Initialize and fit the Random Survival Forest model
    rsf = RandomSurvivalForest(
        n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=random_state
    )
    rsf.fit(X=data_pre, y=y_train)

    # Predict survival function and cumulative hazard function
    rsf_survival_function = rsf.predict_survival_function(data_pre)
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

    return rsf, rsf_survival_function, hazard, feature_importance


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
    cox_model_evals = LifelinesEvaluator(cox_survival_function, train_event_times, train_event_indicators, test_event_times, test_event_indicators)
    rsf_model_evals = ScikitSurvivalEvaluator(rsf_survival_function, train_event_times, train_event_indicators, test_event_times, test_event_indicators)

    metrics = {
        "concordance_cox": cox_model_evals.concordance()[0],
        "concordance_rsf": rsf_model_evals.concordance()[0],
        "brier_cox": cox_model_evals.brier()[0],
        "brier_rsf": rsf_model_evals.brier()[0],
        "ibs_cox": cox_model_evals.integrated_brier_score()[0],
        "ibs_rsf": rsf_model_evals.integrated_brier_score()[0],
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

def plot_coefficients_rsf(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")


def plot_cox_coefficients(cox_model):
    """
    Plot the coefficients from a Cox Proportional Hazards model, focusing on the effect after Lasso regularization.

    Args:
    - cox_model (CoxPHFitter): A fitted CoxPHFitter model from lifelines.

    Returns:
    - None: Displays a plot of the coefficients.
    """
    coefs = cox_model.params_
    std_errors = cox_model.standard_errors_

    fig, ax = plt.subplots(figsize=(8, len(coefs) / 2))
    coefs.plot(kind='barh', ax=ax, xerr=std_errors, color='blue', ecolor='black', capsize=4)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Coefficients from Cox Proportional Hazards Model')
    plt.show()

def plot_model_residuals(residuals_info):
    # Number of models to plot for
    num_models = len(residuals_info)

    # Setup the figure layout
    fig, axes = plt.subplots(num_models, 1, figsize=(10, 6 * num_models), squeeze=False)

    # Iterate over each model's residuals information
    for i, info in enumerate(residuals_info):
        ax = axes[i][0]
        for covariate, values in info['residual_data'].items():
            ax.plot(values.index, values, marker='o', linestyle='-', label=f'{covariate} ({info["residual_type"]})')

        # Customize plot
        ax.set_title(f'{info["model_name"]} Residuals')
        ax.set_xlabel('Time')
        ax.set_ylabel('Residuals')
        ax.axhline(y=0, linestyle='--', color='grey', alpha=0.7)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

def cross_validate_and_plot(models, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {name: [] for name in models.keys()}

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        for name, model in models.items():
            if name == 'CoxPH':
                model.fit(pd.concat([X_train, y_train], axis=1), 'time', 'event')
                preds = model.predict_partial_hazard(X_val)
            elif name == 'RSF':
                model.fit(X_train, y_train)
                preds = -model.predict(X_val)  # Negative for concordance_index
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
