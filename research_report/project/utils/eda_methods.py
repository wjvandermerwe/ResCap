from lifelines import KaplanMeierFitter, NelsonAalenFitter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import numpy as np
from auton_survival.phenotyping import (
    ClusteringPhenotyper,
)
from auton_survival.reporting import plot_kaplanmeier, plot_nelsonaalen, add_at_risk_counts
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, chi2_contingency

def identify_features(dataset: pd.DataFrame) -> pd.DataFrame:
    # Identify numerical features
    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
    phenotyper_clustering = ClusteringPhenotyper(
        clustering_method='kmeans',
        dim_red_method='pca',
        random_seed=100
    )
    cluster_labels = phenotyper_clustering.fit_predict(dataset[numerical_features])
    dataset['phenotypes'] = cluster_labels
    return dataset


def plot_phenotypes_analysis(dataset: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from lifelines import KaplanMeierFitter

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()

    # Cluster Distribution (Composition)
    sns.countplot(x='phenotypes', data=dataset, ax=axes[0])
    axes[0].set_title('Cluster Distribution')
    axes[0].set_xlabel('Phenotypes')
    axes[0].set_ylabel('Count')

    # Cluster Profile Analysis
    cluster_profile = dataset.groupby('phenotypes')[numerical_features].mean()
    cluster_profile.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Cluster Profile Analysis')
    axes[1].set_xlabel('Phenotypes')
    axes[1].set_ylabel('Mean Values')

    plt.tight_layout()
    plt.show()

def survival_analysis_plots(dataset: pd.DataFrame, group_col: str = None) -> None:
    if 'time' not in dataset.columns or 'event' not in dataset.columns:
        raise ValueError("Dataset must contain 'time' and 'event' columns.")

    # Automatically detect the grouping column if not specified
    if group_col is None:
        group_col = next(col for col in dataset.columns if col not in ['time', 'event'])

    # Kaplan-Meier Survival Curve
    plt.figure(figsize=(12, 6))
    kmf = KaplanMeierFitter()
    for group in dataset[group_col].unique():
        group_data = dataset[dataset[group_col] == group]
        kmf.fit(durations=group_data['time'], event_observed=group_data['event'], label=str(group))
        kmf.plot_survival_function()

    plt.title('Kaplan-Meier Survival Curve')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend(title=group_col)
    plt.show()

    # Nelson-Aalen Cumulative Hazard Function
    plt.figure(figsize=(12, 6))
    naf = NelsonAalenFitter()
    for group in dataset[group_col].unique():
        group_data = dataset[dataset[group_col] == group]
        naf.fit(durations=group_data['time'], event_observed=group_data['event'], label=str(group))
        naf.plot_cumulative_hazard()

    plt.title('Nelson-Aalen Cumulative Hazard Function')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Hazard')
    plt.legend(title=group_col)
    plt.show()



def calculate_vif(data: pd.DataFrame) -> pd.DataFrame:
    X = data.drop(columns=['remainder__time', 'remainder__event'])
    X = add_constant(X)
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["Variable Inflation Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif


def data_overview(dataset: pd.DataFrame) -> None:
    # Dataset Dimensions
    print(f"Dataset dimensions: {dataset.shape}")

    # Data Types
    print("\nData Types:")
    print(dataset.dtypes)

    # Missing Values
    print("\nMissing Values:")
    missing_values = dataset.isnull().sum()
    print(missing_values[missing_values > 0])

    plt.figure(figsize=(12, 8))
    sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()


def univariate_analysis(dataset: pd.DataFrame) -> None:

    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = dataset.select_dtypes(include=[object]).columns.tolist()

    # Determine the total number of plots
    total_plots = len(numerical_features) + len(categorical_features)

    # Creating a figure with subplots for each analysis
    fig, axes = plt.subplots(1, total_plots, figsize=(5 * total_plots, 6))
    fig.tight_layout(pad=5.0)

    # Numerical Covariates
    for i, feature in enumerate(numerical_features):
        print(f"\n{feature} Statistics:")
        print(dataset[feature].describe())

        sns.histplot(dataset[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Density')

    # Categorical Covariates
    for i, feature in enumerate(categorical_features):
        print(f"\n{feature} Frequency Counts:")
        print(dataset[feature].value_counts())

        sns.countplot(x=dataset[feature], ax=axes[len(numerical_features) + i])
        axes[len(numerical_features) + i].set_title(f'Distribution of {feature}')
        axes[len(numerical_features) + i].set_xlabel(feature)
        axes[len(numerical_features) + i].set_ylabel('Count')

    plt.show()



def bivariate_analysis(dataset: pd.DataFrame) -> None:

    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = dataset.select_dtypes(include=[object]).columns.tolist()

    # Determine the total number of plots
    total_plots = len(numerical_features) + len(categorical_features)

    # Creating a figure with subplots for each analysis
    fig, axes = plt.subplots(2, total_plots, figsize=(5 * total_plots, 12))
    fig.tight_layout(pad=5.0)

    # Survival Time vs. Covariates - Numerical
    for i, feature in enumerate(numerical_features):
        sns.scatterplot(x=dataset[feature], y=dataset['time'], ax=axes[0, i])
        axes[0, i].set_title(f'Survival Time vs. {feature}')
        axes[0, i].set_xlabel(feature)
        axes[0, i].set_ylabel('Survival Time')

    # Survival Time vs. Covariates - Categorical
    for i, feature in enumerate(categorical_features):
        sns.boxplot(x=dataset[feature], y=dataset['time'], ax=axes[0, len(numerical_features) + i])
        axes[0, len(numerical_features) + i].set_title(f'Survival Time vs. {feature}')
        axes[0, len(numerical_features) + i].set_xlabel(feature)
        axes[0, len(numerical_features) + i].set_ylabel('Survival Time')

    # Event Status vs. Covariates - Categorical
    for i, feature in enumerate(categorical_features):
        sns.countplot(x=dataset[feature], hue=dataset['event'], ax=axes[1, i])
        axes[1, i].set_title(f'Event Status vs. {feature}')
        axes[1, i].set_xlabel(feature)
        axes[1, i].set_ylabel('Count')

    # Event Status vs. Covariates - Numerical
    for i, feature in enumerate(numerical_features):
        sns.violinplot(x=dataset['event'], y=dataset[feature], ax=axes[1, len(categorical_features) + i])
        axes[1, len(categorical_features) + i].set_title(f'{feature} Distribution by Event Status')
        axes[1, len(categorical_features) + i].set_xlabel('Event Status')
        axes[1, len(categorical_features) + i].set_ylabel(feature)

    plt.show()



def censoring_analysis(dataset: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Overall Censoring Level
    censoring_level = dataset['event'].value_counts(normalize=True)
    print("\nOverall Censoring Level:")
    print(censoring_level)

    # Number of subplots needed
    n_categorical = len(dataset.select_dtypes(include=[object]).columns.tolist())
    n_numerical = len(dataset.select_dtypes(include=[np.number]).columns.tolist())
    total_plots = 1 + n_categorical + n_numerical

    # Creating a figure with subplots
    fig, axes = plt.subplots(1, total_plots, figsize=(5 * total_plots, 6))
    fig.tight_layout(pad=5.0)

    # Plot Overall Censoring Level
    sns.barplot(x=censoring_level.index, y=censoring_level.values, color='blue', alpha=0.6, ax=axes[0])
    axes[0].set_title('Overall Censoring Level')
    axes[0].set_xlabel('Event Status')
    axes[0].set_ylabel('Proportion')

    # Censoring vs. Covariates - Categorical
    categorical_features = dataset.select_dtypes(include=[object]).columns.tolist()
    for i, feature in enumerate(categorical_features):
        sns.barplot(x=dataset[feature], y=dataset['event'], alpha=0.6, ax=axes[i + 1])
        axes[i + 1].set_title(f'Censoring vs. {feature}')
        axes[i + 1].set_xlabel(feature)
        axes[i + 1].set_ylabel('Count')

    # Censoring vs. Covariates - Numerical
    for i, feature in enumerate(dataset.select_dtypes(include=[np.number]).columns.tolist()):
        sns.boxplot(x=dataset['event'], y=dataset[feature], ax=axes[n_categorical + 1 + i], boxprops=dict(alpha=0.6))
        axes[n_categorical + 1 + i].set_title(f'{feature} by Censoring Status')
        axes[n_categorical + 1 + i].set_xlabel('Event Status')
        axes[n_categorical + 1 + i].set_ylabel(feature)

    plt.show()

def correlation_analysis(dataset: pd.DataFrame) -> None:
    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = dataset.select_dtypes(include=[object]).columns.tolist()

    # Numerical Covariates Correlation
    corr_matrix = dataset[numerical_features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Covariates')
    plt.show()

    # Categorical Covariates
    if categorical_features:
        print("Categorical-Categorical Relationships:")
        for i in range(len(categorical_features)):
            for j in range(i+1, len(categorical_features)):
                cat1 = categorical_features[i]
                cat2 = categorical_features[j]
                contingency_table = pd.crosstab(dataset[cat1], dataset[cat2])
                chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                print(f"Chi-square test between {cat1} and {cat2}: Chi2 = {chi2_stat:.2f}, p-value = {p_value:.4f}")

        print("\nNumerical-Categorical Relationships:")
        for num_feature in numerical_features:
            for cat_feature in categorical_features:
                groups = dataset.groupby(cat_feature)[num_feature].apply(list)
                correlation, p_value = pointbiserialr(dataset[cat_feature].astype('category').cat.codes, dataset[num_feature])
                print(f"Point-biserial correlation between {num_feature} and {cat_feature}: r = {correlation:.2f}, p-value = {p_value:.4f}")

