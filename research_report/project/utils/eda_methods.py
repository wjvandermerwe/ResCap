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
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()

    # Cluster Profile Analysis
    cluster_profile = dataset.groupby('phenotypes')[numerical_features].mean()
    cluster_profile.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Cluster Profile Analysis')

    # Feature Comparison Across Clusters
    for feature in numerical_features:
        sns.boxplot(x='phenotypes', y=feature, data=dataset, ax=axes[1, 0])
        axes[1, 0].set_title(f'Feature Comparison: {feature}')

    # Cluster Distribution
    sns.countplot(x='phenotypes', data=dataset, ax=axes[1, 1])
    axes[1, 1].set_title('Cluster Distribution')

    # Plot separate Kaplan-Meier survival estimates for phenogroups.
    # plot_kaplanmeier(dataset['event'], groups=dataset['phenotypes'])
    # plot_nelsonaalen(dataset['event'], groups=dataset['phenotypes'])
    plt.tight_layout()
    plt.show()

    # Create and show the pairplot in a separate figure
    pairplot_fig = sns.pairplot(dataset, hue='phenotypes', vars=numerical_features)
    pairplot_fig.fig.suptitle('Pairwise Feature Analysis', y=1.02)
    plt.show()


def survival_analysis_plots(dataset: pd.DataFrame) -> None:
    if 'time' not in dataset.columns or 'event' not in dataset.columns:
        raise ValueError("Dataset must contain 'time' and 'event' columns.")

    # Kaplan-Meier Survival Curve
    plt.figure(figsize=(12, 6))
    plot_kaplanmeier(dataset['time'], dataset['event'], title='Kaplan-Meier Survival Curve')
    plt.show()

    # Nelson-Aalen Cumulative Hazard Function
    plt.figure(figsize=(12, 6))
    plot_nelsonaalen(dataset['time'], dataset['event'], title='Nelson-Aalen Cumulative Hazard Function')
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


def target_variable_analysis(dataset: pd.DataFrame) -> None:
    # Survival Time (time)
    print("\nSurvival Time Statistics:")
    print(dataset['time'].describe())

    plt.figure(figsize=(12, 6))
    sns.histplot(dataset['time'], kde=True)
    plt.title('Distribution of Survival Times')
    plt.show()

    # Event Status (event)
    event_counts = dataset['event'].value_counts()
    print("\nEvent Status Distribution:")
    print(event_counts)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=event_counts.index, y=event_counts.values)
    plt.title('Distribution of Event Status')
    plt.xlabel('Event Status')
    plt.ylabel('Count')
    plt.show()


def univariate_analysis(dataset: pd.DataFrame) -> None:
    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = dataset.select_dtypes(include=[object]).columns.tolist()

    # Numerical Covariates
    for feature in numerical_features:
        print(f"\n{feature} Statistics:")
        print(dataset[feature].describe())

        plt.figure(figsize=(12, 6))
        sns.histplot(dataset[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()

    # Categorical Covariates
    for feature in categorical_features:
        print(f"\n{feature} Frequency Counts:")
        print(dataset[feature].value_counts())

        plt.figure(figsize=(12, 6))
        sns.countplot(x=dataset[feature])
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.show()


def bivariate_analysis(dataset: pd.DataFrame) -> None:
    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = dataset.select_dtypes(include=[object]).columns.tolist()

    # Survival Time vs. Covariates
    for feature in numerical_features:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=dataset[feature], y=dataset['time'])
        plt.title(f'Survival Time vs. {feature}')
        plt.xlabel(feature)
        plt.ylabel('Survival Time')
        plt.show()

    for feature in categorical_features:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=dataset[feature], y=dataset['time'])
        plt.title(f'Survival Time vs. {feature}')
        plt.xlabel(feature)
        plt.ylabel('Survival Time')
        plt.show()

    # Event Status vs. Covariates
    for feature in categorical_features:
        plt.figure(figsize=(12, 6))
        sns.countplot(x=dataset[feature], hue=dataset['event'])
        plt.title(f'Event Status vs. {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.show()

    for feature in numerical_features:
        plt.figure(figsize=(12, 6))
        sns.violinplot(x=dataset['event'], y=dataset[feature])
        plt.title(f'{feature} Distribution by Event Status')
        plt.xlabel('Event Status')
        plt.ylabel(feature)
        plt.show()


def censoring_analysis(dataset: pd.DataFrame) -> None:
    # Overall Censoring Level
    censoring_level = dataset['event'].value_counts(normalize=True)
    print("\nOverall Censoring Level:")
    print(censoring_level)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=censoring_level.index, y=censoring_level.values)
    plt.title('Overall Censoring Level')
    plt.xlabel('Event Status')
    plt.ylabel('Proportion')
    plt.show()

    # Censoring vs. Covariates
    categorical_features = dataset.select_dtypes(include=[object]).columns.tolist()

    for feature in categorical_features:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=dataset[feature], hue=dataset['event'])
        plt.title(f'Censoring vs. {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.show()

    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()

    for feature in numerical_features:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=dataset['event'], y=dataset[feature])
        plt.title(f'{feature} by Censoring Status')
        plt.xlabel('Event Status')
        plt.ylabel(feature)
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

