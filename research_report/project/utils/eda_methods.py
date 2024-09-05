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


def univariate_analysis(dataset: pd.DataFrame) -> None:
    dataset = dataset.drop(columns=['time', 'event', 'phenotypes'])

    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = dataset.select_dtypes(include=[object]).columns.tolist()

    # Set the number of columns and max plots per figure
    columns_per_row = 3
    max_plots_per_figure = 9  # Maximum of 9 plots per figure

    def plot_chunk(features, plot_func, title_suffix, filename_suffix):
        for chunk_start in range(0, len(features), max_plots_per_figure):
            chunk = features[chunk_start:chunk_start + max_plots_per_figure]
            num_rows = int(np.ceil(len(chunk) / columns_per_row))
            fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(5 * columns_per_row, 6 * num_rows))
            axes = axes.flatten()

            for i, feature in enumerate(chunk):
                plot_func(feature, axes[i])
                axes[i].set_title(f'{title_suffix} {feature}')

            # Remove empty subplots
            for j in range(len(chunk), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout(pad=3.0)
            plt.savefig(f'{filename_suffix}_{chunk_start}.png')
            plt.show()
            plt.close(fig)

    # Plot functions
    def histplot_numeric(feature, ax):
        sns.histplot(dataset[feature], kde=True, ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')

    def countplot_categorical(feature, ax):
        sns.countplot(x=dataset[feature], ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')

    # Numerical Covariates
    if numerical_features:
        plot_chunk(numerical_features, histplot_numeric, 'Distribution of', 'numerical_covariates')

    # Categorical Covariates
    if categorical_features:
        plot_chunk(categorical_features, countplot_categorical, 'Distribution of', 'categorical_covariates')


def bivariate_analysis(dataset: pd.DataFrame) -> None:
    # Identify numerical features
    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()

    # Identify one-hot encoded categorical features
    categorical_features = [col for col in dataset.columns if col.startswith('fac_')]

    # Set the number of columns and max plots per figure
    columns_per_row = 3
    max_plots_per_figure = 9  # Maximum of 9 plots per figure

    def plot_chunk(features, plot_func, title_suffix, filename_suffix):
        for chunk_start in range(0, len(features), max_plots_per_figure):
            chunk = features[chunk_start:chunk_start + max_plots_per_figure]
            num_rows = int(np.ceil(len(chunk) / columns_per_row))
            fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(5 * columns_per_row, 6 * num_rows))
            axes = axes.flatten()

            for i, feature in enumerate(chunk):
                plot_func(feature, axes[i])
                axes[i].set_title(f'{title_suffix} {feature}')

            # Remove empty subplots
            for j in range(len(chunk), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout(pad=3.0)
            plt.savefig(f'{filename_suffix}_{chunk_start}.png')
            plt.show()
            plt.close(fig)

    # Plot functions
    def scatterplot_numeric(feature, ax):
        sns.scatterplot(x=dataset[feature], y=dataset['time'], ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel('Survival Time')

    def boxplot_categorical(feature, ax):
        sns.boxplot(x=dataset[feature], y=dataset['time'], ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel('Survival Time')

    def violinplot_numeric(feature, ax):
        sns.violinplot(x=dataset['event'], y=dataset[feature], ax=ax)
        ax.set_xlabel('Event Status')
        ax.set_ylabel(feature)

    # Survival Time vs. Covariates - Numerical
    if numerical_features:
        plot_chunk(numerical_features, scatterplot_numeric, 'Survival Time vs.', 'survival_time_vs_numerical')

    # Survival Time vs. Covariates - One-Hot Encoded Categorical
    if categorical_features:
        plot_chunk(categorical_features, boxplot_categorical, 'Survival Time vs.', 'survival_time_vs_categorical')

    # Event Status vs. Covariates - Numerical
    if numerical_features:
        plot_chunk(numerical_features, violinplot_numeric, '', 'event_status_vs_numerical')


def censoring_analysis(dataset: pd.DataFrame) -> None:
    # Overall Censoring Level
    censoring_level = dataset['event'].value_counts(normalize=True)
    print("\nOverall Censoring Level:")
    print(censoring_level)

    # Plot Overall Censoring Level
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=censoring_level.index, y=censoring_level.values, color='blue', alpha=0.6, ax=ax)
    ax.set_title('Overall Censoring Level')
    ax.set_xlabel('Event Status')
    ax.set_ylabel('Proportion')
    plt.tight_layout(pad=3.0)
    plt.savefig('overall_censoring_level.png')
    plt.show()
    plt.close(fig)

    # Censoring vs. Covariates - Numerical
    numerical_features = [col for col in dataset.columns if col.startswith('num_')]
    if numerical_features:
        num_rows = int(np.ceil(len(numerical_features) / 3))
        fig, axes = plt.subplots(num_rows, 3, figsize=(5 * 3, 6 * num_rows))
        axes = axes.flatten()

        for i, feature in enumerate(numerical_features):
            sns.boxplot(x=dataset['event'], y=dataset[feature], ax=axes[i], boxprops=dict(alpha=0.6))
            axes[i].set_title(f'{feature} by Censoring Status')
            axes[i].set_xlabel('Event Status')
            axes[i].set_ylabel(feature)

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=3.0)
        plt.show()
        plt.close(fig)


def correlation_analysis(dataset: pd.DataFrame) -> None:
    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = dataset.select_dtypes(include=[object]).columns.tolist()

    # Numerical Covariates Correlation
    corr_matrix = dataset[numerical_features].corr()
    plt.figure(figsize=(30, 15))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Covariates')
    # plt.show()

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

