import numpy as np
from auton_survival.preprocessing import StandardScaler, SimpleImputer
from auton_survival.phenotyping import ClusteringPhenotyper, IntersectionalPhenotyper, SurvivalVirtualTwinsPhenotyper

def identify_features(dataset):
    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()

    featureIdentifier = ClusteringPhenotyper(clustering_method="kmeans", dim_red_method="pca", random_seed=100)
    dataset['cluster_labels'] = featureIdentifier.fit_predict(dataset[numerical_features])

    return dataset


def scale_in_place(dataset):
    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()

    normalizer = StandardScaler()
    dataset[numerical_features] = normalizer.fit_transform(dataset[numerical_features])

    return dataset

def impute_in_place(dataset):
    numerical_features = dataset.select_dtypes(include=[np.number]).columns.tolist()

    data_imputer = SimpleImputer()
    dataset[numerical_features] = data_imputer.fit_transform(dataset[numerical_features])

    return dataset


