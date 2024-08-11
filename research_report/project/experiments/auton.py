import numpy as np
from auton_survival.preprocessing import StandardScaler, SimpleImputer
from auton_survival.phenotyping import ClusteringPhenotyper
from utils.config import load_datasets


class CustomPipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit_transform(self, dataset):
        for step in self.steps:
            method_name, method = step
            dataset = method(dataset)
        return dataset

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


pipeline = CustomPipeline(steps=[
    ('identify_features', identify_features),
    ('scale_in_place', scale_in_place),
    ('impute_in_place', impute_in_place)
])

dataset = load_datasets(folder="../outputs/datasets", names=['dataDIVAT1_train'])


processed_df = pipeline.fit_transform(dataset['dataDIVAT1_train'])

print(processed_df)