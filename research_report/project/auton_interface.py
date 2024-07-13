from auton_survival.preprocessing import StandardScaler, SimpleImputer
from auton_survival.phenotyping import ClusteringPhenotyper, IntersectionalPhenotyper, SurvivalVirtualTwinsPhenotyper
from config import load_datasets

def load_dataset():
    datasets = load_datasets(folder="datasets", names=['...'])
    return datasets
def identify_features(dataset):
    featureIdentifier = ClusteringPhenotyper(clustering_method="kmeans", dim_red_method="pca", random_seed=100)
    return featureIdentifier.fit_predict(dataset)

def normalise_data(dataset):
    normalizer = StandardScaler()
    normalizer.fit(dataset)
    return normalizer.transform(dataset)

def impute_data(dataset):
    data_imputer = SimpleImputer()
    data_imputer.fit(dataset)
    return data_imputer.transform(dataset)

def preprocess(dataset):
    return impute_data(normalise_data(dataset))


