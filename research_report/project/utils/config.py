from pathlib import Path
from pandas import DataFrame
import pickle
from typing import List, Dict

def get_config():
    return {
        "datasource": "opus_books",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def save_dataset(df: DataFrame, name: str, folder: str) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    file = Path(folder) / f"{name}.pkl"
    with open(file, 'wb') as f:
        pickle.dump(df, f)

def load_datasets(folder: str, names: List[str]) -> Dict[str, DataFrame]:
    datasets = {}
    for name in names:
        file = Path(folder) / f"{name}.pkl"
        with open(file, 'rb') as f:
            datasets[name] = pickle.load(f)
    return datasets