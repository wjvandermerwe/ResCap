import os
from pathlib import Path
from pandas import DataFrame
import pickle
from typing import List, Dict
import pandas as pd

def get_config():
    return {
        "datasource": "opus_books",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
    }


def save_dataset(df: DataFrame, name: str, folder: str) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    file = Path(folder) / f"{name}.pkl"
    with open(file, 'wb') as f:
        pickle.dump(df, f)

def load_datasets(folder: str, names: List[str]) -> Dict[str, pd.DataFrame]:
    datasets = {}
    for name in names:
        file = Path(folder) / f"{name}.pkl"
        if file.exists():
            try:
                with open(file, 'rb') as f:
                    datasets[name] = pickle.load(f)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"File {file} does not exist.")
    return datasets



def get_train_dataset_indexes(folder: str) -> List[str]:
    folder_path = Path(folder)
    return [file.stem for file in folder_path.glob("*_train.pkl")]

def get_generated_gan_dataset_indexes(folder: str) -> List[str]:
    folder_path = Path(folder)
    return [file.stem for file in folder_path.glob("*_train_gan.pkl")]

def get_generated_vae_dataset_indexes(folder: str) -> List[str]:
    folder_path = Path(folder)
    return [file.stem for file in folder_path.glob("*_train_vae.pkl")]

def get_test_dataset_indexes(folder: str) -> List[str]:
    folder_path = Path(folder)
    return [file.stem[:-5] for file in folder_path.glob("*_test.pkl")]


checkpoint_file = 'checkpoint.txt'


def load_checkpoint():
    """Load the last completed dataset index from the checkpoint file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return -1  # No checkpoint found


def save_checkpoint(index):
    """Save the current dataset index to the checkpoint file."""
    with open(checkpoint_file, 'w') as f:
        f.write(str(index))
