import pandas as pd
from synthcity.plugins import Plugins
from synthcity.metrics.eval_sanity import CloseValuesProbability, DataMismatchScore, CommonRowsProportion, NearestSyntheticNeighborDistance, DistantValuesProbability
from synthcity.utils.serialization import save_to_file
from config import save_dataset, save_checkpoint, load_checkpoint, get_train_dataset_indexes, load_datasets
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader, TimeSeriesSurvivalDataLoader
import torch
from preprocess import preprocess


# https://github.com/vanderschaarlab/synthcity/issues/249
def run_surv_gan(data_loader, device):
    model = Plugins().get("survival_gan", device=device)
    model.fit(data_loader)
    return model


def run_surv_vae(data_loader, device):
    model = Plugins().get("survae", device=device)
    model.fit(data_loader)
    return model


def run_timegan(df: pd.DataFrame, device):
    loader = prepare_data_for_timegan(df)


    model = Plugins().get("timegan", device=device)
    model.fit(loader)
    return model


def prepare_data_for_timegan(df: pd.DataFrame) -> TimeSeriesSurvivalDataLoader:
    # Ensure required columns are present
    required_columns = ['pid', 'event', 'time', 'time2']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Extract unique PIDs and ensure consistency
    unique_pids = df['pid'].drop_duplicates().reset_index(drop=True)

    # Separate static, temporal, and time-related data
    static_data = df.drop(columns=['time', 'time2', 'event']).drop_duplicates(subset='pid').set_index('pid').reset_index(drop=True)

    # Temporal data indexed by 'pid' and 'time'
    temporal_data = []
    observation_times = []
    for pid, group in df.groupby('pid'):
        temporal_df = group.drop(columns=['pid', 'event', 'time2']).set_index('time').reset_index(drop=True)
        if not temporal_df.empty:
            temporal_data.append(temporal_df)
            observation_times.append(group['time'].values)
        else:
            raise ValueError(f"Temporal data for PID {pid} is empty.")

    # Time-to-event (T) and event indicators (E)
    T = df.groupby('pid')['time2'].max().reset_index(drop=True)
    E = df.groupby('pid')['event'].max().reset_index(drop=True)

    # Check that all lists have the same length
    if len(temporal_data) != len(observation_times) or len(temporal_data) != len(T) or len(T) != len(E):
        raise ValueError("Mismatch in lengths of temporal data, observation times, T, or E.")

    # Create the TimeSeriesSurvivalDataLoader
    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        T=T,
        E=E,
        static_data=static_data,
        random_state=0,
        train_size=0.8
    )

    return loader

def evaluate_simulated_model_data(data_loader, generated_data):
    # Initialize metrics
    close = CloseValuesProbability()
    data_mismatch = DataMismatchScore()
    proportion = CommonRowsProportion()
    nn_distance = NearestSyntheticNeighborDistance()
    distant = DistantValuesProbability()

    # Evaluate metrics
    close_val = close.evaluate(data_loader, generated_data)['score']
    mis = data_mismatch.evaluate(data_loader, generated_data)['score']
    prop = proportion.evaluate(data_loader, generated_data)['score']
    nn_dist = nn_distance.evaluate(data_loader, generated_data)['mean']
    dist = distant.evaluate(data_loader, generated_data)['score']

    # Determine if metrics are within expected values
    correct = True

    # Define correctness based on descriptions
    if not (0 <= close_val <= 1):
        correct = False
    if not (0 <= mis <= 1):
        correct = False
    if not (0 <= prop <= 1):
        correct = False
    if nn_dist < 0:
        correct = False
    if not (0 <= dist <= 1):
        correct = False

    # Prepare results
    results = {
        'close_values': {
            "value": close_val,
        },
        'data_mismatch': {
            "value": mis,
        },
        'proportion': {
            "value": prop,
        },
        'nn_distance': {
            "value": nn_dist,
        },
        'distant_values': {
            "value": dist,
        }
    }

    return results, correct

def check_for_time_varying_features(df: pd.DataFrame) -> bool:
    # Implement your logic to determine if TimeGAN should be used.
    # This can be based on whether there are time-varying features,
    # or any other specific criteria related to the dataset.
    # For example:
    return 'time2' in df.columns and df['time2'].notna().any()


def run_datasets_simulation(dataset_indexes, datasets):
    start_index = load_checkpoint()
    device = torch.device('cuda')
    print(torch.cuda.is_available());
    for index, dataset_index in enumerate(dataset_indexes):
        if index < start_index:
            continue
        print(f"training model on {dataset_index}")
        ds_train = datasets[dataset_index]
        ds_train = preprocess(ds_train)
        # for simulation remove 0
        ds_train = ds_train[ds_train['time'] > 0]
        # print(ds_train.head())

        # Check for time-varying features or other criteria to decide on using TimeGAN
        use_timegan = check_for_time_varying_features(ds_train)  # Implement this function based on your criteria

        data_loader = SurvivalAnalysisDataLoader(ds_train, target_column="event",
                                                 time_to_event_column="time")

        if use_timegan:
            print(f"Using TimeGAN for dataset {dataset_index}")
            timegan_model = run_timegan(ds_train, device)
            save_to_file(f"../outputs/model_outputs/sim_model_{dataset_index}_timegan.pkl", timegan_model)
            generated_data = timegan_model.generate(5000)
            generated_data = generated_data.dataframe()
            save_dataset(generated_data, f"{dataset_index}_timegan", "../outputs/generated_datasets")
        else:
            surv_gan_model = run_surv_gan(data_loader, device)
            surv_vae_model = run_surv_vae(data_loader, device)

            save_to_file(f"../outputs/model_outputs/sim_model_{dataset_index}_gan.pkl", surv_gan_model)
            save_to_file(f"../outputs/model_outputs/sim_model_{dataset_index}_vae.pkl", surv_vae_model)

            generated_data_gan = surv_gan_model.generate(5000)
            generated_data_vae = surv_vae_model.generate(5000)

            _, eval_gan = evaluate_simulated_model_data(data_loader, generated_data_gan)
            _, eval_vae = evaluate_simulated_model_data(data_loader, generated_data_vae)

            print(f"training completed, eval: gan:{eval_gan} vae:{eval_vae}")

            # Save generated datasets
            generated_data_gan = generated_data_gan.dataframe()
            generated_data_vae = generated_data_vae.dataframe()

            save_dataset(generated_data_gan, f"{dataset_index}_gan", "../outputs/generated_datasets")
            save_dataset(generated_data_vae, f"{dataset_index}_vae", "../outputs/generated_datasets")

        save_checkpoint(index)


folder = "../outputs/datasets" # load saved data
dataset_indexes = get_train_dataset_indexes(folder)
print(dataset_indexes)
datasets = load_datasets(folder=folder, names=dataset_indexes)
run_datasets_simulation(dataset_indexes, datasets)

