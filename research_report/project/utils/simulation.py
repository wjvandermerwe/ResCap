from synthcity.plugins import Plugins
from synthcity.metrics.eval_sanity import CloseValuesProbability, DataMismatchScore, CommonRowsProportion, NearestSyntheticNeighborDistance, DistantValuesProbability
from synthcity.utils.serialization import save_to_file
from utils.config import save_dataset, save_checkpoint, load_checkpoint
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader, TimeSeriesSurvivalDataLoader
import torch

from utils.preprocess import preprocess


# https://github.com/vanderschaarlab/synthcity/issues/249
def run_surv_gan(data_loader, device):
    model = Plugins().get("survival_gan", device=device)
    model.fit(data_loader)
    return model


def run_surv_vae(data_loader, device):
    model = Plugins().get("survae", device=device)
    model.fit(data_loader)
    return model


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

def run_datasets_simulation(dataset_indexes, datasets):
    start_index = load_checkpoint()
    device = torch.device('cuda')

    for index, dataset_index in enumerate(dataset_indexes):
        if index < start_index:
            continue
        print(f"training model on {dataset_index}")
        ds_train = datasets[dataset_index]
        ds_train = preprocess(ds_train)
        data_loader = SurvivalAnalysisDataLoader(ds_train, target_column="remainder__event",
                                                 time_to_event_column="remainder__time", )
        surv_gan_model = run_surv_gan(data_loader, device)
        surv_vae_model = run_surv_vae(data_loader, device)

        save_to_file(f"../outputs/model_outputs/sim_model_{dataset_index}_gan.pkl", surv_gan_model)
        save_to_file(f"../outputs/model_outputs/sim_model_{dataset_index}_vae.pkl", surv_vae_model)

        generated_data_gan = surv_gan_model.generate(5000)
        generated_data_vae = surv_vae_model.generate(5000)

        _, eval_gan = evaluate_simulated_model_data(data_loader, generated_data_gan)
        _, eval_vae = evaluate_simulated_model_data(data_loader, generated_data_vae)

        print(f"training completed, eval: gan:{eval_gan} vae:{eval_vae}")

        generated_data_vae = generated_data_vae.dataframe()
        generated_data_gan = generated_data_gan.dataframe()

        save_dataset(generated_data_gan, f"{dataset_index}_gan", "../outputs/generated_datasets")
        save_dataset(generated_data_vae, f"{dataset_index}_vae", "../outputs/generated_datasets")
        save_checkpoint(index)