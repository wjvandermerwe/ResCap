from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
import synthcity.plugins.core.models as models
from synthcity.metrics import

from synthcity.

# SurvivalGAN Implementation
class SurvivalGAN(Plugin):
    def __init__(self):
        super().__init__("SurvivalGAN")

    def _fit(self, X, T, E):
        self.model = models.SurvivalGAN()
        self.model.fit(X, T, E)

    def _sample(self, n):
        return self.model.sample(n)

# VAE Implementation
class VAE(Plugin):
    def __init__(self):
        super().__init__("VAE")

    def _fit(self, X):
        self.model = models.VAE()
        self.model.fit(X)

    def _sample(self, n):
        return self.model.sample(n)
class ModelEvaluator:
    def __init__(self, dataset):
        self.dataset = dataset

    def evaluate_survival_gan(self, model, samples):
        eval_results = Eval.survival_eval(model, self.dataset)
        c_index = eval_results["c-index"]
        print(f"SurvivalGAN c-index: {c_index}")
        self.plot_survival_curves(self.dataset, samples, "T", "E")
        return eval_results

    def evaluate_vae(self, model, samples):
        eval_results = Eval.tabular_eval(model, self.dataset)
        fid_score = eval_results["fid"]
        mmd_score = eval_results["mmd"]
        print(f"VAE FID: {fid_score}")
        print(f"VAE MMD: {mmd_score}")
        columns_to_plot = self.dataset["X"].columns[:5]  # Adjust number of columns as needed
        self.plot_distributions(self.dataset["X"], samples, columns_to_plot)
        return eval_results

    def plot_survival_curves(self, original_data, synthetic_data, time_column, event_column):
        fig, ax = plt.subplots()
        sns.kaplan_meier(original_data, time_column, event_column, label='Original', ax=ax)
        sns.kaplan_meier(synthetic_data, time_column, event_column, label='Synthetic', ax=ax)
        plt.title('Survival Curves')
        plt.legend()
        plt.show()

    def plot_distributions(self, original_data, synthetic_data, columns):
        for column in columns:
            plt.figure()
            sns.histplot(original_data[column], kde=True, color='blue', label='Original', stat='density')
            sns.histplot(synthetic_data[column], kde=True, color='orange', label='Synthetic', stat='density')
            plt.title(f'Distribution of {column}')
            plt.legend()
            plt.show()

# Create and fit SurvivalGAN plugin
survival_gan = SurvivalGAN()
survival_gan.fit(dataset["X"], dataset["T"], dataset["E"])

# Create and fit VAE plugin
vae = VAE()
vae.fit(dataset["X"])

# Generate samples from the trained models
survival_gan_samples = survival_gan.sample(100)
vae_samples = vae.sample(100)

# Initialize evaluator and run evaluations
evaluator = ModelEvaluator(dataset)

# Evaluate models
survival_gan_eval_results = evaluator.evaluate_survival_gan(survival_gan, survival_gan_samples)
vae_eval_results = evaluator.evaluate_vae(vae, vae_samples)

# Print detailed evaluations
print("Detailed SurvivalGAN Evaluation:", survival_gan_eval_results)
print("Detailed VAE Evaluation:", vae_eval_results)
# Create and fit SurvivalGAN plugin
survival_gan = SurvivalGAN()
survival_gan.fit(dataset.X, dataset.T, dataset.E)

# Create and fit VAE plugin
vae = VAE()
vae.fit(dataset.X)

# Generate samples from the trained models
survival_gan_samples = survival_gan.sample(100)
vae_samples = vae.sample(100)

# Evaluate the models
survival_gan_eval = Eval.survival_eval(survival_gan, dataset)
vae_eval = Eval.tabular_eval(vae, dataset)

print("SurvivalGAN Evaluation:", survival_gan_eval)
print("VAE Evaluation:", vae_eval)