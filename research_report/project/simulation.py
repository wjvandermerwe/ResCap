# stdlib
import os
import sys


omp_include_path = '/opt/homebrew/Cellar/libomp/18.1.8/include'
os.environ['OMP_PATH'] = omp_include_path
import warnings
from config import load_datasets
import synthcity.logger as log
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins import Plugins
from synthcity.metrics.eval_sanity import  DataMismatchScore, CloseValuesProbability
# log.add(sink=sys.stderr, level="DEBUG")
log.add(sink=sys.stderr, level="DEBUG")
# log_level = logging.getLevelName("DEBUG")
# logging.basicConfig(stream=sys.stdout, level=log_level)
warnings.filterwarnings("ignore")

# data = SurvivalAnalysisDataLoader(X,
# target_column = "arrest",
# time_to_event_column = "week",
# )
datasets = load_datasets("datasets",['dataDIVAT1_train'])
data = datasets['dataDIVAT1_train']

# Convert numeric columns to float32
data_loader = SurvivalAnalysisDataLoader(data, target_column="event", time_to_event_column="time")

syn_model = Plugins().get("survival_gan")

syn_model.fit(data_loader)
syn_model.generate(10)
syn_model.save()
# plugin.generate(count=50)


close = CloseValuesProbability()
data_mismatch = DataMismatchScore()

close.evaluate()