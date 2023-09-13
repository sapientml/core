"""
SapientML uses many artifacts such as project corpus, meta-models (skeleton predictor), data-flow model, and component
templates that are compiled in the training phase. These artifacts are stored in pre-defined paths relative to the AutoML
root directory. This script defines all the internal paths SapientML uses to locate various artifacts.
"""


from pathlib import Path

sapientml_core_root = Path(__file__).parents[0]

adaptation_root_dir = sapientml_core_root / "adaptation"
artifacts_path = adaptation_root_dir / "artifacts"
model_path = sapientml_core_root / "models"

benchmark_path = sapientml_core_root / "benchmarks"
corpus_path = sapientml_core_root / "corpus"
training_cache = sapientml_core_root / ".cache"

execution_cache_dir = training_cache / "exec_info"
analysis_dir = training_cache / "analysis"
clean_notebooks_dir_name = "clean-notebooks"
clean_dir = corpus_path / clean_notebooks_dir_name
project_labels_path = corpus_path / "annotated-notebooks" / "annotated-notebooks-1140.csv"
