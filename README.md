# Benchmarks-UQ-ML

Allows running of benchmarks to compare different models from the Uncertainty Quantification (UQ) field and the Machine Learning (ML) field.

The configuration of the case studies examined can be found in [case_study_config.py](configs/case_study_config.py).

The configuration of the competitors benchmarked can be foind in [competitor_config.py](configs/competitor_config.py).

The benchmark can be run with [run_benchmark.py](run_benchmark.py).

Usage is:
```
python3 run_benchmark.py --case_study "ishigami" --competitors "xgb" "rf" --rep_num 2 --file_dir  "" --verbose 1 --hypertune_iter 1 
```
The `--options` are:

- `--case_study`: Case study to run. Default: `"ishigami"`.
- `--competitors`: Competitors to run.. Default: `"rf" "xgb" "mlp" "resnet" "ft_transformer" "pce" "kriging" "pck"`.
- `--rep_path`: Path to replications if taken from object. Default: `None`.
- `--rep_num`: Number of replications to generate. Default: `1`.
- `--file_dir`: Directory to save the results. Default: `""`.
- `--verbose`: Verbosity level. Default: `0`.
- `--hypertune_iter`: Number of iterations for hyperparameter tuning. Default: `20`.
