from copy import deepcopy
from uqpylab import sessions
from configs.competitor_config import competitor_config
from configs.case_study_config import case_study_config
from models.utils import load_object
import case_studies as cs
import competitor_benchmark as cb
import os


def dataset_from_settings(settings):
    """Return a dataset from the settings dictionary."""
    assert settings["CS_type"] == "ML", "Only ML case studies supported."
    dataset_settings = {key: value for key, value in settings.items()
                        if key not in ["CS_type", "Name", "rep_iter"]}
    dataset = cs.Dataset.from_data(dataset_settings)
    return dataset


def generate_replications(settings, rep_num=1, test_N=100000, uq=None):
    settings = deepcopy(settings)
    name = settings["Name"] if "Name" in settings else "case_study"
    del settings["Name"]
    rep_iter = settings["rep_iter"]
    del settings["rep_iter"]

    if settings["CS_type"] == "ML":
        dataset = dataset_from_settings(settings)
        # Generate replications
        replications = cs.ReplicationsList.from_KFold(
            dataset.X, dataset.y, rep_num, rep_iter, name=name,
            X_names=dataset.feature_names, y_name=dataset.target_names)

    elif settings["CS_type"] == "UQ":
        if uq is None:
            mySession = sessions.cloud()
            uq = mySession.cli
            mySession.reset()
        del settings["CS_type"]
        # Generate replications
        replications = cs.ReplicationsList.from_Model(
            rep_num, settings["ModelOpts"], settings["InputOpts"],
            train_N=rep_iter, test_N=test_N, uq=uq, name=name)

    return replications


def run_benchmark(competitors: list, case_study: str,
                  rep_path=None, rep_num=1, file_dir="", verbose=0,
                  case_study_config=case_study_config,
                  competitor_config=competitor_config,
                  hypertune_iter=1):
    """Run a benchmark for a given case study and competitors. """

    case_study_settings = deepcopy(case_study_config)
    competitor_settings = deepcopy(competitor_config)
    file_dir = os.path.dirname(os.path.abspath(__file__)) \
        if file_dir == "" else file_dir
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    info_benchmarks = {}
    benchmarks = {}
    benchmarks["name"] = case_study
    info_benchmarks["name"] = case_study
    if rep_path is None:
        rep_settings = case_study_settings[case_study]
        case_study_reps = generate_replications(
            settings=rep_settings, rep_num=rep_num)
        rep_path = f'{rep_settings["Name"]}_' \
            f'{"_".join(map(str,rep_settings["rep_iter"]))}'
        case_study_reps.save(file_dir, rep_path)
        rep_path = f"{file_dir}/{rep_path}.obj"
    else:
        case_study_reps = load_object(rep_path)
    benchmarks["replications"] = case_study_reps
    info_benchmarks["replications"] = rep_path
    if verbose > 0:
        print(f"Generated {rep_num} replications for {case_study}.")
    for comp_name in competitors:
        comp_config = competitor_settings[comp_name]
        competitor_benchmark, comp_info = cb.competitor_benchmark(
            case_study_reps, comp_name, config=comp_config["config"],
            report_dir=file_dir, verbose=verbose,
            hypertune_iter=hypertune_iter, return_path=True)
        benchmarks[comp_name] = competitor_benchmark
        info_benchmarks[comp_name] = comp_info
        if verbose > 0:
            print(f"Benchmark ran for {comp_name}.")

    info_file = f"{file_dir}/benchmark_{info_benchmarks['name']}_info.txt"
    with open(info_file, 'w') as f:
        for key in info_benchmarks.keys():
            f.write(f"{key}: {info_benchmarks[key]}\n")

    return benchmarks
