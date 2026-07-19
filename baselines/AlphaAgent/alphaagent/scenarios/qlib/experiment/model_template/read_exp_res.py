import pickle
from pathlib import Path

import pandas as pd
import qlib
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

# Offline guard: block live data access outside the unified load_datasets.py
# flow. See dataloader/_offline_guard.py. Opt in with MASE_ALLOW_LEGACY_FETCH=1.
import os
import sys


def _mase_repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "load_datasets.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("Cannot locate MASE repo root (load_datasets.py).")


sys.path.insert(0, _mase_repo_root())
from dataloader._offline_guard import assert_offline_or_optin as _mase_offline_guard

_mase_offline_guard("baselines/AlphaAgent/.../model_template/read_exp_res.py (qlib.init)")
qlib.init()

from qlib.workflow import R

# here is the documents of the https://qlib.readthedocs.io/en/latest/component/recorder.html

# TODO: list all the recorder and metrics

# Assuming you have already listed the experiments
experiments = R.list_experiments()

# Iterate through each experiment to find the latest recorder
experiment_name = None
latest_recorder = None
for experiment in experiments:
    recorders = R.list_recorders(experiment_name=experiment)
    for recorder_id in recorders:
        if recorder_id is not None:
            experiment_name = experiment
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
            end_time = recorder.info["end_time"]
            if latest_recorder is None or end_time > latest_recorder.info["end_time"]:
                latest_recorder = recorder

# Check if the latest recorder is found
if latest_recorder is None:
    print("No recorders found")
else:
    print(f"Latest recorder: {latest_recorder}")

    # Load the specified file from the latest recorder
    metrics = pd.Series(latest_recorder.list_metrics())

    output_path = Path(__file__).resolve().parent / "qlib_res.csv"
    metrics.to_csv(output_path)

    print(f"Output has been saved to {output_path}")

    ret_data_frame = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    ret_data_frame.to_pickle("ret.pkl")
