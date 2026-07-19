import qlib

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

_mase_offline_guard("baselines/AlphaAgent/.../factor_data_template/generate.py (qlib.init)")
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
# qlib.init(provider_uri="~/.qlib/qlib_data/us_data")
from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume"]  # , "$amount", "$turn", "$pettm", "$pbmrq"
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2015-01-01":].sort_index()

# 计算收益率
data["$return"] = data.groupby(level=0)["$close"].pct_change().fillna(0)

print(data)

data.to_hdf("./daily_pv_all.h5", key="data")

fields = ["$open", "$close", "$high", "$low", "$volume"]  # , "$amount", "$turn", "$pettm", "$pbmrq"
data = (
    (
        D.features(instruments, fields, freq="day")
        .swaplevel()
        .sort_index()
    )
    .swaplevel()
    .loc[data.reset_index()["instrument"].unique()[:100]]
    .swaplevel()
    .sort_index()
)

# 计算收益率
data["$return"] = data.groupby(level=0)["$close"].pct_change().fillna(0)
print(data)
data.to_hdf("./daily_pv_debug.h5", key="data")