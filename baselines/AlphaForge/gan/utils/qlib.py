
from alphagen_qlib.stock_data import StockData

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


def get_data_my(instru,start,end,raw=False,qlib_path = '',freq = 'day'):
    _mase_offline_guard("baselines/AlphaForge/gan/utils/qlib.py::get_data_my (qlib.init)")
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=qlib_path, region='cn')
    def get_instruments(name,start,end):
        instru =  D.instruments(name)
        return D.list_instruments(
            instruments=instru, 
            start_time=start, 
            end_time=end, 
            as_list=True,
            freq=freq,
            
        )
    instru = get_instruments(instru,start,end)
    return StockData(instru,start,end,raw = raw,qlib_path = qlib_path,freq = freq)