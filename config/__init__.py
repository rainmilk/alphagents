universe = 'NA'
start_date = 'NA'
end_date = 'NA'
forward_period = None
holding_period = None


def config_path(prefix_path: str, suffix_path: str) -> str:
    tag = f"{universe}_{start_date}_{end_date}"
    # Only experiment result dirs carry the forward/holding tag, so MASE's
    # directory matches the baseline format
    #   experiments/{universe}_{start}_{end}_forward-{fp}_holding-{hp}/...
    # Data splits (prefix="data") are independent of backtest params, so they
    # stay tag-free. forward/holding are None until step1_load_data sets them.
    if prefix_path == "experiments" and forward_period is not None and holding_period is not None:
        tag += f"_forward-{forward_period}_holding-{holding_period}"
    return f"{prefix_path}/{tag}/{suffix_path}"
