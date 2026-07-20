universe = 'NA'
start_date = 'NA'
end_date = 'NA'
train_start_date = 'NA'
train_end_date = 'NA'
test_start_date = 'NA'
test_end_date = 'NA'
forward_period = None
holding_period = None


def config_path(prefix_path: str, suffix_path: str) -> str:
    # Experiment result dirs are tagged with the TRAIN/TEST window bounds so the
    # directory matches the baseline param_dir convention:
    #   experiments/{universe}_{train_start}_{test_end}_forward-{fp}_holding-{hp}/...
    # Data splits (prefix="data") stay keyed on the FULL data span
    # (start_date/end_date) so a train/test window change does not invalidate
    # cached raw/cache data. forward/holding are None until step1_load_data sets.
    if prefix_path == "experiments":
        tag = f"{universe}_{train_start_date}_{test_end_date}"
    else:
        tag = f"{universe}_{start_date}_{end_date}"
    if prefix_path == "experiments" and forward_period is not None and holding_period is not None:
        tag += f"_forward-{forward_period}_holding-{holding_period}"
    return f"{prefix_path}/{tag}/{suffix_path}"
