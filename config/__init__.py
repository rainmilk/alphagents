universe = 'NA'
start_date = 'NA'
end_date = 'NA'


def config_path(prefix_path: str, suffix_path: str) -> str:
    return f"{prefix_path}/{universe}_{start_date}_{end_date}/{suffix_path}"
