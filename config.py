import yaml
file_path = "config.yaml"

def read_config():
    """
    Reads the config from the config.yaml file and loads it to the application

    Returns:
        config: Configuration as key value pair (Dict)
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config
