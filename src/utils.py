import yaml
import os

# Setting local paths
root_path = os.environ.get("LOCAL_PATH")
conf_path = os.path.join(root_path, "conf", "")

def get_conf(path: str = conf_path, filename: str='conf.yaml'):
    filename_conf = os.path.join(path, filename)

    with open(filename_conf, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)

    return cfg