from pathlib import Path
import yaml


#from configs import config_private



def load_config(name: str="config.yaml"):
    
    project_root = Path(__file__).parents[1]
    config_path = project_root / "configs" / name

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
cfg = load_config("config.yaml")
project_dir = cfg["project_data_dir"]
input_data_dir = project_dir + r"\input"
output_data_dir = project_dir + r"\output"
model_dir = project_dir + r"\models"

def get_dirs(config_file):
    cfg = load_config(config_file)
    project_dir = cfg["project_data_dir"]
    input_data_dir = project_dir + r"\input"
    output_data_dir = project_dir + r"\output"
    model_dir = project_dir + r"\models"
    return project_dir, input_data_dir, output_data_dir, model_dir
