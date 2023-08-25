import os
from pathlib import Path

PROJECT_PATH = Path(os.path.realpath(__file__)).parent
DATASETS_PATH = PROJECT_PATH / "datasets"
BACKGROUNDS_PATH = PROJECT_PATH / "backgrounds"
MODELS_PATH = PROJECT_PATH / "models"

def DATASET_PATH(dataset_name : str) -> Path:
    return DATASETS_PATH / dataset_name

def MODEL_PATH(dir_name : str) -> Path:
    return MODELS_PATH / dir_name

def OBJECT_PATH(dataset_name : str, object_name : str) -> Path:
    return DATASET_PATH(dataset_name) / object_name

def RENDERS_PATH(dataset_name : str, object_name : str) -> Path:
    return OBJECT_PATH(dataset_name, object_name) / "renders"

def MASKS_PATH(dataset_name : str, object_name : str) -> Path:
    return OBJECT_PATH(dataset_name, object_name) / "masks"

def MESH_PATH(dataset_name : str, object_name : str) -> Path:
    return OBJECT_PATH(dataset_name, object_name) / "meshes"
