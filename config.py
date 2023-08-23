import os
from pathlib import Path

PROJECT_PATH = Path(os.path.realpath(__file__)).parent
OBJECTS_PATH = PROJECT_PATH / "datasets"
BACKGROUNDS_PATH = PROJECT_PATH / "backgrounds"
RESULTS_PATH = PROJECT_PATH / "results"

def DATASET_PATH(dataset_name : str) -> Path:
    return OBJECTS_PATH / dataset_name

def RESULT_PATH(model_folder : str) -> Path:
    return RESULTS_PATH / model_folder

def OBJECT_PATH(dataset_name : str, object_name : str) -> Path:
    return DATASET_PATH(dataset_name) / object_name

def RENDERS_PATH(dataset_name : str, object_name : str) -> Path:
    return OBJECT_PATH(dataset_name, object_name) / "renders"

def MASKS_PATH(dataset_name : str, object_name : str) -> Path:
    return OBJECT_PATH(dataset_name, object_name) / "masks"

def MESH_PATH(dataset_name : str, object_name : str) -> Path:
    return OBJECT_PATH(dataset_name, object_name) / "meshes"
