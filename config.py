import os
from pathlib import Path

PROJECT_PATH = Path(os.path.realpath(__file__)).parent
OBJECTS_PATH = PROJECT_PATH / "objects"
BACKGROUNDS_PATH = PROJECT_PATH / "backgrounds"

def OBJECT_PATH(object_name):
    return OBJECTS_PATH / object_name

def MODEL_PATH(object_name):
    return OBJECT_PATH(object_name) / "model"

def RENDERS_PATH(object_name):
    return OBJECT_PATH(object_name) / "renders"

def MASKS_PATH(object_name):
    return OBJECT_PATH(object_name) / "masks"

def MESH_PATH(object_name):
    return OBJECT_PATH(object_name) / "meshes"