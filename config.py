import os
from pathlib import Path

# This is project directory. It is by default set to this config file path.
PROJECT_PATH = Path(os.path.realpath(__file__)).parent

# This directory contains subdirectory for each used dataset. More in "DATASET_PATH".
DATASETS_PATH = PROJECT_PATH / "datasets"
# This directory contains arbitraray images, that whill be used to create background augumentation.
BACKGROUNDS_PATH = PROJECT_PATH / "backgrounds"
# This directory contains subdirectories for each dataset and augumentation dataset used for training. 
# All subdirectories will be created automatically by "run_detector_training.py". More in "MODEL_PATH"
MODELS_PATH = PROJECT_PATH / "models"

# This directory contains subdirectory for every object in dataset. More in "OBJECT_PATH"
def DATASET_PATH(dataset_name : str) -> Path:
    return DATASETS_PATH / dataset_name

# This directory contains results of training. After the training is finished, there will be "model.ckpt" file containing the trained model, 
# metrics.csv file containing losses for each training and validation epoch and "hparams.yaml", needed for running inference.
def MODEL_PATH(dir_name : str) -> Path:
    return MODELS_PATH / dir_name

# This directory contains "meshes", "renders" and "masks" directories for given object in dataset. 
# The "renders" and "masks" dirs are created by "render_model.py"
def OBJECT_PATH(dataset_name : str, object_name : str) -> Path:
    return DATASET_PATH(dataset_name) / object_name

# This directory contains large number of images, rendered by "render_model.py". Each image is named "render'x'.png", where x is render number. 
# This number corresponds to mask in "MASKS_PATH".
def RENDERS_PATH(dataset_name : str, object_name : str) -> Path:
    return OBJECT_PATH(dataset_name, object_name) / "renders"

# This directory contains large number of images, rendered by "render_model.py". Each image is named "mask'x'.png", where x is mask number. 
# This number corresponds to render in "RENDERS_PATH".
def MASKS_PATH(dataset_name : str, object_name : str) -> Path:
    return OBJECT_PATH(dataset_name, object_name) / "masks"

#This directory contains mesh file. This file is either ".obj" or ".ply".
# You can also use "OBJECT_PATH" to store your mesh
def MESH_PATH(dataset_name : str, object_name : str) -> Path:
    return OBJECT_PATH(dataset_name, object_name) / "meshes"
