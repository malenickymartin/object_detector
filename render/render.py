# Standard Library
import argparse
from pathlib import Path

# Third Party
import numpy as np
import torch
from PIL import Image

# HappyPose
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData

from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.utils.logging import get_logger, set_logging_level

# Local
from render.data_generator import generate_object_transform, generate_camera, generate_lights
from render import conversion

from config import (
    RENDERS_PATH,
    MASKS_PATH,
    MESH_PATH
)

logger = get_logger(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def make_object_dataset(object_name: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dir = MESH_PATH(object_name)
    label = object_name
    mesh_path = None
    for fn in object_dir.glob("*"):
        if fn.suffix in {".obj", ".ply"}:
            assert not mesh_path, f"there multiple meshes in the {label} directory"
            mesh_path = fn
    assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
    rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

def make_output_visualization(object_name: Path, args) -> None:
    
    object_dataset = make_object_dataset(object_name)
    renderer = Panda3dSceneRenderer(object_dataset)
    
    for render_num in range(args.num_of_images):
        for attempt in range(10):

            camera = generate_camera(args.camera_res)
            object_transform = generate_object_transform(object_name, camera)
            
            camera = CameraData.from_json(camera)
            camera.TWC = Transform(np.eye(4))
            camera = conversion.camera_to_panda3d(camera)

            object_transform = ObjectData.from_json(object_transform)
            object_transform = conversion.object_to_panda3d(object_transform)

            light_datas = generate_lights()

            renderings = renderer.render_scene(
                [object_transform],
                [camera],
                light_datas,
                render_depth=True,
                render_binary_mask=True,
                render_normals=True,
                copy_arrays=True,
            )[0]
            
            if np.sum(renderings.binary_mask)/np.prod(camera.resolution) <= args.object_image_ratio:
                logger.info(f"Image number {render_num} not saved! Object took only {np.sum(renderings.binary_mask)/np.prod(camera.resolution)*100} % of frame.")
                assert attempt < 9, "Object was not visible (or was too small) on 10 renders in row. Consider changing constants in data_generator.py"
                continue

            render = Image.fromarray(renderings.rgb, "RGB")
            render.save(str(RENDERS_PATH(object_name) / f"render{render_num}.png"))
            mask = renderings.binary_mask.astype(np.uint8)*255
            mask = Image.fromarray(mask)
            mask.save(str(MASKS_PATH(object_name) / f"mask{render_num}.png"))

            logger.info(f"Wrote render number {render_num} to {str(RENDERS_PATH(object_name) / f'render{render_num}.png')}.")
            break
    return

if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("object_name",type=str, nargs='?', default="barbecue-sauce")
    parser.add_argument("num_of_images", type=int, nargs='?', default=2000)
    parser.add_argument("camera_res", type=list, nargs='?', default=[480, 640])
    parser.add_argument("object_image_ratio", type=int, nargs='?', default=0.001)
    args = parser.parse_args()

    object_name = args.object_name

    RENDERS_PATH(object_name).mkdir(exist_ok=True)
    MASKS_PATH(object_name).mkdir(exist_ok=True)

    make_output_visualization(object_name, args)
