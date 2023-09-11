# Standard Library
import argparse
import os
import numpy as np
from PIL import Image

# HappyPose
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.renderer.types import Panda3dCameraData, Panda3dObjectData

from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer

# Local
from renderer.data_generator import (
    generate_object_transform,
    generate_camera,
    generate_lights,
)

from config import RENDERS_PATH, MASKS_PATH, MESH_PATH, DATASET_PATH


def make_object_dataset(dataset_name: str, object_name: str) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dir = MESH_PATH(dataset_name, object_name)
    label = object_name
    mesh_path = None
    for fn in object_dir.glob("*"):
        if fn.suffix in {".obj", ".ply"}:
            assert not mesh_path, f"there multiple meshes in the {label} directory"
            mesh_path = fn
    assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
    rigid_objects.append(
        RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units)
    )
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_output_visualization(args: argparse.ArgumentParser, object_name: str) -> None:
    dataset_name = args.dataset_name
    object_dataset = make_object_dataset(dataset_name, object_name)
    renderer = Panda3dSceneRenderer(object_dataset)

    for render_num in range(args.num_of_images):
        for attempt in range(10):
            camera = generate_camera(args.camera_res)
            object_transform = generate_object_transform(object_name, camera)

            camera = CameraData.from_json(camera)
            camera.TWC = Transform(np.eye(4))
            camera = Panda3dCameraData(
                TWC=camera.TWC, K=camera.K, resolution=camera.resolution
            )

            object_transform = ObjectData.from_json(object_transform)
            object_transform = Panda3dObjectData(
                label=object_transform.label, TWO=object_transform.TWO
            )

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

            if (
                np.sum(renderings.binary_mask) / np.prod(camera.resolution)
                <= args.object_image_ratio
            ):
                print(
                    f"Image number {render_num} not saved! Object took only {np.sum(renderings.binary_mask)/np.prod(camera.resolution)*100} % of frame."
                )
                assert (
                    attempt < 9
                ), "Object was not visible (or was too small) on 10 renders in row. Consider changing distance from camera range in data_generator.py"
                continue

            render = Image.fromarray(renderings.rgb, "RGB")
            render.save(
                str(RENDERS_PATH(dataset_name, object_name) / f"render{render_num}.png")
            )
            mask = renderings.binary_mask.astype(np.uint8) * 255
            mask = Image.fromarray(mask)
            mask.save(
                str(MASKS_PATH(dataset_name, object_name) / f"mask{render_num}.png")
            )

            print(
                f"Wrote render number {render_num} to {str(RENDERS_PATH(dataset_name, object_name) / f'render{render_num}.png')}."
            )
            break
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, nargs="?", default="three-objects")
    parser.add_argument("num_of_images", type=int, nargs="?", default=5000)
    parser.add_argument("--object_name", type=str, default=None)
    parser.add_argument("--camera_res", type=list, default=[480, 640])
    parser.add_argument("--object_image_ratio", type=int, default=0.001)
    args = parser.parse_args()

    if args.object_name == None:
        object_names = sorted(os.listdir(DATASET_PATH(args.dataset_name)))
    else:
        object_names = [args.object_name]

    for object in object_names:
        RENDERS_PATH(args.dataset_name, object).mkdir(exist_ok=True)
        MASKS_PATH(args.dataset_name, object).mkdir(exist_ok=True)

        make_output_visualization(args, object)
