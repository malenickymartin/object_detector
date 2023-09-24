import os
from random import choice
import numpy as np
from typing import List, Optional, Tuple

from mathutils import Vector

from config import MESH_PATH

from blenderproc.python.camera import CameraUtility
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.loader.ObjectLoader import load_obj


def load_objs(
    dataset_name: str,
    sample_objects: bool = False,
    num_of_objs_to_sample: Optional[int] = None,
    obj_instances_limit: int = -1,
    object_model_unit: str = "m",
    move_origin_to_x_y_plane: bool = False,
) -> List[MeshObject]:
    """Loads all or a subset of 3D models of any dataset

    :param dataset_name: Name od dataset dataset.
    :param sample_objects: Toggles object sampling from the specified dataset.
    :param num_of_objs_to_sample: Amount of objects to sample from the specified dataset. If this amount is bigger
                                  than the dataset actually contains, then all objects will be loaded.
    :param obj_instances_limit: Limits the amount of object copies when sampling. Default: -1 (no limit).
    :param object_model_unit: The unit the object model is in. Object model will be scaled to meters. This does not
                              affect the annotation units. Available: ['m', 'dm', 'cm', 'mm'].
    :param move_origin_to_x_y_plane: Move center of the object to the lower side of the object, this will not work
                                     when used in combination with pose estimation tasks! This is designed for the
                                     use-case where objects are used as filler objects in the background.
    :return: The list of loaded mesh objects.
    """

    mesh_path = MESH_PATH(dataset_name)
    num_objs = 0
    is_ply = []
    for f in os.listdir(mesh_path):
        if f.endswith(".ply"):
            is_ply.append(True)
            num_objs += 1
        elif f.endswith(".obj"):
            num_objs += 1
            is_ply.append(False)

    model_p = {
        # ID's of all objects included in the dataset.
        "obj_ids": list(range(1, num_objs + 1)),
        # ID's of objects with symmetries.
        "symmetric_obj_ids": [],
        # Path template to an object model file.
        "model_tpath": os.path.join(mesh_path, "obj_{obj_id:06d}."),
        # Path to a file with meta information about the object models.
        "models_info_path": os.path.join(mesh_path, "models_info.json"),
        "is_ply": is_ply
    }

    assert object_model_unit in ["m", "dm", "cm", "mm"], (
        f"Invalid object model unit: `{object_model_unit}`. "
        f"Supported are 'm', 'dm', 'cm', 'mm'"
    )
    scale = {"m": 1.0, "dm": 0.1, "cm": 0.01, "mm": 0.001}[object_model_unit]

    obj_ids = model_p["obj_ids"]

    loaded_objects = []
    # if sampling is enabled
    if sample_objects:
        loaded_ids = {}
        loaded_amount = 0
        if (
            obj_instances_limit != -1
            and len(obj_ids) * obj_instances_limit < num_of_objs_to_sample
        ):
            raise RuntimeError(
                f"{dataset_name}'s contains {len(obj_ids)} objects, {num_of_objs_to_sample} object "
                f"where requested to sample with an instances limit of {obj_instances_limit}. Raise "
                f"the limit amount or decrease the requested amount of objects."
            )
        while loaded_amount != num_of_objs_to_sample:
            random_id = choice(obj_ids)
            if random_id not in loaded_ids:
                loaded_ids.update({random_id: 0})
            # if there is no limit or if there is one, but it is not reached for this particular object
            if obj_instances_limit == -1 or loaded_ids[random_id] < obj_instances_limit:
                cur_obj = _DatasetLoader.load_mesh(
                    random_id, model_p, dataset_name, scale
                )
                loaded_ids[random_id] += 1
                loaded_amount += 1
                loaded_objects.append(cur_obj)
            else:
                print(
                    f"ID {random_id} was loaded {loaded_ids[random_id]} times with limit of {obj_instances_limit}. "
                    f"Total loaded amount {loaded_amount} while {num_of_objs_to_sample} are being requested"
                )
    else:
        for obj_id in obj_ids:
            cur_obj = _DatasetLoader.load_mesh(obj_id, model_p, dataset_name, scale)
            loaded_objects.append(cur_obj)
    # move the origin of the object to the world origin and on top of the X-Y plane
    # makes it easier to place them later on, this does not change the `.location`
    # This is only useful if the objects are not used in a pose estimation scenario.
    if move_origin_to_x_y_plane:
        for obj in loaded_objects:
            obj.move_origin_to_bottom_mean_point()

    return loaded_objects


def load_intrinsics() -> Tuple[np.ndarray, int, int]:
    """
    Load and set the camera matrix and image resolution of a specified BOP dataset

    :param bop_dataset_path: Full path to a specific bop dataset e.g. /home/user/bop/tless.
    :param split: Optionally, train, test or val split depending on BOP dataset, defaults to "test"
    :param cam_type: Camera type. If not defined, dataset-specific default camera type is used.
    :returns: camera matrix K, W, H
    """

    res = (640, 480)

    fl = np.random.normal(700, 50)
    fl = np.clip(fl, 300, 900)

    K = np.array([[fl, 0.0, res[0] / 2], [0.0, fl, res[1] / 2], [0.0, 0.0, 1.0]])

    # set camera intrinsics
    CameraUtility.set_intrinsics_from_K_matrix(K, res[0], res[1])

    return K, res[0], res[1]


class _DatasetLoader:
    CACHED_OBJECTS = {}

    @staticmethod
    def load_mesh(
        obj_id: int, model_p: dict, dataset_name: str, scale: float = 1
    ) -> MeshObject:
        """Loads BOP mesh and sets category_id.

        :param obj_id: The obj_id of the Object.
        :param model_p: model parameters.
        :param bop_dataset_name: The name of the used dataset.
        :param scale: factor to transform set pose in mm or meters.
        :return: Loaded mesh object.
        """

        model_path = model_p["model_tpath"].format(**{"obj_id": obj_id})
        if model_p["is_ply"][obj_id-1]:
            model_path += "ply"
        else:
            model_path += "obj"

        # if the object was not previously loaded - load it, if duplication is allowed - duplicate it
        duplicated = model_path in _DatasetLoader.CACHED_OBJECTS
        objs = load_obj(model_path, cached_objects=_DatasetLoader.CACHED_OBJECTS)
        assert (
            len(objs) == 1
        ), f"Loading object from '{model_path}' returned more than one mesh"
        cur_obj = objs[0]

        if duplicated:
            # See issue https://github.com/DLR-RM/BlenderProc/issues/590
            for i, material in enumerate(cur_obj.get_materials()):
                material_dup = material.duplicate()
                cur_obj.set_material(i, material_dup)

        """# Change Material name to be backward compatible
        cur_obj.get_materials()[-1].set_name(
            "bop_" + dataset_name + "_vertex_col_material"
        )"""
        cur_obj.set_scale(Vector((scale, scale, scale)))
        cur_obj.set_cp("category_id", obj_id)
        cur_obj.set_cp("model_path", model_path)
        cur_obj.set_cp("is_bop_object", True)
        cur_obj.set_cp("bop_dataset_name", dataset_name)

        return cur_obj
