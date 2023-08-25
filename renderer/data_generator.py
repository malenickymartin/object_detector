import numpy as np
from scipy.spatial.transform import Rotation as R
import panda3d as p3d
from happypose.toolbox.renderer import Panda3dLightData
import panda3d as p3d
from functools import partial


def generate_object_transform(object_name: str, camera: dict) -> dict:
    a = np.random.uniform(0, camera["resolution"][1])
    b = np.random.uniform(0, camera["resolution"][0])
    z = np.random.uniform(0.2, 2)

    cx = camera["K"][0][2]
    cy = camera["K"][1][2]
    f = camera["K"][0][0]

    x = (a * z - cx * z) / f
    y = (b * z - cy * z) / f
    xyz = np.array([x, y, z])
    xyz = np.around(xyz, decimals=3)

    quat = R.random().as_quat()

    transform = {"label": object_name, "TWO": [quat.tolist(), xyz.tolist()]}
    return transform


def generate_camera(res: list[int]) -> dict:
    fl = np.random.normal(700, 50)
    fl = np.clip(fl, 300, 900)
    cam = {
        "K": [[fl, 0.0, res[1] / 2], [0.0, fl, res[0] / 2], [0.0, 0.0, 1.0]],
        "resolution": res,
    }
    return cam


def generate_lights() -> list[Panda3dLightData]:
    def pos_fn(
        root_node: p3d.core.NodePath, light_node: p3d.core.NodePath, pos: np.ndarray
    ) -> None:
        radius = root_node.getBounds().radius
        xyz_ = pos * radius * 10
        light_node.setPos(tuple(xyz_.tolist()))
        return

    rand_amb_lgt = np.random.normal(0.7, 0.2)
    rand_amb_col = np.random.normal(0, 0.05, 3)
    rand_amb_col = rand_amb_lgt * (1 + rand_amb_col)
    rand_amb_col = np.clip(rand_amb_col, 0, 1.5)
    rand_amb_col = tuple(np.append(rand_amb_col, 1))

    rand_point_lgt = np.random.normal(0.15, 0.1)
    rand_point_col = np.random.normal(0, 0.05, 3)
    rand_point_col = rand_point_lgt * (1 + rand_point_col)
    rand_point_col = np.clip(rand_point_col, 0, 1.5)
    rand_point_col = tuple(np.append(rand_point_col, 1))

    rand_pos = np.random.uniform(-2, 2, size=3)

    pos_fn_ = partial(pos_fn, pos=rand_pos)

    light_datas = [Panda3dLightData(light_type="ambient", color=rand_amb_col)]
    light_datas.append(
        Panda3dLightData(
            light_type="point",
            color=(rand_point_col),
            positioning_function=pos_fn_,
        ),
    )
    return light_datas


if __name__ == "__main__":
    object_name = "barbecue-sauce"
    camera = generate_camera([480, 640])
    generate_object_transform(object_name, camera)
    generate_lights()
