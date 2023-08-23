"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# Standard Library
from typing import List

# MegaPose
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.renderer.types import Panda3dCameraData, Panda3dObjectData


def camera_to_panda3d(camera_data: CameraData) -> Panda3dCameraData:
    assert camera_data.TWC is not None
    assert camera_data.K is not None
    assert camera_data.resolution is not None
    panda3d_camera_data = Panda3dCameraData(TWC=camera_data.TWC,K=camera_data.K,resolution=camera_data.resolution,)
    return panda3d_camera_data

def object_to_panda3d(object_data: ObjectData) -> List[Panda3dObjectData]:
    assert object_data.TWO is not None
    panda3d_object_data = Panda3dObjectData(label=object_data.label,TWO=object_data.TWO,)
    return panda3d_object_data