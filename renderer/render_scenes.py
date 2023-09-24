import blenderproc as bproc
import bpy

import argparse
import os
import numpy as np
import sys

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

from renderer import object_loader
from config import TEXTURES_PATH, DATASETS_PATH

parser = argparse.ArgumentParser()
parser.add_argument(
    "train_dataset", nargs="?", help="Name of the target dataset", default="ycbv"
)
parser.add_argument(
    "aug_dataset",
    nargs="?",
    help="Name of the augumentation dataset. Can be multiple datasets, devided by comma, without space.",
    default=None,
)
parser.add_argument(
    "--num-scenes",
    type=int,
    default=4,
    help="How many scenes to generate",
)
parser.add_argument(
    "--imgs-per-scene",
    type=int,
    default=25,
    help="How many viewpoints per scene to generate",
)
args = parser.parse_args()

bproc.init()

# load target objects
target_objs = object_loader.load_objs(
    dataset_name=args.train_dataset, object_model_unit="mm"
)

# load distractor objects
dist_objs = []
if args.aug_dataset != None and "," in args.aug_dataset:
    augs = args.aug_dataset.split(",")
    for aug in augs:
        dist_objs += object_loader.load_objs(
            dataset_name=aug, object_model_unit="mm"
        )
elif args.aug_dataset != None:
    dist_objs = object_loader.load_objs(
        dataset_name=args.aug_dataset, object_model_unit="mm"
    )

# load datset intrinsics
object_loader.load_intrinsics()

# set shading and hide objects
for obj in target_objs + dist_objs:
    obj.set_shading_mode("auto")
    obj.hide(True)

# create room
room_planes = [
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1]),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]
    ),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]
    ),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]
    ),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0]
    ),
]
for plane in room_planes:
    plane.enable_rigidbody(
        False,
        collision_shape="BOX",
        mass=1.0,
        friction=100.0,
        linear_damping=0.99,
        angular_damping=0.99,
    )

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive(
    "PLANE", scale=[3, 3, 1], location=[0, 0, 10]
)
light_plane.set_name("light_plane")
light_plane_material = bproc.material.create("light_material")

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(TEXTURES_PATH)


# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

for i in range(args.num_scenes):
    # Sample objects for a scene
    sampled_target_objs = list(
        np.random.choice(
            target_objs, size=np.random.randint(1, len(target_objs) + 1), replace=False
        )
    )
    num_distr_samples = np.random.randint(1, 5)
    if len(dist_objs) < num_distr_samples:
        num_distr_samples = len(dist_objs)
    sampled_distractor_objs = list(
        np.random.choice(dist_objs, size=num_distr_samples, replace=False)
    )

    # Randomize materials and set physics
    for obj in sampled_target_objs + sampled_distractor_objs:
        mat = obj.get_materials()[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        obj.enable_rigidbody(
            True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99
        )
        obj.hide(False)

    # Sample two light sources
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
    )
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    location = bproc.sampler.shell(
        center=[0, 0, 0],
        radius_min=1,
        radius_max=1.5,
        elevation_min=5,
        elevation_max=89,
    )
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # Sample object poses and check collisions
    bproc.object.sample_poses(
        objects_to_sample=sampled_target_objs + sampled_distractor_objs,
        sample_pose_func=sample_pose_func,
        max_tries=1000,
    )

    # Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=3,
        max_simulation_time=10,
        check_object_interval=1,
        substeps_per_frame=20,
        solver_iters=25,
    )

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(
        sampled_target_objs + sampled_distractor_objs
    )

    cam_poses = 0
    while cam_poses < args.imgs_per_scene:
        # Sample location
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=0.5,
            radius_max=1.5,
            elevation_min=5,
            elevation_max=89,
        )
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(
            np.random.choice(
                sampled_target_objs + sampled_distractor_objs,
                size=1 + len(sampled_target_objs + sampled_distractor_objs) // np.random.randint(2,5),
                replace=False,
            )
        )
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159)
        )
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(
            location, rotation_matrix
        )

        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(
            cam2world_matrix, {"min": 0.3}, bop_bvh_tree
        ):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data in format
    bproc.writer.write_bop(
        DATASETS_PATH,
        target_objects=sampled_target_objs,
        dataset=args.train_dataset,
        depth_scale=0.1,
        depths=data["depth"],
        colors=data["colors"],
        color_file_format="JPEG",
        ignore_dist_thres=10,
    )

    for obj in sampled_target_objs + sampled_distractor_objs:
        obj.disable_rigidbody()
        obj.hide(True)
    
    print(f"Rendering of scene {i+1}/{args.num_scenes} finished.")
