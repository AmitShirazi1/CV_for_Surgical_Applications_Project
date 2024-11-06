import blenderproc as bproc
import numpy as np
import os
import random
import json
import argparse
import debugpy
from colorsys import hsv_to_rgb
import bpy


# TODO: Ask GPT about:
    #   Where is the use of the unlabeled data (vids_tune/ and vids_test/)?
    #   Are you considering that every hdri is in a folder (/datashare/project/haven/hdris/abandoned_bakery/abandoned_bakery_2k.hdr)?
    #   What do you mean by "Apply articulation to instruments"?
    #   Suggest and implement more variations to the synthetic data generation process.

"""
To run file:
blenderproc run /home/student/project/data_generation/synthetic_data_generator.py -b

To debug file:
blenderproc run /home/student/project/data_generation/synthetic_data_generator.py -b -d
"""


# Set the path for 3D models, background images, and HDRI files
resources_dir = "/datashare/project/"
models_dir = os.path.join(resources_dir, "surgical_tools_models/")
needle_holder_model_dir = os.path.join(models_dir, "needle_holder/")
tweezers_model_dir = os.path.join(models_dir, "tweezers/")
coco_dir = os.path.join(resources_dir, "train2017/")
hdri_dir = os.path.join(resources_dir, "haven/hdris/")
output_dir = "./output/"
os.makedirs(output_dir, exist_ok=True)


# Loading the 3D models of surgical instruments
def load_all_instruments():
    extract_objs = lambda dir: [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.obj')]
    load_objs = lambda path: [bproc.loader.load_obj(obj)[0] for obj in extract_objs(path)]
    return load_objs(needle_holder_model_dir), load_objs(tweezers_model_dir)


def load_instruments():
    def choose_obj_file(dir):
        return random.choice([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.obj')])
    nh_rand, t_rand = choose_obj_file(needle_holder_model_dir), choose_obj_file(tweezers_model_dir)
    def load_obj(path):
        return bproc.loader.load_obj(path)[0]
    chosen_tool = random.random() > 0.5
    if chosen_tool:
        needle_holder_obj = load_obj(nh_rand)
        tweezers_obj = load_obj(t_rand) if random.random() > 0.2 else None
    else:
        needle_holder_obj = load_obj(nh_rand) if random.random() > 0.2 else None
        tweezers_obj = load_obj(t_rand)
    return needle_holder_obj, tweezers_obj


def place_instruments(obj, c):
    # Randomly place instruments
    obj.set_cp("category_id", c)
    for mat in obj.get_materials():
        # Check if the material has a Principled BSDF node
        has_principled_bsdf = False
        for node in mat.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                has_principled_bsdf = True
                break  # We found the Principled BSDF node, no need to check further
        if not has_principled_bsdf:
            print(f"Material {mat.get_name()} does not have a Principled BSDF node")
            continue

        if mat.get_name().lower() == "gold":
            try:
                random_gold_hsv_color = np.random.uniform([0.03, 0.95, 48], [0.25, 1.0, 48])
                random_gold_color = list(hsv_to_rgb(*random_gold_hsv_color)) + [1.0] # add alpha
                mat.set_principled_shader_value("Base Color", random_gold_color)
            except AttributeError:
                print(f"Error setting Base Color for material {mat.get_name()}")
        try:
            mat.set_principled_shader_value("Specular IOR Level", random.uniform(0, 1))
            mat.set_principled_shader_value("Roughness", random.uniform(0, 1))
            mat.set_principled_shader_value("Metallic", 1)
            mat.set_principled_shader_value("Roughness", 0.2)
        except AttributeError:
            print(f"Error setting shader value for material {mat.get_name()}")

    obj.set_location(np.random.uniform([-1, -1, 0], [1, 1, 1]))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [2*np.pi, 2*np.pi, 2*np.pi]))
    return obj


def set_lights():
    # Randomize lighting conditions
    num_lights = random.randint(1, 3)
    for i in range(num_lights):
        light = bproc.types.Light()
        light_types = ["POINT", "SUN", "SPOT", "AREA"]
        light.set_type(random.choice(light_types))
        light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]))
        if num_lights == 1:  # Only one main light
            lower, upper = 500, 1500
        elif i == 0:  # Main light, when there are others.
            lower, upper = 1000, 1500
        else:  # Secondary lights.
            lower, upper = 200, 750
        light.set_energy(random.uniform(lower, upper))
        light.set_location(np.random.uniform([-5, -5, 5], [5, 5, 10]))


# Load the camera parameters from json file
def load_camera_parameters(json_path):
    with open(json_path, 'r') as f:
        camera_params = json.load(f)
    return camera_params


def initial_camera_setup():
    # Load camera parameters from camera.json
    camera_params = load_camera_parameters(os.path.join(resources_dir, "camera.json"))

    # Extract the intrinsic camera parameters
    fx, fy, cx, cy = camera_params["fx"], camera_params["fy"], camera_params["cx"], camera_params["cy"]
    image_width, image_height = camera_params["width"], camera_params["height"]

    # Set up the camera intrinsics using fx, fy, cx, cy (focal lengths and principal point)
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    bproc.camera.set_intrinsics_from_K_matrix(K, image_width, image_height)


def choose_background():
    # Randomly choose and set a background (COCO or HDRI)
    dir = random.choice([coco_dir, hdri_dir])
    img_path = os.path.join(dir, random.choice(os.listdir(dir)))
    bproc.world.set_world_background_hdr_img(img_path)
    # Set a random world lighting strength
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.1, 1.5)


def sampling_camera_position(objects, camera_tries, camera_successes):
    camera_tries += 1
    obj1_location = objects[0].get_location()
    center_location = obj1_location
    radius_min, radius_max = 2, 10
    if len(objects) >= 2:  # If there are two objects
        obj2_location = objects[1].get_location()
        # Calculate center and distance between objects
        center_location = (obj1_location + obj2_location) / 2
        distance_between_objects = np.linalg.norm(obj1_location - obj2_location)
        # Set radius based on object distance
        radius_min = distance_between_objects * 0.75
        radius_max = distance_between_objects * 1.5

    # Sample random camera location around the object
    location = bproc.sampler.shell(
        center = center_location,
        radius_min=radius_min,
        radius_max=radius_max,
        elevation_min=-90,
        elevation_max=90
    )
    # Compute rotation based lookat point which is placed randomly around the objects
    poi = center_location + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # TODO: DELETE Add homog cam pose based on location an rotation
    # Add homog cam pose based on location and rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    # Only add camera pose if object is still visible
    if (objects[0] in bproc.camera.visible_objects(cam2world_matrix))\
    or ((len(objects) >= 2) and (objects[1] in bproc.camera.visible_objects(cam2world_matrix))):
        bproc.camera.add_camera_pose(cam2world_matrix, frame=camera_successes)
        camera_successes += 1
    else: # If the object is not visible
        sampling_camera_position(objects, camera_tries, camera_successes)
    return camera_tries, camera_successes


def further_complicate_image(objects):
    # Additional variations:
    # Add some random camera effects like blur or noise
    if random.random() > 0.5:
        # bproc.camera.add_motion_blur()
        bproc.renderer.enable_motion_blur()
    if random.random() > 0.5:
        bproc.camera.add_sensor_noise()


    # # Set the camera's random position and rotation
    bproc.camera.set_location(np.random.uniform([-2, -2, 2], [2, 2, 4]))
    bproc.camera.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi/4, np.pi/4, np.pi/4]))
    """
    # point of interest
    poi = bproc.object.compute_poi(objects)
    # Sample random camera location above objects
    location = np.random.uniform([-10, -10, 8], [10, 10, 12])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location and rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)
    """


def main(args):
    if args.debug:
        # Debugging if specified
        debugpy.listen(5678)
        debugpy.wait_for_client()

    # Set up scene: background, lighting, and instruments
    camera_tries, camera_successes = 0, 0
    while (camera_tries < 10000) and (camera_successes < args.num_images):  # Generate 1000 images
        # Clear all key frames from the previous run
        bproc.utility.reset_keyframes()
        # Initialize BlenderProc
        bproc.init()

        # Load the surgical instruments
        needle_holder_obj, tweezers_obj = load_instruments()

        objects = list()
        c = 0
        def append_instruments(insts):
            nonlocal c
            for inst in insts:
                if inst:
                    c += 1
                    objects.append(place_instruments(inst, c))
        append_instruments([needle_holder_obj, tweezers_obj])
        
        set_lights()
        initial_camera_setup()
        choose_background()
        sampling_camera_position(objects, camera_tries, camera_successes)
        # further_complicate_image(objects)

        bproc.renderer.set_max_amount_of_samples(100) # to speed up rendering, reduce the number of samples
        # Disable transparency so the background becomes transparent
        bproc.renderer.set_output_format(enable_transparency=False)
        # add segmentation masks (per class and per instance)
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

        """
        # activate normal and depth rendering
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_normals_output()
        bproc.renderer.set_noise_threshold(0.01)
        """

        # Render the image and segmentation mask
        data = bproc.renderer.render()

        # Write data to coco file
        bproc.writer.write_coco_annotations(os.path.join(output_dir, 'coco_format'),
                                            instance_segmaps=data["instance_segmaps"],
                                            instance_attribute_maps=data["instance_attribute_maps"],
                                            colors=data["colors"],
                                            mask_encoding_format="polygon",
                                            append_to_existing_output=True)

        # Save images and masks
        bproc.writer.write_hdf5(os.path.join(output_dir, 'hdf5_format'), data, append_to_existing_output=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--blender', action='store_true', help="Run in Blender")
    parser.add_argument('-n', '--num_images', type=int, default=1000, help="Number of images to generate")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debugging")
    main(parser.parse_args())
