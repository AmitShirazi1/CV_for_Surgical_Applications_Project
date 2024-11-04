import blenderproc as bproc
import numpy as np
import os
import random
import json
from time import time


# TODO: Ask GPT about:
    #   Where is the use of the unlabeled data (vids_tune/ and vids_test/)?
    #   Are you considering that every hdri is in a folder (/datashare/project/haven/hdris/abandoned_bakery/abandoned_bakery_2k.hdr)?
    #   What do you mean by "Apply articulation to instruments"?
    #   Suggest and implement more variations to the synthetic data generation process.

"""
To run file:    blenderproc run /home/student/project/synthetic_data_generator.py -b
"""


# Set the path for 3D models, background images, and HDRI files
resources_dir = "/datashare/project/"
models_dir = os.path.join(resources_dir, "surgical_tools_models")
needle_holder_model_dir = os.path.join(models_dir, "needle_holder")
tweezers_model_dir = os.path.join(models_dir, "tweezers")
background_dir = os.path.join(resources_dir, "train2017")
hdri_dir = os.path.join(resources_dir, "haven/hdris")
output_dir = "./output/"
os.makedirs(output_dir, exist_ok=True)


# Loading the 3D models of surgical instruments
def load_instruments():
    extract_objs = lambda dir: [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.obj')]
    load_objs = lambda path: [bproc.loader.load_obj(obj)[0] for obj in extract_objs(path)]
    return load_objs(needle_holder_model_dir), load_objs(tweezers_model_dir)


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
    # Randomly choose a background (COCO or HDRI)
    if random.random() > 0.5:
        background_image = os.path.join(background_dir, random.choice(os.listdir(background_dir)))
        bproc.world.set_world_background_hdr_img(background_image)  # Notice here
    else:
        hdri_image = os.path.join(hdri_dir, random.choice(os.listdir(hdri_dir)))
        bproc.world.set_world_background_hdr_img(hdri_image)


def set_lights():
    # Randomize lighting conditions
    num_lights = random.randint(1, 3)
    for i in range(num_lights):
        light = bproc.types.Light()
        light_types = ["POINT", "SUN", "SPOT", "AREA"]
        light.set_type(random.choice(light_types))
        light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]))
        if num_lights == 1:
            lower, upper = 500, 1500
        elif i == 0:
            lower, upper = 1000, 1500
        else:
            lower, upper = 200, 750
        light.set_energy(random.uniform(lower, upper))
        light.set_location(np.random.uniform([-5, -5, 5], [5, 5, 10]))


def place_instruments(needle_holder_objs, tweezers_objs):
    # Randomly place instruments
    needle_holder, tweezers = random.sample(needle_holder_objs, 1)[0], random.sample(tweezers_objs, 1)[0]
    needle_holder.set_location(np.random.uniform([-1, -1, 0], [1, 1, 1]))
    needle_holder.set_rotation_euler(np.random.uniform([0, 0, 0], [2*np.pi, 2*np.pi, 2*np.pi]))
    tweezers.set_location(np.random.uniform([-1, -1, 0], [1, 1, 1]))
    tweezers.set_rotation_euler(np.random.uniform([0, 0, 0], [2*np.pi, 2*np.pi, 2*np.pi]))
    return [needle_holder, tweezers]


def further_complicate_image(objects):
    # Additional variations:
    # Add some random camera effects like blur or noise
    # if random.random() > 0.5:
    #     # bproc.camera.add_motion_blur()
    #     bproc.renderer.enable_motion_blur()
    # if random.random() > 0.5:
    #     bproc.camera.add_sensor_noise()


    # # Set the camera's random position and rotation
    # bproc.camera.set_location(np.random.uniform([-2, -2, 2], [2, 2, 4]))
    # bproc.camera.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi/4, np.pi/4, np.pi/4]))
    
    # point of interest
    poi = bproc.object.compute_poi(objects)
    # Sample random camera location above objects
    location = np.random.uniform([-10, -10, 8], [10, 10, 12])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location and rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


def main():
    # Initialize BlenderProc
    bproc.init()

    # Load the surgical instruments
    needle_holder_objs, tweezers_objs = load_instruments()

    initial_camera_setup()

    # activate normal and depth rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()
    bproc.renderer.set_noise_threshold(0.01)

    # Set up scene: background, lighting, and instruments
    for i in range(1000):  # Generate 1000 images
        # Clear all key frames from the previous run
        bproc.utility.reset_keyframes()
        
        choose_background()
        set_lights()
        objects = place_instruments(needle_holder_objs, tweezers_objs)
        further_complicate_image(objects)

        

        # Render the image and segmentation mask
        data = bproc.renderer.render()

        # Save images and masks
        bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=True)


if __name__ == "__main__":
    main()
