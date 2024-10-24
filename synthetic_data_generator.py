import blenderproc as bproc
import numpy as np
import os
import random
import json

# TODO: Ask GPT about:
    #   Where is the use of the unlabeled data (vids_tune/ and vids_test/)?
    #   Are you considering that every hdri is in a folder (/datashare/project/haven/hdris/abandoned_bakery/abandoned_bakery_2k.hdr)?
    #   What do you mean by "Apply articulation to instruments"?
    #   Suggest and implement more variations to the synthetic data generation process.


# Set the path for 3D models, background images, and HDRI files
resources_dir = "/datashare/project/"
models_dir = os.path.join(resources_dir, "surgical_tools_models")
needle_holder_model_dir = os.path.join(models_dir, "needle_holder")
tweezers_model_dir = os.path.join(models_dir, "tweezers")
background_dir = os.path.join(resources_dir, "train2017")
hdri_dir = os.path.join(resources_dir, "haven/hdris")

# Loading the 3D models of surgical instruments
def load_instruments():
    extract_obgs = lambda dir: [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.obj')]
    instruments_paths = extract_obgs(needle_holder_model_dir) + extract_obgs(tweezers_model_dir)
    return [bproc.loader.load_obj(instrument)[0] for instrument in instruments_paths]

# Load the camera parameters from json file
def load_camera_parameters(json_path):
    with open(json_path, 'r') as f:
        camera_params = json.load(f)
    return camera_params

# Initialize BlenderProc
bproc.init()

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

# Load the surgical instruments
instruments = load_instruments()

# Set up scene: background, lighting, and instruments
for i in range(1000):  # Generate 1000 images
    # bproc.scene.clear()

    # Randomly choose a background (COCO or HDRI)
    if random.random() > 0.5:
        background_image = os.path.join(background_dir, random.choice(os.listdir(background_dir)))
        bproc.world.set_world_background_hdr_img(background_image)  # Notice here
    else:
        hdri_image = os.path.join(hdri_dir, random.choice(os.listdir(hdri_dir)))
        bproc.world.set_world_background_hdr_img(hdri_image)

    # Randomize lighting conditions
    light = bproc.types.Light()
    light_types = ["POINT", "SUN", "SPOT", "AREA"]
    light.set_type(random.choice(light_types))
    light.set_color(np.random.uniform([0.7, 0.7, 0.7], [1.0, 1.0, 1.0]))
    light.set_energy(random.uniform(50, 1000))
    light.set_location(np.random.uniform([-5, -5, 5], [5, 5, 10]))

    # Randomly place instruments
    for instrument in instruments:
        instrument.set_location(np.random.uniform([-1, -1, 0], [1, 1, 1]))
        instrument.set_rotation_euler(np.random.uniform([0, 0, 0], [2*np.pi, 2*np.pi, 2*np.pi]))

    # Additional variations:
    # Apply articulation to instruments
    for instrument in instruments:
        articulation_value = random.uniform(0, 1)
        # Assuming the articulation state can be set via custom method (depends on how models are rigged)
        instrument.set_articulation_state(articulation_value)

    # Add some random camera effects like blur or noise
    if random.random() > 0.5:
        bproc.camera.add_motion_blur()
    if random.random() > 0.5:
        bproc.camera.add_sensor_noise()

    # Set the camera's random position and rotation
    bproc.camera.set_location(np.random.uniform([-2, -2, 2], [2, 2, 4]))
    bproc.camera.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi/4, np.pi/4, np.pi/4]))

    # Render the image and segmentation mask
    data = bproc.renderer.render()

    # Save images and masks
    output_dir = "./output/synthetic_images/"
    os.makedirs(output_dir, exist_ok=True)
    bproc.writer.write_hdf5(output_dir, data["colors"], data["segmentation"])

print("Synthetic data generation completed.")
