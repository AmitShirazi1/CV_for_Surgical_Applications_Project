import blenderproc as bproc
import numpy as np
import os
import random
import json
import argparse
import debugpy
from colorsys import hsv_to_rgb
import bpy
from skimage import measure
import h5py
from PIL import Image
import matplotlib.cm as cm

"""
To run file:
blenderproc run /home/student/project/data_generation/synthetic_data_generator.py -b

To debug file:
blenderproc run /home/student/project/data_generation/synthetic_data_generator.py -d

If BlenderProc does the f**ing symbol error:
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=/anaconda/envs/synth/lib:$LD_LIBRARY_PATH

If the debugger refuses because the port is taken:
lsof -i :5678
Take the PID
kill -9 <PID>
"""

# Set the path for 3D models, background images, and HDRI files
resources_dir = "/datashare/project/"

# Directory for surgical tools models
models_dir = os.path.join(resources_dir, "surgical_tools_models/")
needle_holder_model_dir = os.path.join(models_dir, "needle_holder/")
tweezers_model_dir = os.path.join(models_dir, "tweezers/")

# Directories for background images and HDRI files
coco_dir = os.path.join(resources_dir, "train2017/")
hdri_dir = os.path.join(resources_dir, "haven/hdris/")

# Directory for the current file
current_file_dir = os.path.dirname(os.path.abspath(__file__))

def create_path(path, dir_name):
    path = os.path.join(path, dir_name)
    os.makedirs(path, exist_ok=True)
    return path
# Output directory for generated data
output_dir = create_path(current_file_dir, "output_objects/")
output_hdri_dir = create_path(output_dir, 'hdri_background/')
output_coco_dir = create_path(output_dir, 'coco_background/')
no_background_images_dir = create_path(output_coco_dir, 'no_background/')
with_background_dir = create_path(output_coco_dir, 'with_background/')

# Loading the 3D models of surgical instruments
def load_all_instruments():
    """
    Loads all instrument models from specified directories.

    This function extracts and loads 3D object files (.obj) from two predefined directories:
    `needle_holder_model_dir` and `tweezers_model_dir`. It uses two lambda functions:
    - `extract_objs`: Extracts the file paths of all .obj files in a given directory.
    - `load_objs`: Loads the .obj files using the `bproc.loader.load_obj` method.

    Returns:
        tuple: A tuple containing two lists:
            - The first list contains the loaded objects from `needle_holder_model_dir`.
            - The second list contains the loaded objects from `tweezers_model_dir`.
    """
    extract_objs = lambda dir: [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.obj')]
    load_objs = lambda path: [bproc.loader.load_obj(obj)[0] for obj in extract_objs(path)]
    return load_objs(needle_holder_model_dir), load_objs(tweezers_model_dir)


def load_instruments():
    """
    Randomly loads 3D models of surgical instruments (needle holder and tweezers) from specified directories.

    The function randomly selects .obj files from the given directories for needle holders and tweezers.
    It then randomly decides which instrument to load and whether to load both or just one of them.

    Returns:
        tuple: A tuple containing two elements:
            - needle_holder_obj: The loaded needle holder object or None.
            - tweezers_obj: The loaded tweezers object or None.
    """
    # Randomly choose an .obj file from the given directory
    def choose_obj_file(dir):
        return random.choice([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.obj')])
    nh_rand, t_rand = choose_obj_file(needle_holder_model_dir), choose_obj_file(tweezers_model_dir)

    # Load the .obj file using the `bproc.loader.load_obj` method
    def load_obj(path):
        return bproc.loader.load_obj(path)[0]
    
    needle_holder_obj, tweezers_obj = load_obj(nh_rand), load_obj(t_rand)
    return needle_holder_obj, tweezers_obj


def place_instruments(obj, c):
    # Set the category ID for the object
    obj.set_cp("category_id", c)
    
    # Iterate through the materials of the object
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

        # If the material is gold, set a random gold color
        if mat.get_name().lower() == "gold":
            try:
                random_gold_hsv_color = np.random.uniform([0.03, 0.95, 48], [0.25, 1.0, 48])
                random_gold_color = list(hsv_to_rgb(*random_gold_hsv_color)) + [1.0]  # add alpha
                mat.set_principled_shader_value("Base Color", random_gold_color)
            except AttributeError:
                print(f"Error setting Base Color for material {mat.get_name()}")
        
        # Set random shader values for the material
        try:
            mat.set_principled_shader_value("Specular IOR Level", random.uniform(0, 1))
            mat.set_principled_shader_value("Roughness", 0.2)
            mat.set_principled_shader_value("Metallic", 1)
        except AttributeError:
            print(f"Error setting shader value for material {mat.get_name()}")

    # Set random location and rotation for the object
    obj.set_location(np.random.uniform([-1.5, -1.5, 0], [1.5, 1.5, 1.5]))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [2*np.pi, 2*np.pi, 2*np.pi]))
    obj.set_scale(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    obj.set_shading_mode(random.choice(["FLAT", "SMOOTH", "AUTO"]), angle_value=random.uniform(20, 45))
    
    return obj


def add_additional_objects():
    # Add a cube, sphere, or random geometry as noise
    for c in range(3, 3+5):  # Add 3 random objects
        obj_type = random.choice(['CUBE', 'SPHERE', 'CYLINDER', 'CONE', "PLANE", "MONKEY"])
        obj = bproc.object.create_primitive(obj_type)
        
        # Randomly place objects around the scene
        obj.set_location([
            random.uniform(-3, 3),  # X-axis range
            random.uniform(-3, 3),  # Y-axis range
            random.uniform(0, 3)    # Z-axis range (to be above the ground)
        ])
        
        # Random scaling
        obj.set_scale([
            random.uniform(0.05, 0.8),  # Scale factor for X-axis
            random.uniform(0.05, 0.8),  # Scale factor for Y-axis
            random.uniform(0.05, 0.8)   # Scale factor for Z-axis
        ])
        
        # Assign a random color or material
        obj_material = bproc.material.create("random_material")
        obj_material.set_principled_shader_value("Base Color", [
            random.uniform(0, 1),  # R
            random.uniform(0, 1),  # G
            random.uniform(0, 1),  # B
            1.0                    # Alpha
        ])
        obj_material.set_principled_shader_value("Roughness", random.uniform(0.4, 1.0))
        obj.replace_materials(obj_material)
        # Assign the background category to the new objects by not setting the category ID
        obj.set_cp("category_id", c)


def set_lights(objects):
    """
    Randomizes and sets up lighting conditions in the scene.

    This function creates a random number of lights (between 1 and 3) and sets their types, colors, energy levels, 
    and locations. The main light (or the only light if there is just one) has higher energy compared to secondary lights.

    The types of lights that can be created are: POINT, SUN, SPOT, and AREA.

    The color of each light is randomly chosen within a specified range of RGB values.

    The energy of the lights is set based on their role:
    - If there is only one light, its energy is set between 500 and 1500.
    - If there are multiple lights, the main light's energy is set between 1000 and 1500, and secondary lights' energy is set between 200 and 750.

    The location of each light is randomly chosen within a specified range.
    """
    # Randomize lighting conditions
    num_lights = random.randint(1, 3)
    obj1_location = objects[0].get_location()
    center_location = obj1_location
    radius_min, radius_max = 40, 60
    if len(objects) >= 2:  # If there are two objects
        obj2_location = objects[1].get_location()
        # Calculate center and distance between objects
        center_location = (obj1_location + obj2_location) / 2

    for i in range(num_lights):
        light = bproc.types.Light()
        light_types = ["POINT", "SUN", "SPOT", "AREA"]
        light.set_type(random.choice(light_types))
        light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]))
        if num_lights == 1:  # Only one main light
            lower, upper = 500, 1000
        elif i == 0:  # Main light, when there are others.
            lower, upper = 800, 1000
        else:  # Secondary lights.
            lower, upper = 200, 500
        light.set_energy(random.uniform(lower, upper))
        light.set_location(bproc.sampler.shell(
                                            center=center_location,
                                            radius_min=radius_min,
                                            radius_max=radius_max,
                                            elevation_min=1,
                                            elevation_max=90
                                            ))
        # light.set_location(np.random.uniform([-5, -5, 5], [5, 5, 10]))


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


def sample_hdri_background():
    # Randomly choose and set a background image from the HDRI directory
    img_directory = os.path.join(hdri_dir, random.choice(os.listdir(hdri_dir)))  # Choose a random hdri directory (that contains the image)
    img_name = random.choice(os.listdir(img_directory))  # Take the image from the chosen directory.
                                                         # The randomization is to avoid errors if the images within the folder have a different name,
                                                         # or if the folder contains more than one image file.
    img_path = os.path.join(img_directory, img_name)
    # img_path = os.path.join(hdri_dir, "abandoned_bakery/abandoned_bakery_2k.hdr")  # Pizza image

    # Ensures that the world has a node tree
    world = bpy.context.scene.world
    if world.node_tree is None:
        world.use_nodes = True  # Enable node tree if it's not set
    bproc.world.set_world_background_hdr_img(img_path)

    # Set a random world lighting strength
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.1, 1.5)


def sampling_camera_position(objects, camera_tries, camera_successes):
    """
    Samples a camera position around given objects and adds the camera pose if the objects are visible.

    Args:
        objects (list): A list of objects to be considered for camera positioning. 
                        The objects should have a method `get_location()` that returns their location.
        camera_tries (int): The number of attempts made to sample a valid camera position.
        camera_successes (int): The number of successful camera positions where the objects are visible.

    Returns:
        tuple: Updated values of `camera_tries` and `camera_successes`.

    Notes:
        - The function recursively calls itself if the sampled camera position does not make the objects visible.
        - The camera position is sampled within a spherical shell defined by `radius_min` and `radius_max` around the center location.
        - The center location is between the first two objects if there are at least two objects, otherwise it is the location of the first object.
        - The function uses `bproc.sampler.shell` to sample the camera location and `bproc.camera.rotation_from_forward_vec` to compute the rotation matrix.
        - The camera pose is added using `bproc.camera.add_camera_pose` if the objects are visible from the sampled position.
    """
    flag = True
    camera_tries += 1
    obj1_location = objects[0].get_location()
    center_location = obj1_location
    radius_min, radius_max = 10, 30
    if len(objects) >= 2:  # If there are two objects
        obj2_location = objects[1].get_location()
        # Calculate center and distance between objects
        center_location = (obj1_location + obj2_location) / 2
        distance_between_objects = np.linalg.norm(obj1_location - obj2_location)
        # Set radius based on object distance
        radius_max = max(distance_between_objects * 3, 30)

    # Sample random camera location around the object
    location = bproc.sampler.shell(
        center=center_location,
        radius_min=radius_min,
        radius_max=radius_max,
        elevation_min=-90,
        elevation_max=90
    )
    # Compute rotation based lookat point which is placed randomly around the objects
    poi = center_location + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location and rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    # Only add camera pose if object is still visible
    if (objects[0] in bproc.camera.visible_objects(cam2world_matrix))\
    or ((len(objects) >= 2) and (objects[1] in bproc.camera.visible_objects(cam2world_matrix))):
        bproc.camera.add_camera_pose(cam2world_matrix)
        camera_successes += 1
    else:
        # camera_tries, camera_successes = sampling_camera_position(objects, camera_tries, camera_successes)
        flag = False
    return camera_tries, camera_successes, flag


def preprocess_instance_segmaps(instance_segmaps):
    """
    Preprocess instance segmentation maps to ensure correct contour format for COCO writer.
    
    :param instance_segmaps: List of binary masks (numpy arrays)
    :return: List of polygons for COCO annotations
    """
    processed_polygons = []
    for binary_mask in instance_segmaps:
        contours = measure.find_contours(binary_mask, 0.5)
        # Convert each contour to a list of points
        contours_list = [contour.tolist() for contour in contours]
        processed_polygons.append(contours_list)
    
    return processed_polygons


def convert_hdf5_to_images(hdf5_format_dir, jpg_format_dir):
    """
    Converts HDF5 files containing image data to JPEG format and saves them.
    Args:
        hdf5_format_dir (str): Directory containing the HDF5 files.
        jpg_format_dir (str): Directory where the JPEG images will be saved.
    The function expects each HDF5 file to contain two datasets:
        - 'colors': RGB image data.
        - 'instance_segmaps': Segmentation maps.
    For each HDF5 file in the input directory, the function:
        1. Reads the 'colors' dataset and saves it as a JPEG image.
        2. Reads the 'instance_segmaps' dataset, normalizes it, applies a colormap,
           and saves it as a JPEG image.
    """
    for i in range(len(os.listdir(hdf5_format_dir))):
        file = str(i) + ".hdf5"
        file_path = os.path.join(hdf5_format_dir, file)
        os.makedirs(jpg_format_dir, exist_ok=True)
        colors_dir = os.path.join(jpg_format_dir, str(i) + "_color.jpg")
        segmaps_dir = os.path.join(jpg_format_dir, str(i) + "_segmaps.jpg")
        with h5py.File(file_path, 'r') as file:
            colors = file['colors'][:]
            color_img = Image.fromarray(colors, 'RGB')
            color_img.save(colors_dir, "JPEG")
            
            segmaps = file['instance_segmaps'][:]
            # Set values greater than 2 to 0
            segmaps[segmaps > 2] = 0
            
            # Normalize or map to a color range for visibility
            segmaps_normalized = (segmaps - segmaps.min()) / (segmaps.max() - segmaps.min())
            segmaps_colormap = (cm.viridis(segmaps_normalized)[:, :, :3] * 255).astype(np.uint8)
            
            segmaps_img = Image.fromarray(segmaps_colormap)
            segmaps_img.save(segmaps_dir, "JPEG")


def paste_coco_backgrounds(no_background_images_dir):
    """
    Pastes images from a directory onto random COCO backgrounds and saves them.

    Args:
        no_background_images_dir (str): The directory containing images without backgrounds.
    """
    # Iterate over each image in the given directory
    for image in os.listdir(no_background_images_dir):
        # if "color" not in image:
        #     continue
        image_path = os.path.join(no_background_images_dir, image)
        img = Image.open(image_path)

        # Choose a random background from the COCO directory
        chosen_background = random.choice(os.listdir(coco_dir))
        background = Image.open(os.path.join(coco_dir, chosen_background))
        # Resize the background to match the size of the original image
        background = background.resize(img.size)

        background.paste(img, mask=img.convert('RGBA'))
        # Save the new image back to the original path, overriding the background-less image
        background.save(os.path.join(with_background_dir, image))


def main(args):
    if args.debug:
        # Debugging if specified
        debugpy.listen(5678)
        debugpy.wait_for_client()

    # Initialize BlenderProc
    bproc.init()

    # Set up scene: background, lighting, and instruments
    camera_tries, camera_successes = 0, 0
    is_hdri = True
    while (camera_tries < 10000) and (camera_successes < args.num_images):  # Generate specified number of images
        # print("\nCamera tries:", camera_tries, "Camera successes:", camera_successes, "\n")
        # Clear all key frames from the previous run
        bproc.utility.reset_keyframes()
        bproc.clean_up()

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

        add_additional_objects()
        
        # Set up lights in the scene
        set_lights(objects)
        
        # Set up initial camera parameters
        initial_camera_setup()
        
        # Choose a random background for the scene
        # is_hdri = random.random() > 0.5
        if is_hdri:
            sample_hdri_background()
            output_and_background_dir = output_hdri_dir
        else:
            output_and_background_dir = output_coco_dir
        
        # Sample camera positions around the objects
        flag = False
        while not flag:
            camera_tries, camera_successes, flag = sampling_camera_position(objects, camera_tries, camera_successes)
        # Uncomment to add additional variations like blur or noise
        # further_complicate_image(objects)

        # Set the maximum number of samples for rendering to speed up the process
        bproc.renderer.set_max_amount_of_samples(16)
        
        # Disable transparency so the background becomes opaque
        bproc.renderer.set_output_format(enable_transparency=False)
        
        # Enable segmentation masks (per class and per instance)
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

        """
        # Uncomment to activate normal and depth rendering
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_normals_output()
        """

        bproc.renderer.set_denoiser("OPTIX")
        bproc.renderer.set_noise_threshold(0.05)

        # Render the image and segmentation mask
        data = bproc.renderer.render()

        # Uncomment to preprocess instance segmentation maps before passing them to the COCO writer
        # processed_instance_segmaps = preprocess_instance_segmaps(data["instance_segmaps"])
        # processed_instance_segmaps = data["instance_segmaps"]
        
        # Uncomment to write data to COCO file
        # try:
        #     bproc.writer.write_coco_annotations(no_background_images_dir,
        #                                         instance_segmaps=data["instance_segmaps"],
        #                                         instance_attribute_maps=data["instance_attribute_maps"],
        #                                         colors=data["colors"],
        #                                         mask_encoding_format="polygon",
        #                                         append_to_existing_output=True)
        # except:
        #     pass

        # Save images and masks in HDF5 format
        hdf5_format_dir = os.path.join(output_and_background_dir, 'hdf5_format/')
        bproc.writer.write_hdf5(hdf5_format_dir, data, append_to_existing_output=True)

    # Convert HDF5 files to JPEG format
    convert_hdf5_to_images(hdf5_format_dir, os.path.join(output_hdri_dir, 'jpg_format/'))
    
    # convert_hdf5_to_images(hdf5_format_dir, os.path.join(output_coco_dir, 'jpg_format/'))
    # paste_coco_backgrounds(no_background_images_dir)
    # paste_coco_backgrounds(os.path.join(no_background_images_dir, 'images/'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--blender', action='store_true', help="Run in Blender")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debugging")
    parser.add_argument('-n', '--num_images', type=int, default=1000, help="Number of images to generate")
    main(parser.parse_args())
