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
import gc


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


-------------- Trying COCO --------------

def preprocess_instance_segmaps(instance_segmaps):
    # Preprocess instance segmentation maps to ensure correct contour format for COCO writer.
    
    # :param instance_segmaps: List of binary masks (numpy arrays)
    # :return: List of polygons for COCO annotations
    processed_polygons = []
    for binary_mask in instance_segmaps:
        contours = measure.find_contours(binary_mask, 0.5)
        # Convert each contour to a list of points
        contours_list = [contour.tolist() for contour in contours]
        processed_polygons.append(contours_list)
    
    return processed_polygons

def paste_coco_backgrounds(no_background_images_dir):
    # Pastes images from a directory onto random COCO backgrounds and saves them.

    # Args:
    #     no_background_images_dir (str): The directory containing images without backgrounds.
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

        
In main:

# Before while:
is_hdri = True

# In while, when choosing background:
is_hdri = random.random() > 0.5
if is_hdri:
    sample_hdri_background()
    output_and_background_dir = output_hdri_dir
else:
    output_and_background_dir = output_coco_dir

# After rendering:
# Uncomment to preprocess instance segmentation maps before passing them to the COCO writer
processed_instance_segmaps = preprocess_instance_segmaps(data["instance_segmaps"])
processed_instance_segmaps = data["instance_segmaps"]

# Uncomment to write data to COCO file
try:
    bproc.writer.write_coco_annotations(no_background_images_dir,
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        mask_encoding_format="polygon",
                                        append_to_existing_output=True)
except:
    pass

# After while, after converting hdf5 to jpg:
convert_hdf5_to_images(hdf5_format_dir, os.path.join(output_coco_dir, 'jpg_format/'))
paste_coco_backgrounds(no_background_images_dir)
paste_coco_backgrounds(os.path.join(no_background_images_dir, 'images/'))

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
    Loads a random number of instrument objects from specified directories.
    This function randomly selects a number of instruments (between 1 and 4) to load.
    It chooses .obj files from the specified directories for needle holders and tweezers,
    loads them using the `bproc.loader.load_obj` method, and assigns a category ID to each object.

    Returns:
        list: A list of loaded instrument objects with assigned category IDs.
    """    
    # Randomly choose an .obj file from the given directory
    def choose_obj_file(dir):
        return random.choice([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.obj')])
    
    # Load the .obj file using the `bproc.loader.load_obj` method
    def load_obj(path):
        return bproc.loader.load_obj(path)[0]
    
    # Randomly choose the number of instruments to load
    instrument_num = np.random.choice(range(1, 5), p=[0.1, 0.6, 0.2, 0.1])
    tool_paths = [choose_obj_file(needle_holder_model_dir), choose_obj_file(tweezers_model_dir)]

    tool_objs = []
    for i in range(instrument_num):
        if len(tool_paths):  # Randomly choose a tool, options are needle holder and tweezers
            tool_path = random.choice(tool_paths)
            tool_paths.remove(tool_path)

        else:  # If we have put two instruments, one needle holder and one tweezers, then we can randomely add more of them.
            dir = random.choice([needle_holder_model_dir, tweezers_model_dir])
            tool_path = choose_obj_file(dir)
        
        tool_obj = load_obj(tool_path)
        # Set the category ID for the object
        c = 2 if "tweezers" in tool_path else 1
        tool_obj.set_cp("category_id", c)
        tool_objs.append(tool_obj)
    return tool_objs


def set_instruments_appearance_and_location(obj): 
    """
    Sets the appearance and location of the given object.
    This function iterates through the materials of the object and performs the following actions:
    - If the material is gold, sets a random gold color.
    - Sets random shader values for the material, including Specular IOR Level, Roughness, and Metallic.
    - Sets a random location, rotation, and scale for the object.
    - Sets a random shading mode for the object.

    Parameters:
        obj: The object whose appearance and location are to be set. The object should have a corresponding '.mtl' file.
    """
    # Iterate through the materials of the object
    for mat in obj.get_materials():
        # Check if the material has a Principled BSDF node (For error-checking purposes).
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
    obj.set_location(np.random.uniform([-2, -2, 0], [2, 2, 1]))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [2*np.pi, 2*np.pi, 2*np.pi]))
    obj.set_scale(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    obj.set_shading_mode(random.choice(["FLAT", "SMOOTH", "AUTO"]), angle_value=random.uniform(20, 45))


def add_additional_objects():
    """
    Adds a random number of additional objects with various shapes to the scene,
    with random locations, scales, and materials.
    The objects are placed randomly within a specified range and assigned random colors and roughness values.

    The function performs the following steps:
    1. Generates a random number of objects to add (between 3 and 7).
    2. For each object:
        - Randomly selects an object type from a predefined list.
        - Creates the object using the selected type.
        - Sets a random location within the specified range.
        - Sets a random scale for each axis.
        - Creates a random material with a random color and roughness.
        - Assigns the material to the object.
        - Sets the category ID to 0 (background category).
    """
    num = random.randint(3, 7)
    for _ in range(num):  # Add random objects in a random amount of times.
        # Add a cube, sphere, or random geometry as additional objects that interact with the medical instruments.
        obj_type = random.choice(['CUBE', 'SPHERE', 'CYLINDER', 'CONE', "PLANE", "MONKEY"])
        obj = bproc.object.create_primitive(obj_type)
        
        # Randomly place objects around the scene
        obj.set_location([
            random.uniform(-4, 4),  # X-axis range
            random.uniform(-4, 4),  # Y-axis range
            random.uniform(0, 3)    # Z-axis range (to be above the ground)
        ])
        
        # Random scaling
        obj.set_scale([
            random.uniform(0.05, 1),  # Scale factor for X-axis
            random.uniform(0.05, 1),  # Scale factor for Y-axis
            random.uniform(0.05, 1)   # Scale factor for Z-axis
        ])
        
        # Assign a random color, material and roughness to the object
        obj_material = bproc.material.create("random_material")
        obj_material.set_principled_shader_value("Base Color", [
            random.uniform(0, 1),  # R
            random.uniform(0, 1),  # G
            random.uniform(0, 1),  # B
            1.0                    # Alpha
        ])
        obj_material.set_principled_shader_value("Roughness", random.uniform(0.2, 0.6))
        obj.replace_materials(obj_material)
        # Assign the background category to the new objects by not setting the category ID
        obj.set_cp("category_id", 0)


def set_lights(objects):
    """
    Randomizes and sets lighting conditions, with respect to the given objects.

    Parameters:
        objects (list): A list of objects for which the lighting conditions are to be set. 

    The function performs the following steps:
        1. Randomly determines the number of lights (between 1 and 3).
        2. If there is only one object, sets the light around that object.
        3. If there are multiple objects, sets the light around the center of the first two objects.
        4. Creates light objects with random types and colors.
        5. Sets the energy of the lights, with the main light having higher energy.
        6. Sets the location of the lights around the objects within a specified radius and elevation range.
    """
    # Randomize lighting conditions
    num_lights = random.randint(1, 3)
    # In the case of only one object, set the light around the object
    obj1_location = objects[0].get_location()
    center_location = obj1_location
    radius_min, radius_max = 40, 60

    # In the case of more than one object, set the light around the center of the first two objects
    if len(objects) >= 2:
        obj2_location = objects[1].get_location()
        # Calculate center and distance between objects
        center_location = (obj1_location + obj2_location) / 2

    for i in range(num_lights):
        # Create a light object and set its type and color.
        light = bproc.types.Light()
        light_types = ["POINT", "SUN", "SPOT", "AREA"]
        light.set_type(random.choice(light_types))
        light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]))

        if num_lights == 1:  # If there is only one main light, set the energy to a high value.
            lower, upper = 500, 1000
        elif i == 0:  # Set the energy of the main light to a higher value, when there are other lights in the scene.
            lower, upper = 800, 1000
        else:  # Set the energy of the additional lights to a lower value.
            lower, upper = 200, 500
        light.set_energy(random.uniform(lower, upper))

        # Set the location of the light around the objects
        light.set_location(bproc.sampler.shell(
                                            center=center_location,
                                            radius_min=radius_min,
                                            radius_max=radius_max,
                                            elevation_min=1,
                                            elevation_max=90
                                            ))


def load_camera_parameters(json_path):
    """ Load the camera parameters from json file. """
    with open(json_path, 'r') as f:
        camera_params = json.load(f)
    return camera_params


def initial_camera_setup():
    """
    Sets up the initial camera parameters for the synthetic data generation.
    This function loads the camera parameters from a JSON file, extracts the intrinsic
    camera parameters (focal lengths and principal point), and sets up the camera
    intrinsics using these parameters.

    The camera parameters are expected to be in a JSON file named 'camera.json' located
    in the resources directory.

    Raises:
        FileNotFoundError: If the 'camera.json' file is not found in the resources directory.
        KeyError: If any of the required camera parameters are missing in the JSON file.
    """
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
    """
    Randomly selects and sets an HDRI background image for the Blender scene.

    Raises:
        FileNotFoundError: If the HDRI directory or image file does not exist.
    """
    # Randomly choose and set a background image from the HDRI directory
    img_directory = os.path.join(hdri_dir, random.choice(os.listdir(hdri_dir)))  # Choose a random hdri directory (that contains the image)
    img_name = random.choice(os.listdir(img_directory))  # Take the image from the chosen directory.
                                                         # The randomization is to avoid errors if the images within the folder have a different name,
                                                         # or if the folder contains more than one image file.
    img_path = os.path.join(img_directory, img_name)

    # Ensures that the world has a node tree (for error-checking purposes).
    world = bpy.context.scene.world
    if world.node_tree is None:
        world.use_nodes = True  # Enable node tree if it's not set
    bproc.world.set_world_background_hdr_img(img_path)

    # Set a random world lighting strength
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.1, 1.5)


def sampling_camera_position(objects, camera_tries, camera_successes):
    """
    Samples a camera position around given objects and updates the camera pose if the objects are visible.

    Parameters:
    objects (list): A list of objects around which the camera position is to be sampled. Each object should have a `get_location` method.
    camera_tries (int): The number of attempts made to sample a camera position.
    camera_successes (int): The number of successful camera positions where the objects are visible.

    Returns:
    tuple: A tuple containing updated values of camera_tries, camera_successes, and a flag indicating if the camera position was successfully added.
    """
    flag = True
    camera_tries += 1

    # In the case of only one object, set the camera around the object
    obj1_location = objects[0].get_location()
    center_location = obj1_location
    radius_min, radius_max = 10, 30

    # In the case of more than one object, set the camera around the center of the first two objects
    if len(objects) >= 2:
        obj2_location = objects[1].get_location()
        # Calculate center and distance between objects
        center_location = (obj1_location + obj2_location) / 2
        distance_between_objects = np.linalg.norm(obj1_location - obj2_location)
        # Set radius based on object distance
        radius_max = max(distance_between_objects * 3, 30)

    # Sample random camera location that's around the object.
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
        flag = False
    return camera_tries, camera_successes, flag


def convert_hdf5_to_images(hdf5_format_dir, jpg_format_dir):
    """
    Converts HDF5 files containing image data to JPEG format.
    This function reads HDF5 files from the specified directory, extracts the 'colors' and 
    'category_id_segmaps' datasets, and saves them as JPEG images in the specified output directory.

    Args:
        hdf5_format_dir (str): The directory containing the HDF5 files.
        jpg_format_dir (str): The directory where the JPEG images will be saved.

    Raises:
        FileNotFoundError: If the specified HDF5 file does not exist.
        KeyError: If the required datasets ('colors' or 'category_id_segmaps') are not found in the HDF5 file.
        OSError: If there is an error in creating directories or saving images.
    """
    for i in range(len(os.listdir(hdf5_format_dir))):
        # Load the HDF5 file
        file = str(i) + ".hdf5"
        file_path = os.path.join(hdf5_format_dir, file)
        # Create the directory that will contain the JPEG images, and set the file paths.
        os.makedirs(jpg_format_dir, exist_ok=True)
        colors_dir = os.path.join(jpg_format_dir, str(i) + "_color.jpg")
        segmaps_dir = os.path.join(jpg_format_dir, str(i) + "_segmaps.jpg")

        with h5py.File(file_path, 'r') as file:
            # Read the 'colors' dataset and save it as a JPEG image.
            colors = file['colors'][:]
            color_img = Image.fromarray(colors, 'RGB')
            color_img.save(colors_dir, "JPEG")
            
            # Read the 'category_id_segmaps' dataset.
            segmaps = file['category_id_segmaps'][:]
            # Normalize and map the segmaps to a color range for visibility.
            segmaps_normalized = (segmaps - segmaps.min()) / (segmaps.max() - segmaps.min())
            segmaps_colormap = (cm.viridis(segmaps_normalized)[:, :, :3] * 255).astype(np.uint8)
            # Save the segmaps as a JPEG image.
            segmaps_img = Image.fromarray(segmaps_colormap)
            segmaps_img.save(segmaps_dir, "JPEG")


def main(args):
    if args.debug:
        # Debugging if specified
        debugpy.listen(5678)
        debugpy.wait_for_client()

    # Initialize BlenderProc
    bproc.init()

    # Set up scene: background, lighting, and instruments
    camera_tries, camera_successes = 0, 0
    while (camera_tries < 10000) and (camera_successes < args.num_images):  # Generate specified number of images
        # Clear all key frames from the previous run
        bproc.utility.reset_keyframes()
        bproc.clean_up()

        # Load the surgical instruments, set their appearance and location.
        instruments = load_instruments()
        for inst in instruments:
            set_instruments_appearance_and_location(inst)

        # Add additional objects to the scene that interact with the instruments.
        add_additional_objects()
        
        # Set up lights in the scene.
        set_lights(instruments)
        
        # Set up initial camera parameters.
        initial_camera_setup()
        
        # Choose a random background for the scene.
        sample_hdri_background()
        
        # Sample camera positions around the objects
        flag = False
        while not flag:
            camera_tries, camera_successes, flag = sampling_camera_position(instruments, camera_tries, camera_successes)
        # Uncomment to add additional variations like blur or noise
        # further_complicate_image(objects)

        # Set the maximum number of samples for rendering to speed up the process
        bproc.renderer.set_max_amount_of_samples(16)
        
        # Disable transparency so the background becomes opaque
        bproc.renderer.set_output_format(enable_transparency=False)
        print("boo")
        objects = bproc.object.get_all_mesh_objects()
        for obj in objects:
            print(obj.get_cp("category_id"), obj.get_name())

        # Enable segmentation masks (per class and per instance)
        bproc.renderer.enable_segmentation_output(map_by=['category_id'])

        bproc.renderer.set_denoiser("OPTIX")
        bproc.renderer.set_noise_threshold(0.05)

        # Render the image and segmentation mask
        data = bproc.renderer.render()

        # Save images and masks in HDF5 format    
        hdf5_format_dir = os.path.join(output_hdri_dir, 'hdf5_format/')
        bproc.writer.write_hdf5(hdf5_format_dir, data, append_to_existing_output=True)

    # Convert HDF5 files to JPEG format
    convert_hdf5_to_images(hdf5_format_dir, os.path.join(output_hdri_dir, 'jpg_format/'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--blender', action='store_true', help="Run in Blender")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debugging")
    parser.add_argument('-n', '--num_images', type=int, default=1000, help="Number of images to generate")
    main(parser.parse_args())
