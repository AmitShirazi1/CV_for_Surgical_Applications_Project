Thank you for the feedback! Here's the revised README for Phase 1 with the necessary changes:
# Phase 1 - Synthetic Data Generation

## Overview
Phase 1 of the project focuses on creating a diverse synthetic dataset of surgical instrument images with segmentation masks. These synthetic images simulate real-world conditions and serve as training data for segmentation models in subsequent phases.

The output includes:
- Images of surgical instruments in random positions, lighting, and backgrounds.
- Corresponding segmentation masks that map each pixel to a specific class ID.

---

## Directory Structure
**Files and folders for Phase 1 are located in:** `./data_generation/`

### Files
1. **`synthetic_data_generator.py`**  
   - **Purpose:** Main script for generating synthetic images and segmentation masks.  
   - **Key Features:**
     - Dynamically randomizes instrument positioning, lighting, and backgrounds.
     - Generates output in both HDF5 and JPEG formats.  
   - **Arguments:**  
     Run the script using the following optional arguments:
     ```bash
     blenderproc run ./data_generation/synthetic_data_generator.py [OPTIONS]
     ```
     Options include:
     - `-n` or `--num_images`: Number of images to generate (default: 1000).
     - `-d` or `--debug`: Enables debug mode.
     - `-b` or `--blender`: Ensures the script runs within Blender.  

2. **`investigate_hdf5_output.ipynb`**  
   - **Purpose:** A Jupyter notebook for inspecting and processing the generated HDF5 files.  
   - **Key Features:**
     - Visualizes the contents of HDF5 files, including RGB images and segmentation maps.
     - Converts HDF5 files to JPEG format for easier analysis.

### Folder
1. **`example_output/`**  
   - **Purpose:** Demonstrates the output of running the synthetic data generator script.  
   - **Contents:**
     - `hdf5_format/`: Contains HDF5 files for 10 example images.
     - `jpg_format/`: JPEG versions of the RGB images and segmentation maps.

---

## Requirements
1. **Software:**
   - [Blender](https://www.blender.org/) (version 2.93 or later).
   - Python (version 3.7 or later).
   - Jupyter Notebook (for running `investigate_hdf5_output.ipynb`).

2. **Python Packages:**
   Don't forget to install required packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Resources:**
   Ensure the following resources are available in `/datashare/project/`:
   - 3D models of surgical instruments (`.obj` format).
   - Camera intrinsic parameters (`camera.json`).
   - HDRI files for backgrounds.

---

## Outputs
The script generates outputs in two formats:
1. **HDF5 Format (`output/hdf5_format/`):**
   - Includes datasets such as:
     - `colors`: RGB images.
     - `category_id_segmaps`: Segmentation masks with class ID annotations.

2. **JPEG Format (`output/jpg_format/`):**
   - RGB images (`*_color.jpg`).
   - Normalized segmentation maps (`*_segmaps.jpg`).

---

## How to Use
1. **Generate Synthetic Data:**
   Run the script to create a dataset:
   ```bash
   blenderproc run ./data_generation/synthetic_data_generator.py -b -n <NUM_IMAGES>
   ```
   Replace `<NUM_IMAGES>` with the desired number of images (default is 1000).

2. **Analyze Generated Data:**
   Open `investigate_hdf5_output.ipynb` in Jupyter:
   ```bash
   jupyter notebook ./data_generation/investigate_hdf5_output.ipynb
   ```
   Use the notebook to inspect HDF5 files or convert them into JPEG format.

3. **Example Validation:**
   Explore `example_output/` to review samples of generated data.
