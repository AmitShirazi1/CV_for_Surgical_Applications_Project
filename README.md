# Computer Vision for Surgical Applications Project

~ Need to add a link to the final model weights ~

## Overview

This project implements an advanced image segmentation system for surgical instruments using synthetic data generation, segmentation model development, and domain adaptation techniques. It simulates a real-world scenario where annotated medical data is unavailable due to the high cost of expert annotation.

The project is divided into three phases:
1. **Synthetic Data Generation**: Creating a synthetic dataset of surgical instrument images and segmentation masks.
2. **Segmentation Model Development**: Training and fine-tuning a DeepLabV3+ model on the synthetic dataset.
3. **Domain Adaptation**: Enhancing the model's performance on real surgical images by bridging the domain gap.

Additionally, the project provides two utility scripts:
- `predict.py`: For running segmentation on single images.
- `video.py`: For applying segmentation on video frames.

---

## Project Structure

```
project/
│
├── data_generation/          # Phase 1: Synthetic Data Generation
├── model_development/        # Phase 2: Segmentation Model Development
├── domain_adaptation/        # Phase 3: Domain Adaptation
├── predict.py                # Utility script for single-image predictions
├── video.py                  # Utility script for video segmentation
└── requirements/             # Dependency requirements for each phase
```

## Key Scripts

### `predict.py`

This script predicts segmentation masks for a single image using a trained DeepLabV3+ model.

#### Key Arguments

1. **`-i` or `--image_idx`** (optional):  
   Specify the index of an image from the default path `./data_generation/output/jpg_format/`.  
   Example: `-i 0` for the image `0_color.jpg`.  
   **Cannot be used with `--image_path`.**

2. **`-p` or `--image_path`** (optional):  
   Provide the full path to the image file.  
   Example: `-p ./path/to/image.jpg`.  

3. **`-m` or `--model_path`**:  
   Path to the trained model. Default: `./model_development/deeplabv3_model.pth`.

4. **`-o` or `--output_dir`**:  
   Directory to save the results. Must be specified unless `--dev` or `--test` is used.

5. **`--dev`**: Use the development directory (`./model_development/predictions/`) as the output location.  
6. **`--test`**: Use the test directory (`./domain_adaptation/predictions/`) as the output location.

#### Rules and Examples
- Provide **either** `--image_idx` or `--image_path` (not both).
- Specify **one output option**: `-o`, `--dev`, or `--test`.
  
Example:
```bash
python predict.py -i 0 --dev
python predict.py -p ./image.jpg -o ./results/
```

### `video.py` Arguments

This script processes a video and applies segmentation frame-by-frame.

### Key Arguments

1. **`-v` or `--input_video_path`** (required):  
   Path to the input video. Default: `/datashare/project/vids_test/4_2_24_A_1.mp4`.

2. **`-m` or `--model_path`**:  
   Path to the trained model. Default: `./model_development/deeplabv3_model.pth`.

3. **`-o` or `--output_dir`**:  
   Directory to save the segmented video. Must be specified unless `--dev` or `--test` is used.

4. **`--dev`**: Use the development directory (`./model_development/predictions/`) as the output location.  
5. **`--test`**: Use the test directory (`./domain_adaptation/predictions/`) as the output location.

### Rules and Examples
- Specify **one output option**: `-o`, `--dev`, or `--test`.  

Example:
```bash
python video.py -v ./input.mp4 --test
python video.py -o ./output/
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AmitShirazi1/CV_for_Surgical_Applications_Project.git
   cd CV_for_Surgical_Applications_Project/
   ```

2. Install basic dependencies:
   ```bash
   pip install -r requirements/basic.txt
   ```

3. Install dependencies for the desired phase:
   ```bash
   pip install -r requirements/<phase>.txt
   ```
   Replace `<phase>` with `data_generation`, `model_development`, or `domain_adaptation`.

4. Install Blender for Phase 1 (Synthetic Data Generation):  
   [Download and Install Blender](https://www.blender.org/download/).

---

## How to Run

### Phase 1: Synthetic Data Generation
1. Generate synthetic images and segmentation masks:
   ```bash
   blenderproc run ./data_generation/synthetic_data_generator.py -n 1000
   ```
   Replace `1000` with the desired number of images.

2. Review and inspect the generated dataset:
   Open `investigate_hdf5_output.ipynb` in Jupyter Notebook to visualize or convert the dataset.

Refer to the [Phase 1 README](./data_generation/) for details.

---

### Phase 2: Segmentation Model Development
1. Train the model using synthetic data:
   ```bash
   python ./model_development/train.py --images_path <train_data> --val_images_path <val_data>
   ```
   Replace `<train_data>` and `<val_data>` with the paths to training and validation datasets.

2. Fine-tune the trained model:
   ```bash
   python ./model_development/finetuning.py --images_path <train_data>
   ```

Refer to the [Phase 2 README](./model_development/) for details.

---

### Phase 3: Domain Adaptation
1. Prepare and organize datasets:
   ```bash
   python ./domain_adaptation/adapting_with_cyclegan.py
   ```

2. Train a CycleGAN model for translating synthetic images into real-style images:
   ```bash
   python ./pytorch-CycleGAN-and-pix2pix/train.py --dataroot ./domain_adaptation/data --name synthetic2real
   ```

3. Fine-tune the segmentation model using the domain-adapted dataset:
   ```bash
   python ./domain_adaptation/finetuning.py -i <images_dir> -m <masks_dir>
   ```

Refer to the [Phase 3 README](./domain_adaptation/) for details.

---

## Outputs

1. **Phase 1**: Synthetic dataset in HDF5 and JPEG formats.
2. **Phase 2**: Trained segmentation model (`deeplabv3_model.pth`) and fine-tuned model (`deeplabv3_model_tuned.pth`).
3. **Phase 3**: Domain-adapted model (`deeplabv3_model_adapted.pth`).

Predicted results are saved in the specified `output_dir` during inference.

---

## Notes and Recommendations

- Use a GPU for training and fine-tuning to improve performance.
- Phase-specific READMEs provide additional implementation details, insights, and challenges encountered.
- Further refinements could include experimenting with additional domain adaptation techniques or expanding the synthetic dataset's diversity.





---

## `predict.py` Arguments

This script predicts the segmentation mask for a given image using a trained DeepLabV3+ model.

### Arguments

1. **`-i` or `--image_idx`** (optional):
   - Specifies the index of an image from the default dataset path: `./data_generation/output/jpg_format/`.
   - Example:
     ```bash
     python predict.py -i 0
     ```
     This will process the image named `0_color.jpg` from the default dataset path.
   - **Default**: None.

2. **`-p` or `--image_path`** (optional):
   - Specifies the absolute or relative path to the image file you want to predict on.
   - Example:
     ```bash
     python predict.py -p ./path/to/your/image.jpg
     ```
   - **Default**: None.

   **Mutual Exclusivity**: You cannot use both `--image_idx` and `--image_path` together. You must specify **either** an image index **or** an image path, but not both. If both are provided, the script raises an error.

3. **`-m` or `--model_path`**:
   - Specifies the file path of the trained model to use for prediction.
   - Example:
     ```bash
     python predict.py -m ./model_development/deeplabv3_model.pth
     ```
   - **Default**: `./model_development/deeplabv3_model.pth`.

4. **`-o` or `--output_dir`**:
   - Specifies the directory where the prediction result will be saved.
   - Example:
     ```bash
     python predict.py -o ./predictions/
     ```
   - **Default**: None. If not provided, you must use one of the flags (`--dev` or `--test`) to determine the output directory.

5. **`--dev`** (flag):
   - A flag indicating that the script is running in **development mode**, which sets the output directory to `./model_development/predictions/`.

6. **`--test`** (flag):
   - A flag indicating that the script is running in **test mode**, which sets the output directory to `./domain_adaptation/predictions/`.

### Argument Rules and Restrictions
- At least **one of the following arguments must be provided**: `--image_idx` or `--image_path`.
- At least **one of the following arguments must be provided**: `-o`, `--dev`, or `--test`.
- **Conflicts**:
  - You cannot use both `--image_idx` and `--image_path` simultaneously.
  - You cannot use more than one output-related argument (`-o`, `--dev`, `--test`) at the same time.

---

## `video.py` Arguments

This script processes video frames to predict segmentation masks for each frame using a trained DeepLabV3+ model. The results are saved as a new video file.

### Arguments

1. **`-v` or `--input_video_path`** (required):
   - Specifies the absolute or relative path to the video file to process.
   - Example:
     ```bash
     python video.py -v ./path/to/video.mp4
     ```
   - **Default**: `/datashare/project/vids_test/4_2_24_A_1.mp4`.

2. **`-m` or `--model_path`**:
   - Specifies the file path of the trained model to use for prediction.
   - Example:
     ```bash
     python video.py -m ./model_development/deeplabv3_model.pth
     ```
   - **Default**: `./model_development/deeplabv3_model.pth`.

3. **`-o` or `--output_dir`**:
   - Specifies the directory where the segmented video will be saved.
   - Example:
     ```bash
     python video.py -o ./predictions/
     ```
   - **Default**: None. If not provided, you must use one of the flags (`--dev` or `--test`) to set the output directory.

4. **`--dev`** (flag):
   - A flag indicating that the script is running in **development mode**, which sets the output directory to `./model_development/predictions/`.

5. **`--test`** (flag):
   - A flag indicating that the script is running in **test mode**, which sets the output directory to `./domain_adaptation/predictions/`.

### Argument Rules and Restrictions
- The `--input_video_path` argument is required and cannot be omitted.
- At least **one of the following arguments must be provided**: `-o`, `--dev`, or `--test`.
- **Conflicts**:
  - You cannot use more than one output-related argument (`-o`, `--dev`, `--test`) at the same time.

---

## Examples

### Using `predict.py`

1. Predict segmentation for the first image in the default dataset, saving results in the development directory:
   ```bash
   python predict.py -i 0 --dev
   ```

2. Predict segmentation for a specific image file and save results in a custom directory:
   ```bash
   python predict.py -p ./images/my_image.jpg -o ./results/
   ```

### Using `video.py`

1. Process a video using the default model and save results in the test directory:
   ```bash
   python video.py -v ./videos/input_video.mp4 --test
   ```

2. Process a video, specifying both the model and custom output directory:
   ```bash
   python video.py -v ./videos/input_video.mp4 -m ./custom_model.pth -o ./output_videos/
   ```

These details ensure proper usage of both scripts and avoid conflicts in argument combinations. Let me know if you'd like further elaboration or adjustments!