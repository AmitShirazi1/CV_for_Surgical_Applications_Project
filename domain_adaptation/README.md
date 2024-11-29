# Phase 3: Domain Adaptation for Surgical Instrument Segmentation

This phase implements **domain adaptation** to enhance a segmentation model's performance on real surgical images by bridging the gap between synthetic and real-world data. It includes data organization, domain adaptation, and model fine-tuning using the provided datasets and tools.

## Files and Directories

### `./domain_adaptation/`
This directory contains the scripts and tools for the domain adaptation process:

#### 1. `adapting_with_cyclegan.py`
- **Purpose**: Prepares and organizes datasets for domain adaptation.
- **Key Features**:
  - Splits synthetic images into training and testing subsets.
  - Extracts and organizes frames from real surgical videos.
- **Main Functions Used**:
  - `copy_and_split_folder_with_images` from `organizing_data.py`
  - `copy_and_split_video_frames` from `organizing_data.py`

#### 2. `organizing_data.py`
- **Purpose**: Provides helper functions for organizing and preparing datasets.
- **Key Functions**:
  1. `copy_and_split_folder_with_images(src_folder, dest_folder, direction, train_ratio)`:
     - Copies synthetic images from the source folder.
     - Splits the data into training and testing subsets based on the `train_ratio`.
  2. `extract_frames(video_path, tuning_output_dir, testing_output_dir, split_ratio, fps, saved_tuning_count, saved_testing_count)`:
     - Extracts frames from video files and splits them between tuning and testing directories.
  3. `copy_and_split_video_frames(video_path, direction, output_dir, fps, saved_tuning_count, saved_testing_count)`:
     - Combines video frame extraction and dataset splitting for real surgical videos.

#### 3. `finetuning.py`
- **Purpose**: Fine-tunes the segmentation model on the domain-adapted dataset.
- **Key Features**:
  - Loads domain-adapted datasets using the custom dataset class `SegmentationDatasetPNGAndHDF5`.
  - Fine-tunes a DeepLabV3+ model pre-trained on ImageNet.
  - Saves the fine-tuned model weights.
- **Key Components**:
  - **Dataset Class**: `SegmentationDatasetPNGAndHDF5` handles loading and preprocessing of PNG images and corresponding HDF5 masks.
  - **Training**:
    - Loss: CrossEntropyLoss
    - Optimizer: Adam (learning rate = 0.0001)
    - Epochs: 30
    - Batch Size: 64
  - **Output**: Fine-tuned model weights saved as `deeplabv3_model_adapted.pth`.

## Workflow Summary

### Data Preparation
1. **Synthetic Dataset**:
   - Synthetic images are organized into training and testing subsets using `copy_and_split_folder_with_images`.

2. **Real Surgical Videos**:
   - Frames are extracted from videos in the `vids_tune` folder using `extract_frames`.
   - Frames are split into training and testing datasets using `copy_and_split_video_frames`.

### Model Adaptation
1. **CycleGAN Setup and Training**:
   - Install CycleGAN and its dependencies:
     ```bash
     pip install torch torchvision pytorch-fid
     git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
     cd pytorch-CycleGAN-and-pix2pix
     pip install -r requirements/domain_adaptation.txt
     cd ..
     ```
   - Train CycleGAN for domain adaptation:
     ```bash
     python ./pytorch-CycleGAN-and-pix2pix/train.py --dataroot ./domain_adaptation/data --name synthetic2real --model cycle_gan --n_epochs 50 --n_epochs_decay 50 --batch_size 4 --netG resnet_9blocks
     ```
     The trained model is saved in `./domain_adaptation/checkpoints/synthetic2real/`.
   - Translate synthetic images to real-style images:
     ```bash
     python ./pytorch-CycleGAN-and-pix2pix/test.py --dataroot ./domain_adaptation/data --name synthetic2real --model cycle_gan --direction AtoB --dataset_mode unaligned --phase train --netG resnet_9blocks
     ```
     The translated images are saved in `./domain_adaptation/results/synthetic2real/train_latest/images/`.

2. **Model Fine-Tuning**:
   - The DeepLabV3+ segmentation model is fine-tuned using the domain-adapted datasets.
   - Training runs for 30 epochs with batch size 64, utilizing the Adam optimizer and CrossEntropyLoss.

3. **Output**:
   - The fine-tuned model is saved to `./domain_adaptation/deeplabv3_model_adapted.pth` for further evaluation.

## How to Run

### 1. Data Preparation
Run `adapting_with_cyclegan.py` to organize and prepare the datasets:
```bash
python ./domain_adaptation/adapting_with_cyclegan.py
```

### 2. CycleGAN Training and Translation
Train and use CycleGAN to translate synthetic images into real-style images:
```bash
# Train the CycleGAN model
python ./pytorch-CycleGAN-and-pix2pix/train.py --dataroot ./domain_adaptation/data --name synthetic2real --model cycle_gan --n_epochs 50 --n_epochs_decay 50 --batch_size 4 --netG resnet_9blocks

# Translate synthetic images to real-style images
python ./pytorch-CycleGAN-and-pix2pix/test.py --dataroot ./domain_adaptation/data --name synthetic2real --model cycle_gan --direction AtoB --dataset_mode unaligned --phase train --netG resnet_9blocks
```

### 3. Model Fine-Tuning
Run `finetuning.py` to fine-tune the segmentation model:
```bash
python ./domain_adaptation/finetuning.py -i <images_dir> -m <masks_dir>
```
- Replace `<images_dir>` with the path to the PNG images.
- Replace `<masks_dir>` with the path to the HDF5 masks.

## Outputs
- **Fine-Tuned Model**: `deeplabv3_model_adapted.pth`
- **Logs**: Training progress and loss values for each epoch.

## Notes
- Ensure the environment is set up with the necessary dependencies (e.g., PyTorch, segmentation-models-pytorch).
- Data paths for synthetic images and video frames must be correctly specified in the scripts or passed as arguments.


