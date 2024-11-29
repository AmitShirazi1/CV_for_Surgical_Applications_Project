# Phase 2: Segmentation Model Development for Surgical Instruments

This phase of the project focuses on training and fine-tuning a DeepLabV3+ segmentation model to detect surgical instruments in synthetic images. The model is trained using synthetic data generated in Phase 1 and prepared for further domain adaptation in Phase 3.

## Project Structure

The code for this phase is located in the `./model_developement/` directory and includes the following files:

### 1. `train.py`
This script trains a DeepLabV3+ segmentation model using the synthetic dataset. It includes:

- **Dataset Handling:** A custom `SegmentationDataset` class that loads HDF5 files containing images and segmentation masks. It preprocesses the images with transformations like resizing, normalization, and tensor conversion.
- **Model Selection:** Uses the DeepLabV3+ architecture with a ResNet-50 backbone pre-trained on ImageNet.
- **Training Pipeline:** Implements a training loop with CrossEntropy loss and the Adam optimizer, training the model for 50 epochs.
- **Output:** Saves the trained model weights to the specified path.

**Command-line Arguments:**
- `--images_path (-i)`: Path to the directory containing HDF5 images and masks. (Default: `./data_generation/output/hdf5_format/`)
- `--model_path (-m)`: Path to save the trained model. (Default: `./model_developement/deeplabv3_model.pth`)

### 2. `finetuning.py`
This script fine-tunes the trained model with a smaller learning rate, refining its performance on the synthetic dataset. It includes:

- **Model Initialization:** Loads the pre-trained DeepLabV3+ model and its weights.
- **Fine-tuning Pipeline:** Fine-tunes the model for an additional 10 epochs, using the same dataset and loss function but a lower learning rate for optimization.
- **Output:** Saves the fine-tuned model with the `_tuned` suffix added to the model's filename.

**Command-line Arguments:**
- `--images_path (-i)`: Path to the directory containing HDF5 images and masks. (Default: `./data_generation/output/hdf5_format/`)
- `--model_path (-m)`: Path to the model to be fine-tuned. The tuned model is saved in the same directory with the suffix `_tuned.pth`. (Default: `./model_developement/deeplabv3_model.pth`)

## Dependencies
Don't forget to install required packages using:
   ```bash
   pip install -r requirements.txt
   ```

**Key Libraries:**
- `segmentation-models-pytorch`
- `torch`
- `torchvision`
- `h5py`
- `tqdm`

## Usage

### Training the Model
Run the `train.py` script to train the segmentation model on the synthetic dataset.

```bash
python train.py --images_path ./data_generation/output/hdf5_format/ --model_path ./model_developement/deeplabv3_model.pth
```

### Fine-tuning the Model
Run the `finetuning.py` script to fine-tune the pre-trained model for additional epochs.

```bash
python finetuning.py --images_path ./data_generation/output/hdf5_format/ --model_path ./model_developement/deeplabv3_model.pth
```

## Outputs

1. **Trained Model:** Saved as `deeplabv3_model.pth` in the `./model_developement/` directory.
2. **Fine-tuned Model:** Saved as `deeplabv3_model_tuned.pth` in the same directory.

## Notes

- Ensure the synthetic dataset is available in HDF5 format at the specified path (`./data_generation/output/hdf5_format/`).
- Training and fine-tuning require a GPU for efficient computation. The scripts automatically detect and utilize available GPUs.
