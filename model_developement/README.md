# Phase 2: Segmentation Model Development for Surgical Instruments

This phase of the project focuses on training and fine-tuning a DeepLabV3+ segmentation model to detect surgical instruments in synthetic images. The model is trained using synthetic data generated in Phase 1, evaluated with a validation dataset, and prepared for further domain adaptation in Phase 3.

## Project Structure

The code for this phase is located in the `./model_developement/` directory and includes the following files:

### 1. `train.py`
This script trains a **DeepLabV3+** segmentation model and logs the training progress (optional), including loss and Intersection over Union (IoU) metrics, to a CSV file. Key steps in this script include:
- **Dataset Handling:** Loads the synthetic dataset (HDF5 format) with a custom `SegmentationDataset` class.
- **Model Selection:** The model is based on the **DeepLabV3+** architecture with a **ResNet-50** backbone pre-trained on ImageNet.
- **Training Loop:** The model is trained for 50 epochs, and every 5 epochs, validation is performed on a separate dataset. Loss and IoU metrics are logged to a CSV file.
- **Output:** The trained model is saved as a `.pth` file after the final epoch.

**Command-line Arguments:**
- `--images_path (-i)`: Path to the directory containing HDF5 images and masks for training data.
- `--val_images_path (-v)`: Path to the directory containing HDF5 images and masks for validation data. If not specified, the model only trains without evaluation.
- `--model_path (-m)`: Path to save the trained model. (Default: `./model_developement/deeplabv3_model.pth`)

### 2. `finetuning.py`
This script fine-tunes the trained model with a smaller learning rate, refining its performance on the synthetic dataset. It includes:

- **Model Initialization:** Loads the pre-trained DeepLabV3+ model and its weights.
- **Fine-tuning Pipeline:** Fine-tunes the model for an additional 10 epochs, using the same dataset and loss function but a lower learning rate for optimization.
- **Output:** Saves the fine-tuned model with the `_tuned` suffix added to the model's filename.

**Command-line Arguments:**
- `--images_path (-i)`: Path to the directory containing HDF5 images and masks. (Default: `./data_generation/output/hdf5_format/`)
- `--model_path (-m)`: Path to the model to be fine-tuned. The tuned model is saved in the same directory with the suffix `_tuned.pth`. (Default: `./model_developement/deeplabv3_model.pth`)

### Log Output
The training progress, including **train loss**, **validation loss**, and **validation IoU**, is saved in the `./model_developement/training_logs.csv` file for analysis.

## Dependencies
Don't forget to install required packages using:
   ```bash
   pip install -r requirements/model_developement.txt
   ```

**Key Libraries:**
- `segmentation-models-pytorch`
- `torch`
- `torchvision`
- `h5py`
- `tqdm`
- `scikit-learn`

## Usage

### Training the Model
Run the `train.py` script to train the segmentation model on the synthetic dataset and log metrics.

```bash
python train.py --images_path ./data_generation/output/hdf5_format/ --val_images_path ./data_generation/output_val/hdf5_format/ --model_path ./model_developement/deeplabv3_model.pth
```

### Fine-tuning the Model
Run the `finetuning.py` script to fine-tune the pre-trained model for additional epochs (optional).

```bash
python finetuning.py --images_path ./data_generation/output/hdf5_format/ --model_path ./model_developement/deeplabv3_model.pth
```

## Outputs

1. **Trained Model:** Saved as `deeplabv3_model.pth` in the `./model_developement/` directory.
2. **Fine-tuned Model:** Saved as `deeplabv3_model_tuned.pth` in the same directory.

## Notes

- Ensure the synthetic dataset is available in HDF5 format at the specified path (`./data_generation/output/hdf5_format/`).
- Training and fine-tuning require a GPU for efficient computation. The scripts automatically detect and utilize available GPUs.
