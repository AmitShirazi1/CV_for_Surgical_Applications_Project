import os
import segmentation_models_pytorch as smp
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import argparse
import glob
from tqdm import tqdm
import h5py
from PIL import Image


def copy_adaptation_results(src_folder, dest_folder):
    """
    Copy the results of the CycleGAN adaptation process to a new folder.
    
    Args:
        src_folder (str): Path to the source folder containing the CycleGAN results.
        dest_folder (str): Path to the destination folder.
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    # Copy the results of the CycleGAN adaptation process
    for file_name in os.listdir(src_folder):
        shutil.copy2(os.path.join(src_folder, file_name), os.path.join(dest_folder, file_name))
    
    print(f"CycleGAN adaptation results copied to '{dest_folder}'")


class SegmentationDatasetPNGAndHDF5(Dataset):
    """
    A custom Dataset class for loading PNG images and HDF5 masks.

    Attributes:
        images_paths (list): A sorted list of paths to the PNG image files.
        masks_paths (list): A sorted list of paths to the HDF5 mask files.
        images_transform (Compose): A torchvision transforms pipeline for preprocessing the images.
        masks_transform (Compose): A torchvision transforms pipeline for preprocessing the masks.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Loads and returns the image and mask at the specified index after applying the transformations.
    """
    def __init__(self, images_dir, masks_dir):
        # Initialize the dataset with paths to the PNG images and HDF5 masks
        self.images_paths = sorted(glob.glob(f"{images_dir}/*.png"))
        self.masks_paths = [os.path.join(masks_dir, f"{int(os.path.basename(image_path).split('_')[0])}.hdf5") for image_path in self.images_paths]

        # Define the transformations
        self.images_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.masks_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

    def __len__(self):
        # Ensure the number of images matches the number of masks
        return min(len(self.images_paths), len(self.masks_paths))

    def __getitem__(self, idx):
        # Load the image from the PNG file
        image = Image.open(self.images_paths[idx]).convert("RGB")

        # Load the mask from the HDF5 file
        with h5py.File(self.masks_paths[idx], "r") as file:
            mask = file['category_id_segmaps'][:]

        # Apply the transformations
        image = self.images_transform(image)
        mask = self.masks_transform(mask)[0].long()  # Ensure mask is in long format for CrossEntropyLoss

        return image, mask


def main(args):
    # Create the dataset and dataloader
    dataset = SegmentationDatasetPNGAndHDF5(args.images_dir, args.masks_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Define the model, loss function, and optimizer
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=3)
    tuned_model_path = "./model_developement/deeplabv3_model_long_tweezers.pth"
    model.load_state_dict(torch.load(tuned_model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to the appropriate device (GPU or CPU)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Use Adam optimizer with learning rate 0.0001

    # Training loop
    for epoch in range(10):  # Train for 10 epochs
        model.train()  # Set the model to training mode
        for images, masks in tqdm(dataloader):  # Iterate over batches of images and masks
            images, masks = images.to(device), masks.to(device)  # Move images and masks to the appropriate device
            # Forward pass
            outputs = model(images)  # Get model predictions
            loss = criterion(outputs, masks)  # Calculate the loss
            # Backward pass
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model parameters

        # Print the loss for the current epoch
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Save the trained model to a file
    tuned_model_path2 = "./deeplabv3_model_adapted.pth"
    torch.save(model.state_dict(), tuned_model_path2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_dir', type=str, required=True, default="./results/synthetic2real/train_30/images",
                        help='Path to directory containing the PNG images')
    parser.add_argument('-m', '--masks_dir', type=str, required=True, default="../data_generation/output_objects/hdri_background/hdf5_format",
                        help='Path to directory containing the HDF5 masks')
    main(parser.parse_args())


