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
import sys

# Add the parent folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predict import predict


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
    def __init__(self, images_dir, masks):
        # Initialize the dataset with paths to the PNG images and HDF5 masks
        self.images_paths = sorted(glob.glob(f"{images_dir}/*fake_B.png"))
        self.masks = masks

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
        return min(len(self.images_paths), len(self.masks))

    def __getitem__(self, idx):
        # Load the image from the PNG file
        image = Image.open(self.images_paths[idx]).convert("RGB")

        # Apply the transformations
        image = self.images_transform(image)
        mask = self.masks[idx].long()  # Ensure mask is in long format for CrossEntropyLoss

        return image, mask


def main(args):
    original_images = sorted(glob.glob(f"{args.images_dir}/*real_A.png"))
    masks = [predict(image_path, None, args.models_path, save_output=False) for image_path in original_images]
    # Create the dataset and dataloader
    dataset = SegmentationDatasetPNGAndHDF5(args.images_dir, masks)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Define the model, loss function, and optimizer
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=3)
    model.load_state_dict(torch.load(args.models_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to the appropriate device (GPU or CPU)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Use Adam optimizer with learning rate 0.0001

    # Training loop
    for epoch in range(30):  # Train for 10 epochs
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
    tuned_model_path2 = "./domain_adaptation/deeplabv3_model_adapted.pth"
    torch.save(model.state_dict(), tuned_model_path2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_dir', type=str, default="./domain_adaptation/results/synthetic2real/train_latest/images",
                        help='Path to directory containing the PNG images')
    parser.add_argument('-m', '--models_path', type=str, default="./model_developement/deeplabv3_model.pth",
                        help='Path to directory containing the trained model')
    main(parser.parse_args())


