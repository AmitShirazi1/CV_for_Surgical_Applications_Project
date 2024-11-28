import segmentation_models_pytorch as smp
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import argparse
import glob
import h5py
from tqdm import tqdm


class SegmentationDatasetLongTweezers(Dataset):
    """
    A custom Dataset class for loading and transforming segmentation datasets stored in HDF5 files.

    Attributes:
        images_paths (list): A sorted list of paths to the HDF5 files containing the images and masks.
        images_transform (Compose): A torchvision transforms pipeline for preprocessing the images.
        masks_transform (Compose): A torchvision transforms pipeline for preprocessing the masks.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Loads and returns the image and mask at the specified index after applying the transformations.
    """
    def __init__(self, images_path):
        # Initialize the dataset with the path to the images
        self.images_paths = sorted(
                                    glob.glob(f"{images_path}/*.hdf5"),
                                    key=lambda x: int(x.split('/')[-1].split('.')[0])
                                    )
        # define the transformations
        self.images_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
        self.masks_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((256, 256))])

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # Load the image and mask from the HDF5 file at the given index
        with h5py.File(self.images_paths[idx], "r") as file:
            image = file['colors'][:]
            mask = file['category_id_segmaps'][:]
        # Apply the transformations to the image and mask
        image = self.images_transform(image)
        mask = self.masks_transform(mask)[0]
        return image, mask


def main(args):
    # Create the dataset and dataloader
    dataset = SegmentationDatasetLongTweezers(args.images_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Define the model, loss function, and optimizer
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=3)
    tuned_model_path = "./model_developement/deeplabv3_model_long_tweezers.pth"
    model.load_state_dict(torch.load(tuned_model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to the appropriate device (GPU or CPU)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Use Adam optimizer with learning rate 0.001

    # Training loop
    for epoch in range(10):  # Train for 25 epochs
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
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save the trained model to a file
    tuned_model_path2 = "./model_developement/deeplabv3_model_red.pth"
    torch.save(model.state_dict(), tuned_model_path2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_path', type=str, default='./data_generation/output_red/hdri_background/hdf5_format/', help='Path to hdf5 images directory')
    main(parser.parse_args())