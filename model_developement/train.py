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
import csv
from sklearn.metrics import jaccard_score


class SegmentationDataset(Dataset):
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


def calculate_metrics(outputs, masks):
    """Calculate performance metrics like IoU."""
    outputs = outputs.argmax(dim=1).cpu().numpy().flatten()  # Get predicted class
    masks = masks.cpu().numpy().flatten()  # Flatten true labels
    iou = jaccard_score(masks, outputs, average='macro')
    return iou


def main(args):
    # Create the dataset and dataloader
    dataset = SegmentationDataset(args.images_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    if args.val_images_path:
        val_dataset = SegmentationDataset(args.val_images_path)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Define the model, loss function, and optimizer
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to the appropriate device (GPU or CPU)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer with learning rate 0.001

    # Open a CSV file to log the training progress
    with open('./model_developement/training_logs.csv', 'w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Validation IoU"])  # Header row
        
        for epoch in range(50):
            # Training loop
            model.train()
            train_loss = 0.0
            for images, masks in tqdm(dataloader):
                images, masks = images.to(device), masks.to(device)
                # Forward pass
                outputs = model(images)  # Get model predictions
                loss = criterion(outputs, masks)  # Calculate the loss
                train_loss += loss.item()
                # Backward pass
                optimizer.zero_grad()
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters
            
            train_loss /= len(dataloader)
            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}")
            
            if (args.val_images_path) and (epoch % 5 == 0):
                # Validation loop
                model.eval()
                val_loss = 0.0
                val_iou = 0.0
            
                with torch.no_grad():
                    for images, masks in val_dataloader:
                        images, masks = images.to(device), masks.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                        val_loss += loss.item()
                        val_iou += calculate_metrics(outputs, masks)
                
                val_loss /= len(val_dataloader)
                val_iou /= len(val_dataloader)
            
                # Log the metrics
                log_writer.writerow([epoch + 1, train_loss, val_loss, val_iou])
                print(f"Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}")
        
        # Save the trained model
        torch.save(model.state_dict(), args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_path', type=str, default='./data_generation/output/hdf5_format/', help='Path to hdf5 images directory')
    parser.add_argument('-v', '--val_images_path', type=str, help='Path to hdf5 validation images directory')
    parser.add_argument('-m', '--model_path', type=str, default='./model_developement/deeplabv3_model.pth', help='Path to save the trained model')
    main(parser.parse_args())