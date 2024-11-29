import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse
from tqdm import tqdm
from train import SegmentationDataset


def main(args):
    # Create the dataset and dataloader
    dataset = SegmentationDataset(args.images_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Define the model, loss function, and optimizer
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=3)
    model.load_state_dict(torch.load(args.model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to the appropriate device (GPU or CPU)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Use Adam optimizer with learning rate 0.001

    # Training loop
    for epoch in range(10):
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

    # Save the new model to a file
    tuned_model_path = args.model_path.replace('.pth', '_tuned.pth')
    torch.save(model.state_dict(), tuned_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_path', type=str, default='./data_generation/output/hdf5_format/',
                        help='Path to hdf5 images directory')
    parser.add_argument('-m', '--model_path', type=str, default='./model_developement/deeplabv3_model.pth',
                        help='The path to the model to be fine-tuned. The tuned model will be saved in the same directory with the suffix "_tuned"')
    main(parser.parse_args())