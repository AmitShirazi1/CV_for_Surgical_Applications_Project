import segmentation_models_pytorch as smp
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


def main(args):
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SegmentationDataset(args.image_dir, args.mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # Define model, loss function, and optimizer
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=3)
    model = model.to("cuda")  # Use GPU if available

    criterion = nn.CrossEntropyLoss()  # For multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):  # Number of epochs
        model.train()
        for images, masks in dataloader:
            images, masks = images.to("cuda"), masks.to("cuda")

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs["out"], masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "deeplabv3_model.pth")

if __name__ == "main":
    main()