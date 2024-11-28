import cv2
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import numpy as np
import os
import h5py


OUTPUT_PATH = "./domain_adaptation/frame_predictions/"

CATEGORY_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
}


def apply_custom_colormap_with_transparency(mask, category_colors):
    """ Map each category to a color and set the background transparent. """
    height, width = mask.shape
    color_mask = np.zeros((height, width, 4), dtype=np.uint8)  # Add alpha channel (RGBA)

    for category, color in category_colors.items():
        if category == 0:
            continue  # Skip category 0 (leave transparent)
        color_mask[mask == category, :3] = color  # Set RGB color
        color_mask[mask == category, 3] = 255    # Fully opaque (not transparent)

    return color_mask


def load_pretrained_model(model_path):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",  # Pretrained on ImageNet
        in_channels=3,
        classes=3  # Number of output classes
    )

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))
    return model


def create_folder_for_predictions(output_path):
    pred_folder_path = os.path.join(output_path, 'frame_predictions')
    os.makedirs(pred_folder_path, exist_ok=True)
    return pred_folder_path


def postprocess_output(output, original_size):
    output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    output = cv2.resize(output.astype(np.uint8), original_size)  # Resize back to original size
    return output


# Function to process video frames
def predict_video(video_path, output_dir, model_path):

    pred_folder_path = create_folder_for_predictions(output_dir)

    model = load_pretrained_model(model_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    video_cap = cv2.VideoCapture(video_path)

    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Define the image transformation
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame_count = 0
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            output = model(input_batch)[0]
        softed_output = torch.nn.functional.softmax(output, dim=0)
        max_values, argmax_indices = softed_output.max(dim=0)
        max_values = max_values.cpu().numpy()
        mask = max_values >= 0  # TODO: Check this value.
        output_predictions = argmax_indices.cpu() * np.long(mask)

        segmentation = postprocess_output(output_predictions, (width, height))

        hdf5_file_path = os.path.join(pred_folder_path, f"frame_{frame_count}.h5")
        with h5py.File(hdf5_file_path, "w") as hdf5_file:
            hdf5_file.create_dataset("frame", data=frame, compression="gzip")
            hdf5_file.create_dataset("segmentation", data=segmentation, compression="gzip")
        frame_count += 1

    video_cap.release()

    # Get video properties
    

    # Create a video writer for output
      # Codec for .mp4

    # Move model to GPU if available


if __name__ == "__main__":
    output_dir = OUTPUT_PATH
    input_video_path = '/datashare/HW1/ood_video_data/surg_1.mp4'
    model_path = 'model_developement/deeplabv3_model_long_tweezers.pth'

    predict_video(input_video_path, output_dir, model_path)
