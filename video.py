import cv2
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import numpy as np
import argparse
import os

DEV_OUTPUT_PATH = "./model_developement/output/"
TEST_OUTPUT_PATH = "./domain_adaptation/output/"


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


# Function to process video frames
def predict_video(video_path, output_dir):
    model_path = "./model_developement/deeplabv3_model_1000data.pth"
    output_path = os.path.join(output_dir, "video_pred.mp4")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",  # Pretrained on ImageNet
        in_channels=3,
        classes=3  # Number of output classes
    )

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))

    # Switch to evaluation mode
    model.eval()

    # Define the image transformation
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define custom colors for each category (as BGR tuples)
    CATEGORY_COLORS = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 0, 255),
    }

    # Open the video file
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Process each frame
    for frame_idx in range(total_frames):
        ret, frame = video_cap.read()
        if not ret:
            break

        # Convert frame to PIL image and preprocess
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0)

        input_batch = input_batch.to(device)

        # Predict segmentation
        with torch.no_grad():
            output = model(input_batch)[0]
        softed_output = torch.nn.functional.softmax(output, dim=0)
        # Find the maximum value along dimension 0 (across the first axis)
        max_values, argmax_indices = softed_output.max(dim=0)
        max_values = max_values.cpu().numpy()

        # Step 2: Check if the max values are greater than or equal to 0.7
        mask = max_values >= 0.5  # TODO: Check this value.

        # Step 3: Set argmax indices to 0 where max value is less than 0.7
        output_predictions = argmax_indices.cpu() * np.long(mask)


        # # Generate segmentation mask (output_predictions is the predicted class map)
        # output_predictions = output.argmax(0).cpu().numpy()

        # Apply custom colormap with transparency
        color_mask = apply_custom_colormap_with_transparency(output_predictions, CATEGORY_COLORS)

        # Resize the mask to match the frame size (if necessary)
        color_mask = cv2.resize(color_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # Overlay the segmentation mask on the original frame
        # Separate the RGB and Alpha channels
        overlay = frame.copy()
        alpha_mask = color_mask[:, :, 3] / 255.0
        for c in range(3):  # Blend each channel
            overlay[:, :, c] = overlay[:, :, c] * (1 - alpha_mask) + color_mask[:, :, c] * alpha_mask

        # Write the frame to the output video
        video_writer.write(overlay)
        print(f"Processed frame {frame_idx + 1}/{total_frames}")

    # Release resources
    video_cap.release()
    video_writer.release()
    print(f"Output video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--input_video_path", type=str, default="/datashare/HW1/ood_video_data/surg_1.mp4",
                        help="Path to input video.\n\
                        For tunning, choose a file from dir: '/datashare/project/vids_tune/'.\n\
                        For testing, choose a file from dir: '/datashare/project/vids_test/'.")
    # For trying on a short video: /datashare/HW1/ood_video_data/surg_1.mp4
    parser.add_argument("-o", "--output_dir", type=str, help="Path to the directory where the output video will be saved.\n\
                        For example: f{DEV_OUTPUT_PATH}.")
    parser.add_argument("--dev", action="store_true", help="Use this flag to run the script in development mode.\
                        This will use the output directory: f{DEV_OUTPUT_PATH}")
    parser.add_argument("--test", action="store_true", help="Use this flag to run the script in test mode.\
                        This will use the output directory: f{TEST_OUTPUT_PATH}")
    args = parser.parse_args()

    if args.input_video_path is None:
        print("Please provide the path to the input video using the '-v' flag.")
        exit(1)
    if (args.output_dir is None) and (args.dev is False) and (args.test is False):
        print("Please provide an output directory using the '-o' flag, or use --dev or --test.")
        exit(1)
    if sum([bool(args.output_dir), args.dev, args.test]) > 1:
        print("Please use only one of the '--dev', '--test' flags, or provide an output directory using the '-o' flag.")
        exit(1)
    output_dir = args.output_dir if args.output_dir else DEV_OUTPUT_PATH if args.dev else TEST_OUTPUT_PATH

    predict_video(args.input_video_path, output_dir)
