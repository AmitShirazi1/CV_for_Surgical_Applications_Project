import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import argparse
import os
import sys

DEV_OUTPUT_PATH = "./model_developement/output_objects/"
TEST_OUTPUT_PATH = "./domain_adaptation/output_objects/"

# Function to predict on a new image
def predict(image_path, output_dir):
    model_path = "./model_developement/deeplabv3_model_1000data.pth"
    output_path = os.path.join(output_dir, "image_pred.jpg")
    os.makedirs(output_dir, exist_ok=True)

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
    
    # Load and preprocess the image
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)[0]
    softed_output = torch.nn.functional.softmax(output, dim=0)
    # Find the maximum value along dimension 0 (across the first axis)
    max_values, argmax_indices = softed_output.max(dim=0)

    # Check if the max values are greater than or equal to 0.7
    mask = max_values >= 0  # TODO: Check this value.

    # Set argmax indices to 0 where max value is less than 0.7
    output_predictions = argmax_indices * mask.long()
    print("\n", output_predictions)
    print("\n", output_predictions.max())
    print("\n", output_predictions.unique(), "\n")

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title('Input Image')
    plt.subplot(1, 2, 2)
    plt.imshow(output_predictions.cpu().numpy())
    plt.title('Predicted Segmentation')
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_idx", type=int, help="Index of the image to predict.\
                        For example: '-i 0' will predict on the first image.")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to the directory where the output image will be saved.\n\
                        For example: f{DEV_OUTPUT_PATH}")
    parser.add_argument("--dev", action="store_true", help="Use this flag to run the script in development mode.\
                        This will use the output directory: f{DEV_OUTPUT_PATH}")
    parser.add_argument("--test", action="store_true", help="Use this flag to run the script in test mode.\
                        This will use the output directory: f{TEST_OUTPUT_PATH}")
    args = parser.parse_args()

    image_path = f"./data_generation/output_objects/hdri_background/jpg_format/{args.image_idx}_color.jpg"

    if (args.output_dir is None) and (args.dev is False) and (args.test is False):
        print("Please provide an output directory using the '-o' flag, or use --dev or --test.")
        sys.exit(1)
    if sum([bool(args.output_dir), args.dev, args.test]) > 1:
        print("Please use only one of the '--dev', '--test' flags, or provide an output directory using the '-o' flag.")
        sys.exit(1)
    output_dir = args.output_dir if args.output_dir else DEV_OUTPUT_PATH if args.dev else TEST_OUTPUT_PATH
    print("output_dir:", output_dir)

    predict(image_path, output_dir)