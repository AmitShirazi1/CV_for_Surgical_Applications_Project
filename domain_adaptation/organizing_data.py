import os
import shutil
import random
import argparse
import cv2


def copy_and_split_folder_with_images(src_folder, dest_folder, direction="A", train_ratio=0.8):
    """
    Copies content from src_folder to dest_folder, splits into train/test subfolders.
    
    Args:
        src_folder (str): Path to the source folder.
        dest_folder (str): Path to the destination folder.
        train_ratio (float): Proportion of files to go into the "train" folder.
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    # Create "train" and "test" subfolders
    train_folder = os.path.join(dest_folder, f"train{direction}")
    test_folder = os.path.join(dest_folder, f"test{direction}")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Get all files from the source folder
    files = [f for f in os.listdir(src_folder)]
    
    # Shuffle files for random distribution
    random.shuffle(files)
    
    # Split files into train and test
    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    
    def copy_files(files, dest_folder):
        # Copy files to the respective folders
        for file_name in files:
            new_name = str(len(os.listdir(dest_folder))) + '.jpg'
            if not file_name.split('_')[-1].startswith('color'):
                continue
            shutil.copy2(os.path.join(src_folder, file_name), os.path.join(dest_folder, new_name))

    copy_files(train_files, train_folder)
    copy_files(test_files, test_folder)
    
    print(f"Files copied and split into '{train_folder}' and '{test_folder}'")


def extract_frames(video_path, tuning_output_dir, testing_output_dir, split_ratio=0.8, fps=1, saved_tuning_count=0, saved_testing_count=0):
    """
    Extract frames from a video and split them between tuning and testing directories.

    Args:
        video_path (str): Path to the input video file.
        tuning_output_dir (str): Path to the directory for tuning frames.
        testing_output_dir (str): Path to the directory for testing frames.
        split_ratio (float): Proportion of frames to save in tuning directory.
        fps (int): Number of frames per second to extract.
    """    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    # Get video properties
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)  # Interval between frames to capture

    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of video

        # Save every n-th frame
        if frame_count % frame_interval == 0:
            # Determine whether this frame goes to tuning or testing
            total_frames = saved_tuning_count + saved_testing_count
            if total_frames % (10 * split_ratio) != 0:
                output_dir = tuning_output_dir
                frame_filename = os.path.join(output_dir, f"frame_{saved_tuning_count:04d}.jpg")
                saved_tuning_count += 1
            else:
                output_dir = testing_output_dir
                frame_filename = os.path.join(output_dir, f"frame_{saved_testing_count:04d}.jpg")
                saved_testing_count += 1

            # Save the frame
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    video.release()
    print(f"Extracted {saved_tuning_count} frames of video {video_path.split('/')[-1]} to {tuning_output_dir}")
    print(f"Extracted {saved_testing_count} frames of video {video_path.split('/')[-1]} to {testing_output_dir}")
    return saved_tuning_count, saved_testing_count


def copy_and_split_video_frames(video_path, direction="B", output_dir="./data/", fps=2, saved_tuning_count=0, saved_testing_count=0):
    os.makedirs(output_dir, exist_ok=True)
    # Ensure output directories exist
    tuning_output_dir = os.path.join(output_dir, f"train{direction}")
    os.makedirs(tuning_output_dir, exist_ok=True)
    testing_output_dir = os.path.join(output_dir, f"test{direction}")
    os.makedirs(testing_output_dir, exist_ok=True)
    return extract_frames(video_path, tuning_output_dir, testing_output_dir, split_ratio=0.7, fps=fps, saved_tuning_count=saved_tuning_count, saved_testing_count=saved_testing_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', default="../data_generation/output_objects/hdri_background/jpg_format/")
    parser.add_argument('-d', '--destination', default="./data/")
    parser.add_argument('-r', '--train_ratio', default=0.8)
    args = parser.parse_args()
    copy_and_split_folder_with_images(args.source, args.destination, "A", float(args.train_ratio))
