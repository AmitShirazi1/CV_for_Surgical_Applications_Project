from organizing_data import copy_and_split_folder_with_images, copy_and_split_video_frames

if __name__ == "__main__":
    copy_and_split_folder_with_images("./data_generation/output/jpg_format/", "./domain_adaptation/data/", "A", 0.7)
    saved_tuning_count, saved_testing_count = copy_and_split_video_frames("/datashare/project/vids_tune/20_2_24_1.mp4", "B")
    copy_and_split_video_frames("/datashare/project/vids_tune/4_2_24_B_2.mp4", "B", saved_tuning_count=saved_tuning_count, saved_testing_count=saved_testing_count)
    