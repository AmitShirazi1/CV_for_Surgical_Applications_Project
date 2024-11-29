from organizing_data import copy_and_split_folder_with_images, copy_and_split_video_frames

if __name__ == "__main__":
    copy_and_split_folder_with_images("./data_generation/output/jpg_format/", "./domain_adaptation/data/", "A", 0.7)
    saved_tuning_count, saved_testing_count = copy_and_split_video_frames("/datashare/project/vids_tune/20_2_24_1.mp4", "B")
    copy_and_split_video_frames("/datashare/project/vids_tune/4_2_24_B_2.mp4", "B", saved_tuning_count=saved_tuning_count, saved_testing_count=saved_testing_count)

""" Then, in the command line:

    To Install CycleGAN:
        pip install torch torchvision pytorch-fid
        git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
        cd pytorch-CycleGAN-and-pix2pix
        pip install -r requirements.txt

    To train the model:
        python ../pytorch-CycleGAN-and-pix2pix/train.py --dataroot ./data --name synthetic2real --model cycle_gan --n_epochs 50 --n_epochs_decay 50 --batch_size 4 --netG resnet_9blocks
    The model is saved in ./checkpoints/synthetic2real/

    To translate the synthetic images to real images:
        python ../pytorch-CycleGAN-and-pix2pix/test.py --dataroot ./data --name synthetic2real --model cycle_gan --direction AtoB --dataset_mode unaligned --phase train --netG resnet_9blocks
    The translated images are saved in ./results/synthetic2real/train_latest/images/

"""