import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import glob

# Dataset for CARLA (H5 format)
class CarlaH5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        # Open in read mode; note that HDF5 is picky about multi-threading
        with h5py.File(file_path, "r") as f:
            # get the number of images in the h5 file
            self.data_len = len(f["rgb"])

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        with h5py.File(self.file_path, "r") as f:
            # load the image and steering angle
            image = f["rgb"][index]
            targets = f["targets"][index]
            # targets has steer, brake, throttle, etc
        
        # if type(image) == torch.Tensor:
        #     print("Image shape right now: ", image.shape)

        return image, targets


## Use glob to get all the h5 file paths at the same time
class CarlaH5GlobalDataset(Dataset):
    def __init__(self, h5_dir):
        # Gather ALL the h5 files at once, and store them in some location
        ## SORT the file lists by name so that they are in order (guaranteed)
        self.file_paths = sorted(glob.glob(f"{h5_dir}/*.h5"))
        ## test read one file
        # with h5py.File(self.file_paths[0], "r") as f:
        #     self.frames_per_file = len(f["rgb"]) # most likely 200
        self.frames_per_file = 200
        print("len of file_paths: ", len(self.file_paths)) # 3289
        self.total_files = 1000
        print("Len of total files used to train: ", self.total_files)

        # LOAD ALL FILES here
        self.all_images = []
        self.all_targets = []

        print("Loading all the data into RAM...") # hope the RAM doesn't collapse
        ## OS Concept -> whereby RAM data can be accessed much faster
        for path in self.file_paths[:self.total_files]:
            with h5py.File(path, "r") as f:
                # Load everything into numpy arrays immediately
                # do processing + reshaping here
                image = f["rgb"] # like (200, 88, 66, 3)
                #print("Image shape right now: ", image.shape)

                # Slicing here
                image = image[:, 11:-11, :, :] # crop image (in the HEIGHT axis)
                image = torch.from_numpy(image).permute(0, 3, 1, 2).float() / 255.0 # permute to CHW

                #print("Image shape after reshaping: ", image.shape) # (200, 3, 66, 200)
                self.all_images.append(image[:])
                self.all_targets.append(f["targets"][:])

        # Concatenate into giant arrays: [Total_Frames, H, W, C]
        self.all_images = np.concatenate(self.all_images, axis=0)
        self.all_targets = np.concatenate(self.all_targets, axis=0)

        print("Finished loading all the data into RAM.")

        print("Image shape right now: ", self.all_images.shape)
        print("Targets shape right now: ", self.all_targets.shape)



    
    def __len__(self):
        #return self.frames_per_file * len(self.file_paths)
        return self.frames_per_file * self.total_files

    def __getitem__(self, index):
        # file_idx = index // self.frames_per_file
        # frame_idx = index % self.frames_per_file 
        # no file loading here for now to save OS resources
        # with h5py.File(self.file_paths[file_idx], "r") as f:
        #     image = f["rgb"][frame_idx]
        #     targets = f["targets"][frame_idx]
        img = self.all_images[index]
        targets = self.all_targets[index][:2]

        # Preprocessing (Crop, Permutate, Normalize)
        #print("Image shape right now: ", img.shape)
        return img, torch.tensor(targets)

## Use glob to get all the h5 file paths at the same time
class CarlaH5GlobalDatasetWithSequence(Dataset):
    def __init__(self, h5_dir, total_files=1000, sequence_len=1, stride=3):
        # Gather ALL the h5 files at once, and store them in some location
        ## SORT the file lists by name so that they are in order (guaranteed)
        self.file_paths = sorted(glob.glob(f"{h5_dir}/*.h5"))
        ## test read one file
        # with h5py.File(self.file_paths[0], "r") as f:
        #     self.frames_per_file = len(f["rgb"]) # most likely 200
        self.frames_per_file = 200
        self.sequence_len = sequence_len
        self.stride = stride
        self.total_files = total_files

        print("len of file_paths: ", len(self.file_paths)) # 3289
        print("Len of total files used to train: ", self.total_files)

        # LOAD ALL FILES here
        self.all_images = []
        self.all_targets = []

        print("Loading all the data into RAM...") # hope the RAM doesn't collapse
        ## OS Concept -> whereby RAM data can be accessed much faster
        for path in self.file_paths[:self.total_files]:
            with h5py.File(path, "r") as f:
                # Load everything into numpy arrays immediately
                # do processing + reshaping here
                image = f["rgb"] # like (200, 88, 66, 3)
                #print("Image shape right now: ", image.shape)

                # Slicing here
                image = image[:, 11:-11, :, :] # crop image (in the HEIGHT axis)
                image = torch.from_numpy(image).permute(0, 3, 1, 2).float() / 255.0 # permute to CHW

                #print("Image shape after reshaping: ", image.shape) # (200, 3, 66, 200)
                self.all_images.append(image[:])
                self.all_targets.append(f["targets"][:])

        # Concatenate into giant arrays: [Total_Frames, H, W, C]
        self.all_images = np.concatenate(self.all_images, axis=0)
        self.all_targets = np.concatenate(self.all_targets, axis=0)

        print("Finished loading all the data into RAM.")

        print("Image shape right now: ", self.all_images.shape)
        print("Targets shape right now: ", self.all_targets.shape)



    
    def __len__(self):
        #return self.frames_per_file * len(self.file_paths)
        #return self.frames_per_file * self.total_files
        return len(self.all_images)# - (self.sequence_len - 1) * self.stride

    def __getitem__(self, index):
        # Ensure no look back at past the start of the data
        # If index is too small, repeat the first frame
        start_offset = (self.sequence_len - 1) * self.stride

        if index < start_offset:
            # "Cold start" for the beginning of the dataset
            indices = [0] * self.sequence_len  # literally just use the first one
        else:
            # Look BACKWARDS: e.g., [index-4, index-2, index]
            indices = [index - (self.sequence_len - 1 - i) * self.stride for i in range(self.sequence_len)]
        
        # Stack images: (9, 66, 200)
        imgs = self.all_images[indices]
        stacked_img = np.concatenate([imgs[i] for i in range(self.sequence_len)], axis=0)
        
        # Target is the control for the CURRENT frame (index)
        targets = self.all_targets[index][:2] 

        return stacked_img, torch.tensor(targets) # targets