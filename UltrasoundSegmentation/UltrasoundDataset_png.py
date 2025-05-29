import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

class UltrasoundDataset(Dataset):
    """
    Dataset class for ultrasound images, segmentations, and transformations.
    """

    def __init__(self, root_folder, imgs_dir="images", gts_dir="masks", tfms_dir="transforms", transform=None, out_channels=2, in_channels=1):
        self.transform = transform

        # Find all ultrasound image files in the input directory
        self.images = glob.glob(os.path.join(root_folder, "**", imgs_dir, "**", "*.png"), recursive=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __len__(self):
        """
        Returns the total number of ultrasound images in the dataset.
        
        Returns
        -------
        int
            Total number of ultrasound images in the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Find the data files that contain the index and return the image, 
        segmentation, and transform if they exist.

        Parameters
        ----------
        index : int
            Index of the image, segmentation, and transform to return

        Returns
        -------
        dict
            A dictionary containing the ultrasound image, segmentation mask, and transform matrix
        """
        name_mask = self.images[index].replace('/images/', '/masks/').replace('_ultrasound', '_segmentation')
        name_trans = self.images[index].replace('/images/', '/transforms/').replace('_ultrasound', '_transform').replace('.png', '.npy')
        file_name = os.path.basename(self.images[index])

        # Load the ultrasound image
        ultrasound_data = np.array(Image.open(self.images[index]).convert('L'))  # Convert to grayscale
        # Load the segmentation mask
        segmentation_data = np.array(Image.open(name_mask).convert('L'))// 255 # Convert to grayscale
        
        # if self.out_channels == 2:        
        #     segmentation_data = np.where(segmentation_data == 255, 0, segmentation_data)
        #     segmentation_data= np.where(segmentation_data == 128, 1, segmentation_data)
        # elif self.out_channels == 3:        
        #     segmentation_data = np.where(segmentation_data == 255, 2, segmentation_data)
        #     segmentation_data= np.where(segmentation_data == 128, 1, segmentation_data)
        # elif self.out_channels == 4:
        #     segmentation_data = np.where(segmentation_data == 255, 3, segmentation_data)
        #     segmentation_data= np.where(segmentation_data == 128, 2, segmentation_data)
        #     segmentation_data= np.where(segmentation_data == 64, 1, segmentation_data)
            # print(f"ultrasound_data: {ultrasound_data.dtype}, segmentation_data: {segmentation_data.dtype}")
        try:
            # Load the transform matrix
            transform_data = np.load(name_trans)
        except:
            transform_data = np.identity(4)
        
        # If ultrasound_data is 2D, add a channel dimension as the last dimension
        if len(ultrasound_data.shape) == 2:
            ultrasound_data = np.expand_dims(ultrasound_data, axis=-1)  # Add channel dimension at the beginning
        
        # If segmentation_data is 2D, add a channel dimension as the last dimension
        if len(segmentation_data.shape) == 2:
            segmentation_data = np.expand_dims(segmentation_data, axis=-1)  # Add channel dimension at the beginning
        
        if self.in_channels == 3:
            ultrasound_data = np.concatenate([ultrasound_data,ultrasound_data,ultrasound_data], axis=2)
        # # Convert to torch tensors
        # ultrasound_data = torch.tensor(ultrasound_data, dtype=torch.float32)
        # segmentation_data = torch.tensor(segmentation_data, dtype=torch.float32)
        # transform_data = torch.tensor(transform_data, dtype=torch.float32)

        data = {
            "image": ultrasound_data,
            "label": segmentation_data,
            "transform": transform_data,
            "filename": os.path.splitext(os.path.basename(self.images[index]))[0]

        }

        if self.transform:
            data = self.transform(data)

        return data