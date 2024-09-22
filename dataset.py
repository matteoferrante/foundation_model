import torch
import numpy as np
import glob
import os
from os.path import join as opj
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import tqdm




class TimeSeriesDataset(Dataset):
    """TimeSeriesDataset is a custom PyTorch Dataset for loading time series data from .npy files.
    Attributes:
        scaling (int): Scaling factor for the time series data. Default is 60.
    Methods:
        __len__():
            Returns the number of files in the dataset.
        __getitem__(idx):
                idx (int): Index of the file to retrieve.
            Returns:
                tuple: A tuple containing:
                    - timeseries_tensor (torch.Tensor): The time series data as a PyTorch tensor with shape (n_timepoints, n_features).
                    - label (int): The label extracted from the file name.

    """
    def __init__(self, file_list, scaling = 60):
        """
        Args:
            file_list (list of str): List of file paths to the .npy files.
        """
        self.file_list = file_list
        self.scaling = scaling

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the file path
        file_path = self.file_list[idx]
        
        # Load the timeseries from .npy file
        timeseries = np.load(file_path)
        
        # Convert to PyTorch tensor to have shape (n_timesteps, n_features) 
        timeseries_tensor = torch.tensor(timeseries, dtype=torch.float32).T/self.scaling
        
        # Extract the label from the file name. We assume the label is the integer after the first underscore.
        file_name = os.path.basename(file_path)
        label_str = file_name.split('_')[1].replace('.npy', '')
        label = int(label_str)  # Convert label to integer
        
        return timeseries_tensor, label
