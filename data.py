from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as TF
import random
import rasterio as rio
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from glob import glob
import os
import torch
from main import SEED

random.seed(SEED)

class StandardizeRasterTransform:
    
    def __init__(self, maxes, mins):
        self.maxes = maxes
        self.mins = mins
    
    def __call__(self, raster, *args): # *args for compatibility with torchvision.transforms
        return torch.div(torch.sub(raster, self.mins), (self.maxes - self.mins))

class RasterSegmentationTransforms:
    
    def __init__(self, flip_probability=0.5):
        self.angles = [0, 90, 180, 270] # multiples of 90 ONLY
        self.flip_probability = flip_probability
    
    def __call__(self, X, y=None):
        angle = random.choice(self.angles)
        X = TF.rotate(X, angle)
        if y is not None: y = TF.rotate(y, angle)
        if random.random() < self.flip_probability:
            X = TF.hflip(X)
            if y is not None: y = TF.hflip(y)
        if random.random() < self.flip_probability:
            X = TF.vflip(X)
            if y is not None: y = TF.vflip(y)
        # X = self.standardize_transform(X)
        if y is not None: return X, y
        else: return X
    
class LandCoverNetDataset(Dataset):
        
    def __init__(self, paths, bands=['blue', 'green', 'red', 'nir'], mode='train', load_from_disk=True):
        self.paths = paths
        self.bands = bands
        self.n_bands = len(bands)
        self.mode = mode
        
        # first: make a pass through the dataset to get the maxes and mins

        dataset = cp.array([rio.open(path[0]).read() for path in self.paths])
        self.dataset_maxes = dataset.max(axis=(0, 2, 3))
        self.dataset_mins = dataset.min(axis=(0, 2, 3))
        self.dataset_means = dataset.mean(axis=(0, 2, 3))
        self.dataset_stds = dataset.std(axis=(0, 2, 3))
        if load_from_disk: del dataset
        
        
        # next: get the label weights
        labels_raster = cp.array([rio.open(path[1]).read(1) for path in self.paths])
        cp_weights, self.weights_bins = cp.histogram(labels_raster, range=(1,7), bins=7)
        self.weights = torch.tensor(np.insert(cp.asnumpy(cp_weights / cp.sum(cp_weights)), 0, 0)).float()
        if load_from_disk: labels_raster
        
        # self.dataset_maxes = cp.zeros(self.n_bands)
        # self.dataset_mins = cp.zeros(self.n_bands)
        # self.dataset_means = cp.zeros(self.n_bands)
        # self.dataset_stds = cp.zeros(self.n_bands)
        # for path in self.paths:
        #     raster = cp.array(rio.open(path[0]).read())
        #     raster_maxes = raster.max(axis=(1, 2))
        #     raster_mins = raster.min(axis=(1, 2))
        #     self.dataset_means += raster.mean(axis=(1, 2))
        #     for i in range(self.n_bands):
        #         if raster_maxes[i] > self.dataset_maxes[i]:
        #             self.dataset_maxes[i] = raster_maxes[i]
        #         if raster_mins[i] < self.dataset_mins[i]:
        #             self.dataset_mins[i] = raster_mins[i]
        # # average out maxes and mins
        # self.dataset_means /= len(self.paths)
        
        self.min_norm = (self.dataset_mins - self.dataset_means) / self.dataset_stds
        self.max_norm = (self.dataset_maxes - self.dataset_means) / self.dataset_stds
        
        # define transforms to use. these will be applied to both X and y
        if mode == 'train':
            self.transforms = RasterSegmentationTransforms()
        else:
            self.transforms = None
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with rio.open(path[0]) as src:
            raster = cp.array(src.read().astype(np.int32))
            meta = src.meta
        label = torch.tensor(np.expand_dims(rio.open(path[1]).read(1).astype(np.uint8), axis=0))
        raster = torch.tensor(self.standardize_raster(raster), dtype=torch.float32)
        if self.mode == 'train': raster, label = self.transforms(raster, label)
        return raster, torch.squeeze(label).type(torch.LongTensor) #, meta
    
    def visualize(self, index):
        path = self.paths[index]
        with rio.open(path[0]) as src:
            raster = src.read()
            meta = src.meta
        label = rio.open(path[1]).read()
        
        # standardize the raster
        raster = self.standardize_raster(raster)
        
        # compose images
        red_channel = raster[2]
        green_channel = raster[1]
        blue_channel = raster[0]
        nir_channel = raster[3]        
        
        RGB_image = np.stack((red_channel, green_channel, blue_channel))
        GRNIR_image = np.stack((green_channel, red_channel, nir_channel))
        
        # define colormap for LC labels
        seg_cmap = ListedColormap(['black', '#0000ff', '#888888', '#d1a46d', '#f5f5ff', '#d64c2b', '#186818', '#00ff00'])
        
        legend_elements = [
            Patch(edgecolor='white', facecolor='black', label='Other'),
            Patch(edgecolor='white', facecolor='#0000ff', label='Water'),
            Patch(edgecolor='white', facecolor='#888888', label='Artificial'),
            Patch(edgecolor='white', facecolor='#d1a46d', label='Barren'),
            Patch(edgecolor='white', facecolor='#f5f5ff', label='Snow/Ice'),
            Patch(edgecolor='white', facecolor='#d64c2b', label='Woody'),
            Patch(edgecolor='white', facecolor='#186818', label='Cultivated'),
            Patch(edgecolor='white', facecolor='#00ff00', label='(Semi) Natural'),
        ]
        
        # finally, plot the images
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax = ax.ravel()
        ax[0].imshow(RGB_image.transpose(1, 2, 0), interpolation=None)
        ax[0].set_title('RGB')
        ax[0].axis('off')
        ax[1].imshow(GRNIR_image.transpose(1, 2, 0), interpolation=None)
        ax[1].set_title('False Color')
        ax[1].axis('off')
        ax[2].imshow(label, cmap=seg_cmap, interpolation=None)
        ax[2].set_title('Land Cover')
        ax[2].legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        ax[2].axis('off')
        
        fig.tight_layout()
        # TODO: save figure to file
        # TODO: finish method
    
    def standardize_raster(self, raster):
        """
        Standardizes a raster using the maxes and mins of the dataset
        """

        raster = cp.transpose(raster, (1, 2, 0))
        normalized = (raster - self.dataset_means) / self.dataset_stds
        standardized = (normalized - self.min_norm) / (self.max_norm - self.min_norm)
        return cp.transpose(standardized, (2, 0, 1))

class LandCoverNetMultilabelDataset(Dataset):
    def __init__(self, paths, bands=['blue', 'green', 'red', 'nir'], mode='train'):
        self.paths = paths
        self.bands = bands
        self.n_bands = len(bands)
        self.mode = mode
        
        # first: make a pass through the dataset to get the maxes and mins

        dataset = cp.array([rio.open(path[0]).read() for path in self.paths])
        self.dataset_maxes = dataset.max(axis=(0, 2, 3))
        self.dataset_mins = dataset.min(axis=(0, 2, 3))
        self.dataset_means = dataset.mean(axis=(0, 2, 3))
        self.dataset_stds = dataset.std(axis=(0, 2, 3))
        del dataset
        
        self.min_norm = (self.dataset_mins - self.dataset_means) / self.dataset_stds
        self.max_norm = (self.dataset_maxes - self.dataset_means) / self.dataset_stds
        
        # next: get lables
        # rule: if class constitutes at least 5% of the patch, it is labeled as that class
        self.labels = []
        labels_raster = cp.array([rio.open(path[1]).read(1) for path in self.paths])
        self.labels = self.get_labels_from_patches(labels_raster)
        cp_weights, self.weights_bins = cp.histogram(labels_raster, range=(1,7), bins=7)
        self.weights = torch.tensor(cp.asnumpy(cp_weights / cp.sum(cp_weights))).float()
        del labels_raster
        
        # define transforms to use. these will be applied to both X and y
        if mode == 'train':
            self.transforms = RasterSegmentationTransforms()
        else:
            self.transforms = None
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with rio.open(path[0]) as src:
            raster = cp.array(src.read().astype(np.int32))
            meta = src.meta
        label = multihot_encode(self.labels[index]).float()
        raster = torch.tensor(self.standardize_raster(raster), dtype=torch.float32)
        if self.mode == 'train': raster = self.transforms(raster)
        return raster, label
    
    def get_labels_from_patches(self, labels_raster):
        """
        Returns a list of labels for the given patches
        """
        labels = []
        print(labels_raster.shape)
        n_pixels_per_label = labels_raster.shape[1] * labels_raster.shape[2]
        for patch in labels_raster:
            patch_labels = []
            for lab_class in range(0, 7):
                if cp.sum(patch == lab_class+1) / n_pixels_per_label >= 0.05:
                    patch_labels.append(lab_class)
            labels.append(patch_labels)
        return labels
    
    def visualize(self, index):
        path = self.paths[index]
        with rio.open(path[0]) as src:
            raster = src.read()
            meta = src.meta
        label = rio.open(path[1]).read()
        
        # standardize the raster
        raster = self.standardize_raster(raster)
        
        # compose images
        red_channel = raster[2]
        green_channel = raster[1]
        blue_channel = raster[0]
        nir_channel = raster[3]        
        
        RGB_image = np.stack((red_channel, green_channel, blue_channel))
        GRNIR_image = np.stack((green_channel, red_channel, nir_channel))
        
        # define colormap for LC labels
        seg_cmap = ListedColormap(['black', '#0000ff', '#888888', '#d1a46d', '#f5f5ff', '#d64c2b', '#186818', '#00ff00'])
        
        legend_elements = [
            Patch(edgecolor='white', facecolor='black', label='Other'),
            Patch(edgecolor='white', facecolor='#0000ff', label='Water'),
            Patch(edgecolor='white', facecolor='#888888', label='Artificial'),
            Patch(edgecolor='white', facecolor='#d1a46d', label='Barren'),
            Patch(edgecolor='white', facecolor='#f5f5ff', label='Snow/Ice'),
            Patch(edgecolor='white', facecolor='#d64c2b', label='Woody'),
            Patch(edgecolor='white', facecolor='#186818', label='Cultivated'),
            Patch(edgecolor='white', facecolor='#00ff00', label='(Semi) Natural'),
        ]
        
        # finally, plot the images
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax = ax.ravel()
        ax[0].imshow(RGB_image.transpose(1, 2, 0), interpolation=None)
        ax[0].set_title('RGB')
        ax[0].axis('off')
        ax[1].imshow(GRNIR_image.transpose(1, 2, 0), interpolation=None)
        ax[1].set_title('False Color')
        ax[1].axis('off')
        ax[2].imshow(label, cmap=seg_cmap, interpolation=None)
        ax[2].set_title('Land Cover')
        ax[2].legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        ax[2].axis('off')
        
        fig.tight_layout()
        # TODO: save figure to file
        # TODO: finish method
    
    def standardize_raster(self, raster):
        """
        Standardizes a raster using the maxes and mins of the dataset
        """

        raster = cp.transpose(raster, (1, 2, 0))
        normalized = (raster - self.dataset_means) / self.dataset_stds
        standardized = (normalized - self.min_norm) / (self.max_norm - self.min_norm)
        return cp.transpose(standardized, (2, 0, 1))

class ChesapeakeBayDataset(Dataset):
    
    def __init__(self, paths: list[tuple[str, str]], bands=[2, 3, 4], mode='train', load_from_disk=True):
        
        dataset_cp_array = cp.array(
            [[rio.open(path[0]).read(band) for band in bands] for path in paths]
        )
        self.dataset_max = dataset_cp_array.max(axis=(0, 2, 3))
        self.dataset_min = dataset_cp_array.min(axis=(0, 2, 3))
        
        self.transforms = RasterSegmentationTransforms()
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError

def split_files(filepaths, val_split=0.2):
    """
    Splits a list of filepaths
    """
    # https://stackoverflow.com/a/17413351
    random.shuffle(filepaths)
    idx = int(len(filepaths) * val_split)
    val_paths = filepaths[:idx]
    train_paths = filepaths[idx:]
    
    return train_paths, val_paths

def get_file_paths(path):
    """
    Returns a list of filepaths for all files in the given directory
    """
    filepaths = []
    directories = next(os.walk(path))[1] # https://stackoverflow.com/a/142535
    for dir in directories:
        filepaths.append(
            (
                glob(os.path.join(path, dir, '*source*.tif'))[0],
                os.path.join(path, dir, 'target.tif'),
            )
        )
    return filepaths

def standardize_np_array(array, maxes, mins):
    """
    Standardizes a numpy array using the maxes and mins of the dataset
    """
    return np.divide(np.subtract(array, mins), (maxes - mins))

def multihot_encode(x, n_classes=7):
    out = torch.zeros(n_classes)
    for i in range(n_classes):
        out[i] = i in x
    return out

def get_cpblulc_filepaths(path):
    filepaths = []
    directories = next(os.walk(path))[1] # https://stackoverflow.com/a/142535
    
    for directory in directories:
        pass