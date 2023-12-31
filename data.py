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
    
    def __init__(self, paths: list[tuple[str, str]], bands=[4, 1, 2], mode='train', load_from_disk=True, device='cpu'):
        
        xp = cp
        
        print(xp)
        
        
        self.dataset_paths = paths
        self.bands = bands
        self.mode = mode
        self.device = device
        self.n_classes = 6
        
        # self.dataset_X_xp_array = xp.array(
        #     [[rio.open(path[0]).read(band) for band in bands] for path in paths]
        # )
        # print('loaded X')
        self.dataset_y_xp_array = xp.array(
            [rio.open(path[1]).read(1) for path in paths]
        ) - 1 # subtract 1 to make classes 0-5 instead of 1-6
        print('loaded y')
        
        # # remove samples that include aberdeen proving ground (254)
        # for i in range(len(self.dataset_y_xp_array)):
        #     n_removed_samples = 0
        #     if np.any(self.dataset_y_xp_array[i] == 14):
        #         np.delete(self.dataset_y_xp_array, i)
        #         np.delete(self.dataset_paths, i)
        #         n_removed_samples += 1
        # print(f'removed {n_removed_samples} samples with no data values')
            
        
        self.class_weights = xp.bincount(self.dataset_y_xp_array.flatten()) / len(self.dataset_y_xp_array.flatten())
        print('got class weights: ', self.class_weights)
        
        # NAIP data range [0, 255]
        self.dataset_max = 255
        self.dataset_min = 0
        
        self.class_min = 0
        self.class_max = 5
        
        print('target min/max: ', self.class_min, self.class_max)
        
        
        self.transforms = RasterSegmentationTransforms()
        
        # 1 - water
        # 2 - tree canopy/forest
        # 3 - low vegetation/field
        # 4 - barren
        # 5 - impervious other
        # 6 - impervious road 
        self.land_cover_map_colormap = ListedColormap(['aqua', 'darkgreen', 'lawngreen', 'lightyellow', 'dimgray', 'lightgray'])
        self.land_cover_map_legend_elements = [
            Patch(edgecolor='black', facecolor='aqua', label='Water'),
            Patch(edgecolor='black', facecolor='darkgreen', label='Tree Canopy/Forest'),
            Patch(edgecolor='black', facecolor='lawngreen', label='Low Vegetation/Field'),
            Patch(edgecolor='black', facecolor='lightyellow', label='Barren'),
            Patch(edgecolor='black', facecolor='dimgray', label='Impervious Other'),
            Patch(edgecolor='black', facecolor='lightgray', label='Impervious Road'),
        ]
            
    
    def __len__(self):
        
        return len(self.dataset_paths)
    
    def __getitem__(self, index):
        
        X_np = np.array([[rio.open(self.dataset_paths[index][0]).read(band) for band in self.bands]])
        X_tensor = torch.Tensor(X_np)
        y_tensor = torch.Tensor(self.dataset_y_xp_array[index]).unsqueeze(0)
        
        X_tensor = self.standardize_raster_tensor(X_tensor)
        if self.mode == 'train': X_tensor, y_tensor = self.transforms(X_tensor, y_tensor)
        
        X_tensor = X_tensor.squeeze(0)
        y_tensor = y_tensor.squeeze(0).long()
        
        return X_tensor, y_tensor
    
    def visualize(self, index):
        
        false_color = np.array([rio.open(self.dataset_paths[index][0]).read(band) for band in [4, 1, 2]])
        RGB = np.array([rio.open(self.dataset_paths[index][0]).read(band) for band in [1, 2, 3]])
        false_color = self.standardize_raster_tensor(false_color)
        RGB = self.standardize_raster_tensor(RGB)
        y = self.dataset_y_xp_array[index]
        
        if type(y) == cp.ndarray: y = cp.asnumpy(y)
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(RGB.transpose(1, 2, 0), interpolation=None)
        axes[0].set_title('RGB Image (Standardized)')
        axes[0].axis('off')
        axes[1].imshow(false_color.transpose(1, 2, 0), interpolation=None)
        axes[1].set_title('False Color Image (Standardized)')
        axes[1].axis('off')
        axes[2].imshow(y, cmap=self.land_cover_map_colormap, interpolation=None, vmin=self.class_min, vmax=self.class_max)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        axes[2].legend(handles=self.land_cover_map_legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', ncol=1)
        fig.tight_layout()
        plt.savefig(f'test_fig_CPB_{index}.png')
        plt.close()
    
    def standardize_raster_tensor(self, raster_tensor):
        
        # raster = cp.transpose(raster, (1, 2, 0))
        # normalized = (raster - self.dataset_means) / self.dataset_stds
        # standardized = (normalized - self.min_norm) / (self.max_norm - self.min_norm)
        # return cp.transpose(standardized, (2, 0, 1))
        
        # no need to transpose since subtracting a scalar is commutative
        standardized = (raster_tensor - self.dataset_min) / (self.dataset_max - self.dataset_min)
        return standardized

class NYCDataset(Dataset):
    
    def __init__(self, paths: list[tuple[str, str]], bands=[4, 1, 2], mode='train', load_from_disk=True, device='cpu'):
        
        xp = np
        
        print(xp)
        
        
        self.dataset_paths = paths
        self.bands = bands
        self.mode = mode
        self.device = device

        # NAIP data range [0, 255]
        self.dataset_max = 255
        self.dataset_min = 0
        
        self.class_min = 0
        self.class_max = 7
        self.n_classes = 8
        
        # self.dataset_X_xp_array = xp.array(
        #     [[rio.open(path[0]).read(band) for band in bands] for path in paths]
        # )
        # print('loaded X')
        self.dataset_y_xp_array = xp.array(
            [rio.open(path[1]).read(1) for path in paths]
        )
        # print('loaded y')

        
        # remove samples with no data values
        # n_removed_samples = 0
        # i = 0
        # while i < len(self.dataset_y_xp_array):
        #     if np.any(self.dataset_y_xp_array[i] == 15):
        #         n_removed_samples += 1
        #         self.dataset_y_xp_array = np.delete(self.dataset_y_xp_array, i, axis=0)
        #         self.dataset_paths = np.delete(self.dataset_paths, i, axis=0)
        #     else: i+=1
        # print(len(self.dataset_y_xp_array))
        # print(f'removed {n_removed_samples} samples with no data values')

        self.class_max = self.dataset_y_xp_array.max()            
        
        self.class_weights = xp.bincount(self.dataset_y_xp_array.flatten()) / len(self.dataset_y_xp_array.flatten())
        if self.class_weights.shape[0] < self.n_classes:
            self.class_weights = xp.pad(self.class_weights, (0, self.n_classes - self.class_weights.shape[0]), 'constant', constant_values=0)
        # print('got class weights: ', self.class_weights)
        

        
        # print('target min/max: ', self.class_min, self.class_max)
        
        
        self.transforms = RasterSegmentationTransforms()
        
        # 0 - tree canopy
        # 1 - Grass\Shrubs
        # 2 - bare soils
        # 3 - Water
        # 4 - buildings
        # 5 - Roads
        # 6 - Other impervious
        # 7 - railroads
        self.land_cover_map_colormap = ListedColormap(['darkgreen', 'lawngreen', 'lightyellow', 'aqua', 'dimgray', 'lightgray', 'black', 'brown'])
        self.land_cover_map_legend_elements = [
            Patch(edgecolor='black', facecolor='darkgreen', label='Tree Canopy'),
            Patch(edgecolor='black', facecolor='lawngreen', label='Grass/Shrubs'),
            Patch(edgecolor='black', facecolor='lightyellow', label='Bare Soils'),
            Patch(edgecolor='black', facecolor='aqua', label='Water'),
            Patch(edgecolor='black', facecolor='dimgray', label='Buildings'),
            Patch(edgecolor='black', facecolor='lightgray', label='Roads'),
            Patch(edgecolor='black', facecolor='black', label='Other Impervious'),
            Patch(edgecolor='black', facecolor='brown', label='Railroads'),
        ]
        

    
    def __len__(self):
        
        return len(self.dataset_paths)
    
    def __getitem__(self, index):
        
        X_np = np.array([[rio.open(self.dataset_paths[index][0]).read(band) for band in self.bands]])
        X_tensor = torch.Tensor(X_np)
        y_tensor = torch.Tensor(self.dataset_y_xp_array[index]).unsqueeze(0)
        
        X_tensor = self.standardize_raster_tensor(X_tensor)
        if self.mode == 'train': X_tensor, y_tensor = self.transforms(X_tensor, y_tensor)
        X_tensor = X_tensor.squeeze(0)
        y_tensor = y_tensor.squeeze(0).long()
        return X_tensor, y_tensor
    
    def visualize(self, index):
        
        false_color = np.array([rio.open(self.dataset_paths[index][0]).read(band) for band in [4, 1, 2]])
        RGB = np.array([rio.open(self.dataset_paths[index][0]).read(band) for band in [1, 2, 3]])
        false_color = self.standardize_raster_tensor(false_color)
        RGB = self.standardize_raster_tensor(RGB)
        y = self.dataset_y_xp_array[index]
        
        if type(y) == cp.ndarray: y = cp.asnumpy(y)
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(RGB.transpose(1, 2, 0), interpolation=None)
        axes[0].set_title('RGB Image (Standardized)')
        axes[0].axis('off')
        axes[1].imshow(false_color.transpose(1, 2, 0), interpolation=None)
        axes[1].set_title('False Color Image (Standardized)')
        axes[1].axis('off')
        axes[2].imshow(y, cmap=self.land_cover_map_colormap, interpolation=None, vmin=self.class_min, vmax=self.class_max)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        axes[2].legend(handles=self.land_cover_map_legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', ncol=1)
        fig.tight_layout()
        plt.savefig(f'test_fig_NYC_{index}.png')
        plt.close()
    
    def standardize_raster_tensor(self, raster_tensor):
        
        # raster = cp.transpose(raster, (1, 2, 0))
        # normalized = (raster - self.dataset_means) / self.dataset_stds
        # standardized = (normalized - self.min_norm) / (self.max_norm - self.min_norm)
        # return cp.transpose(standardized, (2, 0, 1))
        
        # no need to transpose since subtracting a scalar is commutative
        standardized = (raster_tensor - self.dataset_min) / (self.dataset_max - self.dataset_min)
        return standardized

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

def remove_nodata_samples(filepaths: list[tuple[str, str]], nodata_value=15, n_samples=0):
    if n_samples == 0: n_samples = len(filepaths)
    random.shuffle(filepaths)
    i = 0
    removed_samples = 0
    clean_filepaths = []
    while len(clean_filepaths) < n_samples and i < len(filepaths):
        print(filepaths[i][1])
        if np.any(rio.open(filepaths[i][1]).read(1) == nodata_value):
            removed_samples += 1
        else:
            clean_filepaths.append(filepaths[i])
        i+=1
    return clean_filepaths
