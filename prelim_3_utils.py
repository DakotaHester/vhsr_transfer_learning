import torch
import torch.nn as nn
import numpy as np
import rasterio
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os


# grad caching library
from grad_cache.functional import cached, cat_input_tensor

class NT_Xent_Loss(nn.Module):
    # code adapted from https://towardsdatascience.com/nt-xent-normalized-temperature-scaled-cross-entropy-loss-explained-and-implemented-in-pytorch-cc081f69848
    
    def __init__(self, temperature=0.5):
        super(NT_Xent_Loss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        
        # normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        offset = z1.size(0)
        device = z1.device
        
        # concatenate embeddings
        embeddings = torch.cat([z1, z2], dim=0)
        
        # cosine similarity matrix
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=-1)

        # remove diagonal elements - cosine similarity between the same embeddings will be 
        eye = torch.eye(2 * z1.size(0)) 
        mask = eye.bool()
        similarity_matrix[mask] = float('-inf')
        
        # create ground truth labels
        # ground truth labels should follow the pattern: [1, 0, 3, 2, 5, 4, 7, 6, ...]
        target = torch.cat([torch.arange(offset, 2 * offset), torch.arange(0, offset)]).to(device)
        
        # compute loss
        loss = F.cross_entropy(similarity_matrix / self.temperature, target, reduction='mean')
        return loss


# https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    
    gaussian_kernel_size = int(0.1 * size)
    if gaussian_kernel_size % 2 == 0: # kernel size must be odd
        gaussian_kernel_size += 1
    
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=gaussian_kernel_size),
    ])
    return data_transforms

class SimClrProjectionHead(nn.Module):
    '''
    
    From the paper:
    We use ResNet-50 as the base encoder network, and a 2-layer MLP projection 
    head to project the representation to a 128-dimensional latent space
    
    '''
    def __init__(self, mlp_dim, output_dim=128):
        
        super().__init__()
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.projection_head = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.output_dim)
        )
        
    def forward(self, x):
        return self.projection_head(x)

class SimClrModel(nn.Module):
    
    def __init__(self, backbone):
        
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimClrProjectionHead(self.backbone.fc.out_features)
        
    def forward(self, x):
        h = self.backbone(x)
        return self.projection_head(h)

class NAIP_CPB_Contrastive_Dataset(Dataset):
    
    def __init__(self, file_list, n_views=2, image_size=224):
        
        self.file_list = file_list
        self.transform = get_simclr_pipeline_transform(image_size)
        self.n_views = n_views
    
    def __len__(self):
        return len(self.file_list)

    def _generate_views(self, x):
        # create n_views augmented versions of the same image
        return [self.transform(x) for _ in range(self.n_views)]
    
    def __getitem__(self, index):
        with rasterio.open(self.file_list[index][0]) as src:
            raster = torch.tensor(src.read([4, 1, 2])) / 255. # normalize to [0,1]
        
        views = self._generate_views(raster)
        return views

def get_list_of_files(data_path: str='./data') -> list[tuple[str, str]]:
    '''Returns a list of tuples of the form (path_to_input_sample, path_to_label)
    
    Parameters:
    data_path (str): path to the root of the data directory
    
    Returns:
    list[tuple[str, str]]: List of file paths with corresponding ground truth labels
    '''
    
    # get list of patch ids. list comprehension pretty much says "look at the 
    # items in the data_path directory and if they are directories, add them to 
    # the list"
    file_paths = []
    patch_ids = [sub_dir for sub_dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, sub_dir))]
    
    for patch in patch_ids:
        # get all subsamples of each patch
        subsamples = os.listdir(os.path.join(data_path, patch, 'input'))
        # add subsamples to file_path list
        file_paths.extend(
            (
                os.path.join(data_path, patch, 'input', subsample),
                os.path.join(data_path, patch, 'target', subsample)
            ) for subsample in subsamples if subsample.endswith('.tif')
        )
        
    
    # check to make sure files exist
    for file_path in file_paths:
        if not os.path.isfile(file_path[0]): # if source file doesn't exist
            raise FileNotFoundError(f'Input file {file_path[0]} not found')
        if not os.path.isfile(file_path[1]): # if label file doesn't exist
            raise FileNotFoundError(f'Label file {file_path[1]} not found')
    
    return file_paths

def visualize_transformations(X1, X2, title=None):
    
    X1 = X1.numpy()
    X2 = X2.numpy()
    
    X1 = X1.transpose(1, 2, 0) # convert from (bands, rows, cols) to (rows, cols, bands)
    X2 = X2.transpose(1, 2, 0)
    
    # create plt figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # plot RGB image
    ax[0].imshow(X1, vmin=0, vmax=255)
    ax[0].set_title('X1')
    ax[0].axis('off')
    
    # plot CIR image
    ax[1].imshow(X2, vmin=0, vmax=255)
    ax[1].set_title('X2')
    ax[1].axis('off')
    
    if title is not None:
        fig.suptitle(title)
    
    fig.tight_layout() # formatting
    plt.show()