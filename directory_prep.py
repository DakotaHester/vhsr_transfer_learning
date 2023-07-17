# a simple script for getting LCN data into the following format:
# | dataset_name
# || patch_id
# ||| input.tif
# ||| target.tif

import os
from tqdm import trange
from glob import glob
import cupy as cp
from numba import njit, prange
import rasterio as rio
from shutil import copy2
import numpy as np

dataset_names = [
    'na',
    'sa',
    'as',
    'eu',
    'af',
    'au',
]
dataset_samples = {
    
}
for dataset_name in dataset_names:
    
    print(f'prepping {dataset_name}...')

    labels_path = f'/scratch/landcovernet_{dataset_name}/ref_landcovernet_{dataset_name}_v1/ref_landcovernet_{dataset_name}_v1_labels'
    source_path = f'/scratch/landcovernet_{dataset_name}/ref_landcovernet_{dataset_name}_v1/ref_landcovernet_{dataset_name}_v1_source_sentinel_2'
    source_dir_name = f'ref_landcovernet_{dataset_name}_v1_source_sentinel_2'
    labels_dir_name = f'ref_landcovernet_{dataset_name}_v1_labels'
    bands = ['B02.tif', 'B03.tif', 'B04.tif', 'B08.tif'] # B, G, R, NIR
    out_dir = f'/scratch/lcn_global_prepped/'

    if not os.path.exists(out_dir): os.mkdir(out_dir)

    print('getting patch ids...')
    ids = [directory[-8:] for directory in os.listdir(labels_path)]

    patches_to_use = []
    n_samples = 0

    for i in trange(len(ids), desc='copying patches...'):
        id = ids[i]
        # print(id)
        potential_patches = glob(os.path.join(source_path, source_dir_name + '_' + id + '*'))
        
        try:
            # print(potential_patches)
            if len(potential_patches) == 0:
                continue
            elif len(potential_patches) == 1:
                cloud_data = rio.open(os.path.join(potential_patches[0], 'CLD.tif')).read()
                cloud_cover_percent = int(100 * cloud_data.sum() / (cloud_data.shape[1] * cloud_data.shape[2]))
                best_patch = potential_patches[0]
            else:
                cloud_data = cp.array([rio.open(os.path.join(patch, 'CLD.tif')).read() for patch in potential_patches])
                cloud_cover = cloud_data.sum(axis=(1, 2, 3))
                cloud_cover_argmin = int(cloud_cover.argmin())
                best_patch = potential_patches[cloud_cover_argmin]
                cloud_cover_percent = int(cloud_cover[cloud_cover_argmin] / (cloud_data.shape[1] * cloud_data.shape[2] * cloud_data.shape[3]) * 100)
            
            with rio.open(os.path.join(best_patch, 'CLD.tif')) as src:
                raster_meta = src.meta
            merged_raster = np.zeros((len(bands), cloud_data.shape[2], cloud_data.shape[3]))
        except rio.errors.RasterioIOError:
            continue
        
        for band in bands:
            merged_raster[bands.index(band)] = rio.open(os.path.join(best_patch, band)).read()
        
        if not os.path.exists(os.path.join(out_dir, id)): os.mkdir(os.path.join(out_dir, id))
        with rio.open(
            os.path.join(out_dir, id, f's2_source_{best_patch[-8:]}_{cloud_cover_percent:02}.tif'),
            'w',
            driver='GTiff',
            width=merged_raster.shape[2],
            height=merged_raster.shape[1],
            count=len(bands),
            crs=raster_meta['crs'],
            transform=raster_meta['transform'],
            dtype=merged_raster.dtype,
            ) as dst:
            dst.write(merged_raster)
        
        # copy labels
        copy2(os.path.join(labels_path, labels_dir_name + '_' + id, 'labels.tif'), os.path.join(out_dir, id, 'target.tif'))
        n_samples += 1
    
    dataset_samples[dataset_name] = n_samples

print(dataset_samples)

