import torch
import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import data
import utils
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

encoder = 'efficientnet-b7'
model = 'deeplabv3plus'
loss = 'focal'

dataset_test_path = {
    'nyc': 'nyc_deeplabv3plus_efficientnet-b7_focal_input_resolution_512_n_samples_128/nyc_deeplabv3plus_efficientnet-b7_focal_input_resolution_512_n_samples_128_nyc_test.pkl',
    'cpblulc': 'cpblulc_deeplabv3plus_efficientnet-b7_focal_input_resolution_512_n_samples_128/cpblulc_deeplabv3plus_efficientnet-b7_focal_input_resolution_512_n_samples_128_cpblulc_test.pkl'
}

model_paths = {
    'nyc': 'nyc_deeplabv3plus_efficientnet-b7_focal_input_resolution_512_n_samples_128/models/nyc_deeplabv3plus_efficientnet-b7_focal_input_resolution_512_n_samples_128_60_1.3546112775802612.pt',
    'cpblulc': 'cpblulc_deeplabv3plus_efficientnet-b7_focal_input_resolution_512_n_samples_128/models/cpblulc_deeplabv3plus_efficientnet-b7_focal_input_resolution_512_n_samples_128_38_0.9489578008651733.pt'
}

model2_paths = {
    'nyc': 'nyc_unet_tu-xception71_focal_input_resolution_512_n_samples_128/models/nyc_unet_tu-xception71_focal_input_resolution_512_n_samples_128_41_1.3675282001495361.pt',
    'cpblulc': '/home/dhester/vhsr_transfer_learning/cpblulc_unet_tu-xception71_focal_input_resolution_512_n_samples_128/models/cpblulc_unet_tu-xception71_focal_input_resolution_512_n_samples_128_27_0.9454334378242493.pt'
}

xgboost_paths = {
    'nyc': 'XGBOOST_nyclc_n_samples_128_resolution_224/model.pkl',
    'cpblulc': 'XGBOOST_cpblulc_n_samples_128_resolution_224/model.pkl'
}

datasets = ['nyc', 'cpblulc']
# xgboost_model_path = 'XGBOOST_cpblulc_n_samples_128_resolution_224/model.pkl'

for dataset_name, test_path in dataset_test_path.items():
    
    test_data = pickle.load(open(test_path, 'rb'))
    test_samples = [test_data[i] for i in range(5)]
    # print(test_samples)
    
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes =8 if dataset_name == 'nyc' else 6,
    )
    model.load_state_dict(torch.load(model_paths[dataset_name]))
    model.eval()
    
    model2 = smp.Unet(
        encoder_name='tu-xception71',
        encoder_weights='imagenet',
        in_channels=3,
        classes=8 if dataset_name == 'nyc' else 6,
    )
    model2.load_state_dict(torch.load(model2_paths[dataset_name]))
    model2.eval()
    
    test_samples_X = torch.stack([test_samples[i][0] for i in range(len(test_samples))])
    test_samples_Y = torch.stack([test_samples[i][1] for i in range(len(test_samples))])
    
    print(f'test_samples_X - type: {type(test_samples_X)}, shape: {test_samples_X.shape}, range: ({test_samples_X.min()}, {test_samples_X.max()}), dtype: {test_samples_X.dtype}')
    print(f'test_samples_Y - type: {type(test_samples_Y)}, shape: {test_samples_Y.shape}, range: ({test_samples_Y.min()}, {test_samples_Y.max()}), dtype: {test_samples_X.dtype}')
    
    preds = model.predict(test_samples_X).argmax(axis=1)
    preds2 = model2.predict(test_samples_X).argmax(axis=1)
    
    print(f'preds - type: {type(preds)}, shape: {preds.shape}, range: ({preds.min()}, {preds.max()}), dtype: {preds.dtype}')
    print(f'preds2 - type: {type(preds2)}, shape: {preds2.shape}, range: ({preds2.min()}, {preds2.max()}), dtype: {preds2.dtype}')
    
    test_samples_X = test_samples_X.numpy()
    test_samples_Y = test_samples_Y.numpy()
    preds = preds.numpy()
    preds2 = preds2.numpy()
    
    error = np.where(preds != test_samples_Y, 1, 0)
    error2 = np.where(preds2 != test_samples_Y, 1, 0)
    print(f'error - type: {type(error)}, shape: {error.shape}, range: ({error.min()}, {error.max()}), dtype: {error.dtype}')
    print(f'error2 - type: {type(error2)}, shape: {error2.shape}, range: ({error2.min()}, {error2.max()}), dtype: {error2.dtype}')
    
                                            #     0  1  2  3
    xgb_test_samples_X = test_samples_X.transpose(0, 2, 3, 1) * 255
    og_xgb_test_samples_shape_X = xgb_test_samples_X.shape
    xgb_test_samples_X_flat = xgb_test_samples_X.reshape(-1, 3)
    og_xgb_test_samples_shape_Y = test_samples_Y.shape
    xgb_test_samples_Y = test_samples_Y.reshape(-1)
    print(f'xgb_test_samples_X - type: {type(xgb_test_samples_X)}, shape: {xgb_test_samples_X.shape}, range: ({xgb_test_samples_X.min()}, {xgb_test_samples_X.max()}), dtype: {xgb_test_samples_X.dtype}')
    print(f'xgb_test_samples_X_flat - type: {type(xgb_test_samples_X_flat)}, shape: {xgb_test_samples_X_flat.shape}, range: ({xgb_test_samples_X_flat.min()}, {xgb_test_samples_X_flat.max()}), dtype: {xgb_test_samples_X_flat.dtype}')
    print(f'xgb_test_samples_Y - type: {type(xgb_test_samples_Y)}, shape: {xgb_test_samples_Y.shape}, range: ({xgb_test_samples_Y.min()}, {xgb_test_samples_Y.max()}), dtype: {xgb_test_samples_Y.dtype}')
    
    
    xgb_model = pickle.load(open(xgboost_paths[dataset_name], 'rb'))
    
    xgb_preds_flat = xgb_model.predict(xgb_test_samples_X_flat)
    xgb_preds = xgb_preds_flat.reshape(og_xgb_test_samples_shape_Y)
    print(xgb_model.score(xgb_test_samples_X_flat, xgb_test_samples_Y))
    xgb_error = np.where(test_samples_Y != xgb_preds, 1, 0)
    print(f'xgb_preds_flat - type: {type(xgb_preds_flat)}, shape: {xgb_preds_flat.shape}, range: ({xgb_preds_flat.min()}, {xgb_preds_flat.max()}), dtype: {xgb_preds_flat.dtype}')
    print(f'xgb_preds - type: {type(xgb_preds)}, shape: {xgb_preds.shape}, range: ({xgb_preds.min()}, {xgb_preds.max()}), dtype: {xgb_preds.dtype}')
    print(f'xgb_error - type: {type(xgb_error)}, shape: {xgb_error.shape}, range: ({xgb_error.min()}, {xgb_error.max()}), dtype: {xgb_error.dtype}')
    

    
    if dataset_name == 'nyc':
        land_cover_map_colormap = ListedColormap(['darkgreen', 'lawngreen', 'lightyellow', 'aqua', 'dimgray', 'lightgray', 'black', 'brown'])
        land_cover_map_legend_elements = [
            Patch(edgecolor='black', facecolor='darkgreen', label='Tree Canopy'),
            Patch(edgecolor='black', facecolor='lawngreen', label='Grass/Shrubs'),
            Patch(edgecolor='black', facecolor='lightyellow', label='Bare Soils'),
            Patch(edgecolor='black', facecolor='aqua', label='Water'),
            Patch(edgecolor='black', facecolor='dimgray', label='Buildings'),
            Patch(edgecolor='black', facecolor='lightgray', label='Roads'),
            Patch(edgecolor='black', facecolor='black', label='Other Impervious'),
            Patch(edgecolor='black', facecolor='brown', label='Railroads'),
        ]
        vmax = 7
    else:
        land_cover_map_colormap = ListedColormap(['aqua', 'darkgreen', 'lawngreen', 'lightyellow', 'dimgray', 'lightgray'])
        land_cover_map_legend_elements = [
            Patch(edgecolor='black', facecolor='aqua', label='Water'),
            Patch(edgecolor='black', facecolor='darkgreen', label='Tree Canopy/Forest'),
            Patch(edgecolor='black', facecolor='lawngreen', label='Low Vegetation/Field'),
            Patch(edgecolor='black', facecolor='lightyellow', label='Barren'),
            Patch(edgecolor='black', facecolor='dimgray', label='Impervious Other'),
            Patch(edgecolor='black', facecolor='lightgray', label='Impervious Road'),
        ]
        vmax = 5
    
    
    fig, ax = plt.subplots(5, 8, figsize=(18, 10))
    ax[0, 0].set_title('False Color Input')
    ax[0, 1].set_title('Ground Truth')
    ax[0, 2].set_title('DeepLabV3+ Inference')
    ax[0, 3].set_title('DeepLabV3+ Error')
    ax[0, 4].set_title('UNet Inference')
    ax[0, 5].set_title('UNet Error')
    ax[0, 6].set_title('XGBoost Inference')
    ax[0, 7].set_title('XGBoost Error')
    for i in range(5):
        ax[i, 0].imshow(test_samples_X[i].transpose((1, 2, 0)), interpolation=None)
        ax[i, 0].axis('off')
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        
        ax[i, 1].imshow(test_samples_Y[i], interpolation=None, cmap=land_cover_map_colormap, vmin=0, vmax=vmax)
        ax[i, 1].axis('off')
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        
        ax[i, 2].imshow(preds[i], interpolation=None, cmap=land_cover_map_colormap, vmin=0, vmax=vmax)
        ax[i, 2].axis('off')
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])
        
        ax[i, 3].imshow(test_samples_X[i].transpose((1, 2, 0)), interpolation=None)
        ax[i, 3].imshow(error[i], interpolation=None, cmap='Reds', alpha=error[i].astype(float), vmin=0, vmax=1)
        ax[i, 3].axis('off')
        ax[i, 3].set_xticks([])
        ax[i, 3].set_yticks([])
        
        ax[i, 4].imshow(preds2[i], interpolation=None, cmap=land_cover_map_colormap, vmin=0, vmax=vmax)
        ax[i, 4].axis('off')
        ax[i, 4].set_xticks([])
        ax[i, 4].set_yticks([])
        
        ax[i, 5].imshow(test_samples_X[i].transpose((1, 2, 0)), interpolation=None)
        ax[i, 5].imshow(error2[i], interpolation=None, cmap='Reds', alpha=error2[i].astype(float), vmin=0, vmax=1)
        ax[i, 5].axis('off')
        ax[i, 5].set_xticks([])
        ax[i, 5].set_yticks([])
        
        ax[i, 6].imshow(xgb_preds[i], interpolation=None, cmap=land_cover_map_colormap, vmin=0, vmax=vmax)
        ax[i, 6].axis('off')
        ax[i, 6].set_xticks([])
        ax[i, 6].set_yticks([])
        
        ax[i, 7].imshow(test_samples_X[i].transpose((1, 2, 0)), interpolation=None)
        ax[i, 7].imshow(xgb_error[i], interpolation=None, cmap='Reds', alpha=xgb_error[i].astype(float), vmin=0, vmax=1)
        ax[i, 7].axis('off')
        ax[i, 7].set_xticks([])
        ax[i, 7].set_yticks([])
        
    ax[0, 7].legend(handles=land_cover_map_legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
    fig.tight_layout()
    fig.savefig('{}_inference_comparison.png'.format(dataset_name), dpi=300, bbox_inches='tight')
    
    