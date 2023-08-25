import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from transformers import SegformerConfig, SegformerModel, SegformerDecodeHead, SegformerForSemanticSegmentation, SegformerFeatureExtractor, SegformerImageProcessor


# Model variant 	Depths 	        Hidden sizes 	        Decoder hidden size 	Params (M) 	ImageNet-1k Top 1
# MiT-b0 	        [2, 2, 2, 2] 	[32, 64, 160, 256] 	    256 	                3.7 	    70.5
# MiT-b1 	        [2, 2, 2, 2] 	[64, 128, 320, 512] 	256 	                14.0 	    78.7
# MiT-b2 	        [3, 4, 6, 3] 	[64, 128, 320, 512] 	768 	                25.4 	    81.6
# MiT-b3 	        [3, 4, 18, 3] 	[64, 128, 320, 512] 	768 	                45.2 	    83.1
# MiT-b4 	        [3, 8, 27, 3] 	[64, 128, 320, 512] 	768 	                62.6 	    83.6
# MiT-b5 	        [3, 6, 40, 3] 	[64, 128, 320, 512] 	768 	                82.0 	    83.8

MODEL_CONFIGS = {
    'MiT-b0': {
        'depths': [2, 2, 2, 2],
        'hidden_sizes': [32, 64, 160, 256],
        'decoder_hidden_size': 256,
    },
    'MiT-b1': {
        'depths': [2, 2, 2, 2],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 256,
    },
    'MiT-b2': {
        'depths': [3, 4, 6, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 768,
    },
    'MiT-b3': {
        'depths': [3, 4, 18, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 768,
    },
    'MiT-b4': {
        'depths': [3, 8, 27, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 768,
    },
    'MiT-b5': {
        'depths': [3, 6, 40, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 768,
    },
}

model_config_kw_args = MODEL_CONFIGS['MiT-b0']
config = SegformerConfig(
    num_channels=4,
    **model_config_kw_args,
)
model = SegformerModel(config)
decoder_head = SegformerDecodeHead(config)
print(model.encoder, decoder_head)

