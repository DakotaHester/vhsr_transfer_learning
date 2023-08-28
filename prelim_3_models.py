import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from transformers import SegformerConfig, SegformerModel, SegformerDecodeHead, SegformerForSemanticSegmentation, SegformerFeatureExtractor, SegformerImageProcessor
import torchvision

# Model variant 	Depths 	        Hidden sizes 	        Decoder hidden size 	Params (M) 	ImageNet-1k Top 1
# MiT-b0 	        [2, 2, 2, 2] 	[32, 64, 160, 256] 	    256 	                3.7 	    70.5
# MiT-b1 	        [2, 2, 2, 2] 	[64, 128, 320, 512] 	256 	                14.0 	    78.7
# MiT-b2 	        [3, 4, 6, 3] 	[64, 128, 320, 512] 	768 	                25.4 	    81.6
# MiT-b3 	        [3, 4, 18, 3] 	[64, 128, 320, 512] 	768 	                45.2 	    83.1
# MiT-b4 	        [3, 8, 27, 3] 	[64, 128, 320, 512] 	768 	                62.6 	    83.6
# MiT-b5 	        [3, 6, 40, 3] 	[64, 128, 320, 512] 	768 	                82.0 	    83.8

MODEL_CONFIGS = {
    'mit-b0': {
        'depths': [2, 2, 2, 2],
        'hidden_sizes': [32, 64, 160, 256],
        'decoder_hidden_size': 256,
    },
    'mit-b1': {
        'depths': [2, 2, 2, 2],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 256,
    },
    'mit-b2': {
        'depths': [3, 4, 6, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 768,
    },
    'mit-b3': {
        'depths': [3, 4, 18, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 768,
    },
    'mit-b4': {
        'depths': [3, 8, 27, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 768,
    },
    'mit-b5': {
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

class SegFormerEncoder(torchvision.models.vision_transformer.VisionTransformer):
    
    # this class adds a few extra attributes to the SegFormerEncoder class for 
    # compatibility with the MAE class
    
    def __init__(self, config: SegformerConfig) -> None:
    
        super().__init__()
        
        self.model = SegformerModel(self.config)
        
        # important params for MAE
        self.patch_size = None
        self.seq_length = None
        self.hidden_dim = None
        
        self.pos_embedding = 
        self.dropout = self.encoder.config.classifier_dropout_prob
        self.layers = self.encoder.
        self.ln = 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x
    

class SegFormer(nn.Module):

    def __init__(self, num_classes: int =2, variant: str ='mit-b0') -> None:
        
        super().__init__()
        
        # model config setup
        self.num_classes = num_classes
        self.variant = variant.lower()
        model_config_kw_args = MODEL_CONFIGS[self.variant]
        self.config = SegformerConfig(
            num_channels=self.num_classes,
            **model_config_kw_args,
        )
        
        self.encoder = SegformerModel(self.config)
        self.decoder = SegformerDecodeHead(self.config)
        
        # important params for MAE
        self.patch_size = None
        self.seq_length = None
        self.hidden_dim = None
        
        self.pos_embedding = 
        self.dropout = self.encoder.config.classifier_dropout_prob
        self.layers = self.encoder.
        self.ln = 
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    