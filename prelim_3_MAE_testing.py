# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn


from lightly.models import utils
from lightly.models.modules import masked_autoencoder
from lightly.transforms.mae_transform import MAETransform
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



class MAE(pl.LightningModule):
    def __init__(self, vit, patch_size, sequence_length, hidden_dim):
        super().__init__()

        decoder_dim = 512
        # vit = torchvision.models.vit_b_32(pretrained=False)
        self.mask_ratio = 0.75
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=hidden_dim,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=vit.patch_size**2 * 3,
            dropout=0,
            attention_dropout=0,
        )
        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        views = batch[0]
        images = views[0]  # views contains only a single view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim


model_config_kw_args = MODEL_CONFIGS['MiT-b0']
config = SegformerConfig(
    num_channels=4,
    **model_config_kw_args,
)
model = SegformerModel(config)
decoder_head = SegformerDecodeHead(config)
print(model.encoder, decoder_head)
vit = torchvision.models.vit_b_32(pretrained=False)
print(vit)
model = MAE(model, patch_size=model.config.hidden_sizes[0], sequence_length=)
#### IMPORTANT
# sequence_length = ((image_size**2) / (patch_size**2)) + 1
# 50 = ((224**2) / (32**2)) + 1

transform = MAETransform()
# we ignore object detection annotations by setting target_transform to return 0
dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=lambda t: 0,
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator)
trainer.fit(model=model, train_dataloaders=dataloader)