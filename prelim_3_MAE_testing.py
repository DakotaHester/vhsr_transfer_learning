# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

# https://docs.lightly.ai/self-supervised-learning/examples/mae.html
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.models import utils
from lightly.models.modules import masked_autoencoder
from lightly.transforms.mae_transform import MAETransform

import data
import os
import random
import pickle

import timm

class MAE(pl.LightningModule):
    def __init__(self, vit=torchvision.models.vit_b_16(pretrained=False)):
        super().__init__()

        decoder_dim = 512
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
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

# MAE backbone code
 """Backbone for the Masked Autoencoder model [0].

    Converts images into patches and encodes them. Code inspired by [1].
    Note that this implementation uses a learned positional embedding while [0]
    uses a fixed positional embedding.

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae
    - [2]: Early Convolutions Help Transformers See Better, 2021, https://arxiv.org/abs/2106.14881.

    Attributes:
        image_size:
            Input image size.
        patch_size:
            Width and height of the image patches. image_size must be a multiple
            of patch_size.
        num_layers:
            Number of transformer blocks.
        num_heads:
            Number of attention heads.
        hidden_dim:
            Dimension of the input and output tokens.
        mlp_dim:
            Dimension of the MLP in the transformer block.
        dropout:
            Percentage of elements set to zero after the MLP in the transformer.
        attention_dropout:
            Percentage of elements set to zero after the attention head.
        num_classes:
            Number of classes for the classification head. Currently not used.
        representation_size:
            If specified, an additional linear layer is added before the
            classification head to change the token dimension from hidden_dim
            to representation_size. Currently not used.
        norm_layer:
            Callable that creates a normalization layer.
        conv_stem_configs:
            If specified, a convolutional stem is added at the beggining of the
            network following [2]. Not used in the original Masked Autoencoder
            paper [0].

    """

def from_timm_vit(
    timm_vit: timm.models.vision_transformer.VisionTransformer, 
    image_size: int = 224,
) -> masked_autoencoder.MAEBackbone:
    
    patch_size = timm_vit.patch_embed.proj.kernel_size[0]
    num_layers = len(timm_vit.blocks)
    hidden_dim = timm_vit.head.out_features
    mlp_dim =  timm_vit.blocks[0].mlp.fc1.out_features
    
    
    mae_backbone = masked_autoencoder.MAEBackbone(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=1,
        
    

vit = torchvision.models.vit_b_32(
    weights=None,
    image_size=256,
)
model = MAE(vit=vit)

transform = MAETransform()
# we ignore object detection annotations by setting target_transform to return 0
# dataset = torchvision.datasets.VOCDetection(
#     "datasets/pascal_voc",
#     download=True,
#     transform=transform,
#     target_transform=lambda t: 0,
# )
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")


root_file_path = os.path.join('/', 'scratch', 'chesapeake_bay_lulc', 'sampled', '512')
PRE_TRAINING_SAMPLES = 10000
FINE_TUNING_SAMPLES = 1000
VALIDATION_SAMPLES = 250
TEST_SAMPLES = 250

# if os.path.exists(f'pretrain_dataset_{PRE_TRAINING_SAMPLES}.pkl'):
#     pretrain_dataset = pickle.load(open(f'pretrain_dataset_{PRE_TRAINING_SAMPLES}.pkl', 'rb'))
# else:
#     data_paths = data.get_cpblulc_file_paths(root_file_path)
#     print(f'found {len(data_paths)} total samples')
#     data_paths = data.remove_nodata_samples(data_paths, nodata_value=15, n_samples=PRE_TRAINING_SAMPLES)
#     subset = random.sample(data_paths, k=PRE_TRAINING_SAMPLES)

pretrain_paths = pickle.load(open(f'CPBLULC_256_pretrain_samples.pkl', 'rb'))
pretrain_dataset = data.ChesapeakeBayDataset(
    pretrain_paths,
    mode='train',
    device='cpu',
)

    # # quick code to dump the pretrain_dataset to a file
    # pickle.dump(pretrain_dataset, open(f'pretrain_dataset_{PRE_TRAINING_SAMPLES}.pkl', 'wb'))

dataloader = torch.utils.data.DataLoader(
    pretrain_dataset,
    batch_size=1,
    shuffle=True,
    drop_last=True,
    # num_workers=1,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator)
trainer.fit(model=model, train_dataloaders=dataloader)