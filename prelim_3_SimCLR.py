# from: https://docs.lightly.ai/self-supervised-learning/examples/simclr.html
# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
import torchvision
from torch import nn
import timm

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.data import LightlyDataset

from data import CPBLULC_SSL_Dataset, get_cpblulc_file_paths
import os
import pickle
import random

random.seed(1701)

class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


resnet = timm.models.resnet.resnet18(
    in_chans=4,
)
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimCLR(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = SimCLRTransform(input_size=224)
# dataset = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform
# )
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder", transform=transform)

print('getting data paths')
if False:
    dataset_paths = get_cpblulc_file_paths('/scratch/chesapeake_bay_lulc/sampled/224')
    random.shuffle(dataset_paths)
    print(f'found {len(dataset_paths)} total samples')

    # 400k for pretraining, 20k for training, 8193 each for validation and test
    pretrain_dataset_paths = dataset_paths[:400000]
    train_dataset_paths = dataset_paths[400000:400000+20000]
    val_dataset_paths = dataset_paths[400000+20000:400000+20000+8193]
    test_dataset_paths = dataset_paths[400000+20000+8193:400000+20000+8193+8193]
    
    pickle.dump(pretrain_dataset_paths, open(f'CPBLULC_224_pretrain_dataset_paths.pkl', 'wb'))
    pickle.dump(train_dataset_paths, open(f'CPBLULC_224_train_dataset_paths.pkl', 'wb'))
    pickle.dump(val_dataset_paths, open(f'CPBLULC_224_val_dataset_paths.pkl', 'wb'))
    pickle.dump(test_dataset_paths, open(f'CPBLULC_224_test_dataset_paths.pkl', 'wb'))
    pickle.dump(dataset_paths, open(f'CPBLULC_224_dataset_paths.pkl', 'wb'))
else:
    pretrain_dataset_paths = pickle.load(open(f'CPBLULC_224_pretrain_dataset_paths.pkl', 'rb'))


print('creating dataset')
dataset = CPBLULC_SSL_Dataset(
    paths=pretrain_dataset_paths,
    bands=[1, 2, 3, 4],
    mode='train',
    transform=transform,
)
# lightly_dataset = LightlyDataset.from_torch_dataset(dataset, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        print(len(batch))
        print(batch[0].shape)
        x0, x1 = batch[0]
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")