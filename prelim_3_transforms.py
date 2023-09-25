import torch
import torch.nn as nn
import prelim_3_utils as utils
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import seed, shuffle
import wandb

seed(1701)

EPOCHS = 10
LEARNING_RATE = 0.0001
LOADER_BATCH_SIZE = 32 # actual batch size = LOADER_BATCH_SIZE * GRAD_CACHE_STEPS
GRAD_CACHE_STEPS = 8
PATIENCE = 10
TEMPRATURE = 0.1
WARMUP_EPOCHS = 10
TRAIN_SPLIT = 0.8

run = wandb.init(
    project='SimCLR_CPB_test_1',
    config={
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'loader_batch_size': LOADER_BATCH_SIZE,
        'grad_cache_steps': GRAD_CACHE_STEPS,
        'batch size': LOADER_BATCH_SIZE * GRAD_CACHE_STEPS,
        'patience': PATIENCE,
        'temprature': TEMPRATURE,
        'warmup_epochs': WARMUP_EPOCHS
    }
)

print(f'Acutal batch size: {LOADER_BATCH_SIZE * GRAD_CACHE_STEPS}')

list_of_files = utils.get_list_of_files('prelim_2_subset_data_cpblulc')
shuffle(list_of_files)
n_train_samples = int(TRAIN_SPLIT * len(list_of_files))
train_files = list_of_files[:n_train_samples]
val_files = list_of_files[n_train_samples:]
train_dataset = utils.NAIP_CPB_Contrastive_Dataset(train_files, image_size=96)
val_dataset = utils.NAIP_CPB_Contrastive_Dataset(val_files, image_size=96)

model = utils.SimClrModel(timm.create_model('resnet18', pretrained=False, num_classes=2048))
print(model)

train_loader = DataLoader(
    train_dataset,
    batch_size=LOADER_BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=LOADER_BATCH_SIZE,
    shuffle=False,
    drop_last=True
)

print(
    f'training Dataset samples {len(train_dataset)}\ntraining DataLoader batches {len(train_loader)}\n'
)

loss_fn = utils.NT_Xent_Loss(TEMPRATURE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
warmump_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer, 
    lr_lambda=lambda x: x * 0.1 if x < WARMUP_EPOCHS else 1,
    verbose=True
)

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Using device {device}')
loss_fn = loss_fn.to(device)
model = model.to(device)

train_history = []
val_history = []
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(EPOCHS):
    
    # create cache for model output embeddings
    cache_z1, cache_z2 = [], []
    
    model.train()
    tqdm_train_loader = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} train', unit='batch')
    train_epoch_loss = 0.0
    for step, (X1, X2) in enumerate(tqdm_train_loader):
        X1, X2 = X1.to(device), X2.to(device)
        z1, z2 = model(X1), model(X2)
        
        # cache z1 and z2 for computing loss (must be computed over entire batch)
        cache_z1.append(z1)
        cache_z2.append(z2)
        
        # compute loss every GRAD_CACHE_STEPS
        if (step + 1) % GRAD_CACHE_STEPS == 0:
            z1, z2 = torch.cat(cache_z1, dim=0), torch.cat(cache_z2, dim=0)
            loss = loss_fn(z1, z2)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            tqdm_train_loader.set_postfix({'loss': loss.item() / (GRAD_CACHE_STEPS * LOADER_BATCH_SIZE)})
            wandb.log({'train batch loss': loss.item()})
            train_epoch_loss += loss.item()
            cache_z1, cache_z2 = [], []
    train_epoch_loss /= (len(train_loader) * LOADER_BATCH_SIZE)
    wandb.log({'train epoch loss': train_epoch_loss})
    
    cache_z1, cache_z2 = [], []
    tqdm_val_loader = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} val', unit='batch')
    val_epoch_loss = 0.0
    for step, (X1, X2) in enumerate(tqdm_val_loader):
        X1, X2 = X1.to(device), X2.to(device)
        z1, z2 = model(X1), model(X2)
        
        # cache z1 and z2 for computing loss (must be computed over entire batch)
        cache_z1.append(z1)
        cache_z2.append(z2)
        
        # compute loss every GRAD_CACHE_STEPS
        if (step + 1) % GRAD_CACHE_STEPS == 0:
            z1, z2 = torch.cat(cache_z1, dim=0), torch.cat(cache_z2, dim=0)
            val_loss = loss_fn(z1, z2)
            tqdm_val_loader.set_postfix({'val loss': val_loss.item() / (GRAD_CACHE_STEPS * LOADER_BATCH_SIZE)})
            wandb.log({'val batch loss': val_loss.item()})
            val_epoch_loss += val_loss.item()
            cache_z1, cache_z2 = [], []
    
    val_epoch_loss /= (len(val_loader) * LOADER_BATCH_SIZE)
    wandb.log({'val epoch loss': val_epoch_loss})
    
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')
    
    if epoch - best_epoch > PATIENCE:
        print(f'Early stopping at epoch {epoch+1}')
        break
    
    cache_z1, cache_z2 = [], []
    
    warmump_scheduler.step()