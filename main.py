import data
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchinfo import summary
from torchviz import make_dot
from tqdm import tqdm, trange
from utils import TrainHistory, EarlyStopper
import torchvision
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 1000
BATCH_SIZE = 16
ACCUM_STEPS = 1
LEARNING_RATE = 1e-5
LABEL_SMOOTHING = 0.05
PATIENCE = 100
SEED = 1701
NUM_CLASSES = 8
NAME = 'resnet101_unet_multi_class_pretraining'
MODE = 'multilabel'

torch.manual_seed(SEED)

class MultiLabelClassificationModel(nn.Module):
    def __init__(self, encoder=None, n_classes=7):
        super().__init__()
        self.encoder = None
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder.out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classification_head(x)
        return x

def main() -> None:
    # torchvision.disable_beta_transforms_warning().warnings.warn(_BETA_TRANSFORMS_WARNING) # disable warning about beta transforms
    # test loading the dataset
    
    
    lcn_dataset_paths = data.get_file_paths('/scratch/lcn_global_prepped')
    lcn_train_paths, lcn_val_paths = data.split_files(lcn_dataset_paths) # default validation split of 0.2
    
    if is_multilabel(MODE):
        data.LandCoverNetMulticlassDataset(lcn_train_paths, mode='train')
        data.LandCoverNetMulticlassDataset(lcn_val_paths, mode='val')
    else:
        lcn_train = data.LandCoverNetDataset(lcn_train_paths, mode='train')
        lcn_val = data.LandCoverNetDataset(lcn_val_paths, mode='val')
    
    semantic_seg_autoencoder_model = smp.Unet(
        encoder_name='resnet101',
        encoder_weights=None,
        # encoder_output_stride=8,
        in_channels=4,
        classes=8,
    ).to(device)
    
    if is_multilabel(MODE):
        encoder = semantic_seg_autoencoder_model.encoder
        encoder_empty_weights = encoder.state_dict()
        model = MultiLabelClassificationModel(encoder=encoder, n_classes=8).to(device)
    else:
        model = semantic_seg_autoencoder_model.to(device)
        encoder_empty_weights = model.encoder.state_dict()
    
    
    
    # model.encoder.requires_grad_(False) # freeze the encoder
    # model = torch.compile(model).to(device)
    
    # model.to(device)
    summary(model, input_size=(64, 4, 256, 256), device=device)
    # X = torch.randn(64, 4, 256, 256).to(device)
    # y = model(X)
    # model_arch = make_dot(y.mean(), params=dict(model.named_parameters())).render('model_arch', format='png')
    
    # scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    loss_fn = torch.nn.CrossEntropyLoss(weight=lcn_train.weights, label_smoothing=LABEL_SMOOTHING, ignore_index=0 if not is_multilabel(MODE) else None)
    history = TrainHistory(n_classes=8, device=device, mode=MODE)
    stopper = EarlyStopper(patience=PATIENCE)
    
    lcn_train_loader = torch.utils.data.DataLoader(lcn_train, batch_size=BATCH_SIZE // ACCUM_STEPS, shuffle=True, pin_memory=True)
    lcn_val_loader = torch.utils.data.DataLoader(lcn_val, batch_size=BATCH_SIZE // ACCUM_STEPS, shuffle=True, pin_memory=True)
    
    for epoch in range(1, EPOCHS+1):
        
        # train step
        tqdm_lcn_train_loader = tqdm(lcn_train_loader, desc=f'Epoch {epoch} train', unit='batch')
        model.train()
        for batch_idx, (X, y) in enumerate(tqdm_lcn_train_loader):
            X = X.to(device)
            y = y.to(device)
            # with torch.cuda.amp.autocast():
            #     y_hat = model(X)
            #     loss = loss_fn(y_hat, y) / ACCUM_STEPS
            
            y_hat = model(X).softmax(dim=1)
            loss = loss_fn(y_hat, y) / ACCUM_STEPS
            history.minibatch_update(loss, y_hat.argmax(dim=1), y)

            # scaler.scale(loss).backward()
            loss.backward()
            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(lcn_train_loader):
                optimizer.step()
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.zero_grad()
            tqdm_lcn_train_loader.set_postfix(loss=history.temp_loss.compute().item(), iou=history.iou_metric.compute().item())
        
        history.update()
        
        # val step
        model.eval()
        tqdm_lcn_val_loader = tqdm(lcn_val_loader, unit='batch', desc=f'Epoch {epoch} val')
        for batch_idx, (X, y) in enumerate(tqdm_lcn_val_loader):
            X = X.to(device)
            y = y.to(device)
            # with torch.cuda.amp.autocast():
            y_hat = model(X).softmax(dim=1)
            loss = loss_fn(y_hat, y) / ACCUM_STEPS
            history.minibatch_update(loss, y_hat.argmax(dim=1), y)
            tqdm_lcn_val_loader.set_postfix(val_loss=history.temp_loss.compute().item(), val_iou=history.iou_metric.compute().item())
        
        val_loss = history.update()
        if stopper.early_stop(val_loss):
            print('Early stopping at epoch', epoch)
            break
        if history.best_epoch == epoch:
            torch.save(model.state_dict(), f'./resnet101_unet_{epoch}_{val_loss}.pt')
        history.save('resnet101_unet_LCN.csv')

def is_multilabel(mode):
    return mode == 'multilabel'

if __name__ == "__main__":
    main()
