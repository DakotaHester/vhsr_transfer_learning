# preliminary test 2
# model selection
# datasets:
#   NYC
#   CPBLULC
# models:
#   UNet
#   DeepLabV3+
#   PAN
#   XGBoost - separate code for XGboost
# encoders:
#   ResNet101
#   MAXception (from DeepLabv3+)
#   EfficientNet-b7
# losses:
#   CrossEntropyLoss
#   DICELoss
#   FocalLoss


import os
import data
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchinfo import summary
from torchviz import make_dot
from tqdm import tqdm, trange
from utils import TrainHistory, EarlyStopper, MultiLabelClassificationModel
import torchvision
import random
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda': torch.backends.cudnn.benchmark = True

def main(args) -> None:
    
    name = 'prelim2_dev_test'
    
    DATASETS = ['cpblulc', 'nyc']
    MODELS = ['unet', 'deeplabv3plus', 'pan']
    ENCODERS = ['resnet101', 'maxception', 'efficientnet-b7']
    LOSSES = ['crossentropy', 'dice', 'focal']
    
    INPUT_RESOLUTIONS = {
        'resnet101': 224,
        'maxception': 299,
        'efficientnet-b7': 600,
    }
    
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    ACCUM_STEPS = args.accum_steps
    LEARNING_RATE = args.learning_rate
    LABEL_SMOOTHING = args.label_smoothing
    PATIENCE = args.patience
    SEED = args.seed
    NUM_CLASSES = args.num_classes
    NAME = args.name
    NUM_WORKERS = 8
    # torchvision.disable_beta_transforms_warning().warnings.warn(_BETA_TRANSFORMS_WARNING) # disable warning about beta transforms
    # test loading the dataset
    
    torch.manual_seed(SEED)
    
    os.mkdir(NAME, exist_ok=True)
    os.mkdir(f'{NAME}/models', exist_ok=True)
    
    for dataset in DATASETS:
        for model in MODELS:
            for encoder in ENCODERS:
                
                if encoder
                
                for loss in LOSSES:
                    if dataset == 'cpblulc':
                        pass     
                    if dataset == 'nyc':
                        pass
    
    
    
    is_multilabel = MODE == 'multilabel'
    print(is_multilabel)
    n_classes = 7 if is_multilabel else 8
    
    lcn_dataset_paths = data.get_file_paths('/scratch/lcn_global_prepped')
    lcn_train_paths, lcn_val_paths = data.split_files(lcn_dataset_paths) # default validation split of 0.2
    
    if args.use_val_set:
        random.shuffle(lcn_val_paths)
        lcn_train_paths = lcn_val_paths[:args.n_samples]
        lcn_val_paths = lcn_val_paths[args.n_samples:]
    
    
    
    if os.path.exists(f'{NAME}/{NAME}_lcn_train.pkl') and os.path.exists(f'{NAME}/{NAME}_lcn_val.pkl'):
        with open(f'{NAME}/{NAME}_lcn_train.pkl', 'rb') as f:
            lcn_train = pickle.load(f)
        with open(f'{NAME}/{NAME}_lcn_val.pkl', 'rb') as f:
            lcn_val = pickle.load(f)
    else:    
        if is_multilabel:
            lcn_train = data.LandCoverNetMultilabelDataset(lcn_train_paths, mode='train')
            lcn_val = data.LandCoverNetMultilabelDataset(lcn_val_paths, mode='val')
        else:
            lcn_train = data.LandCoverNetDataset(lcn_train_paths, mode='train')
            lcn_val = data.LandCoverNetDataset(lcn_val_paths, mode='val')
            
        with open(f'{NAME}/{NAME}_lcn_train.pkl', 'wb') as f:
            pickle.dump(lcn_train, f)
        with open(f'{NAME}/{NAME}_lcn_val.pkl', 'wb') as f:
            pickle.dump(lcn_val, f)
    
    semantic_seg_autoencoder_model = smp.Unet(
        encoder_name='resnet101',
        encoder_weights='imagenet' if args.use_imagenet else None,
        # encoder_output_stride=8,
        in_channels=4,
        classes=n_classes,
    )
    
    if args.encoder_params:
        ml_model = MultiLabelClassificationModel(encoder=semantic_seg_autoencoder_model.encoder, n_classes=7)
        full_weights = torch.load(args.encoder_params)
        ml_model.load_state_dict(full_weights)
        encoder = ml_model.encoder
        encoder_weights = encoder.state_dict()
        # encoder_empty_weights.update(encoder_weights)
        semantic_seg_autoencoder_model.encoder.load_state_dict(encoder_weights)
    
    if is_multilabel:
        encoder = semantic_seg_autoencoder_model.encoder
        print(encoder)
        encoder_empty_weights = encoder.state_dict()
        model = MultiLabelClassificationModel(encoder=encoder, n_classes=n_classes).to(device)
    else:
        model = semantic_seg_autoencoder_model.to(device)
        encoder_empty_weights = model.encoder.state_dict()
        
    if args.freeze_encoder:
        if is_multilabel:
            raise UserWarning('Freezing the encoder is not reccomended on multilabel problems')
        model.encoder.requires_grad_(False)
    # model.encoder.requires_grad_(False) # freeze the encoder
    # model = torch.compile(model).to(device)
    
    # model.to(device)
    # summary(model, input_size=(64, 4, 256, 256), device=device)
    print(model)
    # X = torch.randn(64, 4, 256, 256).to(device)
    # y = model(X)
    # model_arch = make_dot(y.mean(), params=dict(model.named_parameters())).render('model_arch', format='png')
    
    # scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    if is_multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss(weight=lcn_train.weights.to(device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=lcn_train.weights.to(device), label_smoothing=LABEL_SMOOTHING, ignore_index=0)
    history = TrainHistory(name=NAME, n_classes=n_classes, device=device, mode=MODE)
    stopper = EarlyStopper(patience=PATIENCE)
    
    lcn_train_loader = torch.utils.data.DataLoader(lcn_train, batch_size=BATCH_SIZE // ACCUM_STEPS, shuffle=True)
    lcn_val_loader = torch.utils.data.DataLoader(lcn_val, batch_size=BATCH_SIZE // ACCUM_STEPS, shuffle=True)
    
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
            
            y_hat = model(X)
            if not is_multilabel: y_hat = y_hat.softmax(dim=1) # multiclass segmentation
            
            loss = loss_fn(y_hat, y) / ACCUM_STEPS
            history.minibatch_update(loss, y_hat, y)

            # scaler.scale(loss).backward()
            loss.backward()
            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(lcn_train_loader):
                optimizer.step()
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.zero_grad()
            tqdm_lcn_train_loader.set_postfix(loss=history.temp_loss.compute().item(), f1=history.f1_metric.compute().item())
        
        history.update()
        
        # val step
        model.eval()
        tqdm_lcn_val_loader = tqdm(lcn_val_loader, unit='batch', desc=f'Epoch {epoch} val')
        for batch_idx, (X, y) in enumerate(tqdm_lcn_val_loader):
            X = X.to(device)
            y = y.to(device)
            # with torch.cuda.amp.autocast():
            y_hat = model(X)
            if not is_multilabel: y_hat = y_hat.softmax(dim=1)
            loss = loss_fn(y_hat, y) / ACCUM_STEPS
            # print(X.shape, y.shape, y_hat.shape)
            history.minibatch_update(loss, y_hat, y)
            tqdm_lcn_val_loader.set_postfix(val_loss=history.temp_loss.compute().item(), val_f1=history.f1_metric.compute().item())
        
        val_loss = history.update()
        if stopper.early_stop(val_loss):
            print('Early stopping at epoch', epoch)
            break
        if history.best_epoch == epoch:
            torch.save(model.state_dict(), f'./{NAME}/models/{NAME}_{epoch}_{val_loss}.pt')
        history.save(f'{NAME}/{NAME}_history.csv')

def argparser():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a semantic segmentation model on the LandCoverNet dataset')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--accum_steps', type=int, default=1, help='number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--patience', type=int, default=50, help='patience for early stopping')
    parser.add_argument('--seed', type=int, default=1701, help='random seed')
    parser.add_argument('--num_classes', type=int, default=8, help='number of classes')
    parser.add_argument('--name', type=str, default='resnet101_unet_multi_class_pretraining', help='name of the model')
    parser.add_argument('--mode', type=str, default='multiclass', help='multiclass or multilabel')
    parser.add_argument('--freeze_encoder', dest='freeze_encoder', action='store_true', help='freeze the encoder')
    parser.add_argument('--encoder_params', type=str, default=None, help='path to encoder parameters')
    parser.add_argument('--imagenet', dest='use_imagenet', action='store_true', help='use imagenet pretrained weights in encoder')
    parser.add_argument('--use_val_set', dest='use_val_set', action='store_true', help='use the validation set for training')
    parser.add_argument('--n_samples', type=int, default=None, help='number of samples to use from the dataset')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = argparser()
    main(args)
