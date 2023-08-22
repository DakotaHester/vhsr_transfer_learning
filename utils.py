from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import segmentation_models_pytorch as smp
import numpy as np
from torchmetrics import classification
import torchmetrics
from torchmetrics.aggregation import MeanMetric
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import pickle

class TrainHistory(object):
    
    def __init__(self, n_classes, name='train_history', device='cpu', mode='multiclass', ignore_index=0):
        self._history_dict = {
            'epoch': [],
            'phase': [],
            'loss': [],
            'overall_acc': [],
            'prod_acc': [],
            'user_acc': [],
            'f1': [],
        }
        if mode == 'multiclass': self._history_dict['iou'] = []
        self._writer = SummaryWriter(log_dir=f'runs/{name}')
        
        self.current_phase = 'train'
        self.current_epoch = 1
        # print(n_classes)

        self.best_epoch = 0
        self.best_val_loss = float('inf')
        
        self.task = 'multiclass' if mode == 'multiclass' else 'multilabel'
        tm_kwargs = {
            'task': self.task,
            'num_classes': n_classes,
            'num_labels': n_classes,
            # 'ignore_index': 0 if self.task == 'multiclass' else None,
            'average': 'macro',
        }
        # print(self.task)
        
        self.acc_metric = torchmetrics.Accuracy(**tm_kwargs).to(device)
        self.prod_acc_metric = torchmetrics.Precision(**tm_kwargs).to(device)
        self.user_acc_metric = torchmetrics.Recall(**tm_kwargs).to(device)
        self.f1_metric = torchmetrics.F1Score(**tm_kwargs).to(device)
        if self.task == 'multiclass': self.iou_metric = torchmetrics.JaccardIndex(**tm_kwargs).to(device)
        self.temp_loss = MeanMetric(**tm_kwargs)
    
    def _next_phase(self):
        if self.current_phase == 'train':
            self.current_phase = 'val'
            return False
        else:
            self.current_phase = 'train'
            self.current_epoch += 1
            return True
    
    def _update_tensorboard(self):
        self._history_df = pd.DataFrame(self._history_dict).set_index(['epoch', 'phase'])
        self._writer.add_scalar('Loss/train', self._history_df.loc[(self.current_epoch, 'train'), 'loss'], self.current_epoch)
        self._writer.add_scalar('Loss/val', self._history_df.loc[(self.current_epoch, 'val'), 'loss'], self.current_epoch)
        self._writer.add_scalar('Acc/train', self._history_df.loc[(self.current_epoch, 'train'), 'overall_acc'], self.current_epoch)
        self._writer.add_scalar('Acc/val', self._history_df.loc[(self.current_epoch, 'val'), 'overall_acc'], self.current_epoch)
        self._writer.add_scalar('Prod_acc/train', self._history_df.loc[(self.current_epoch, 'train'), 'prod_acc'], self.current_epoch)
        self._writer.add_scalar('Prod_acc/val', self._history_df.loc[(self.current_epoch, 'val'), 'prod_acc'], self.current_epoch)
        self._writer.add_scalar('User_acc/train', self._history_df.loc[(self.current_epoch, 'train'), 'user_acc'], self.current_epoch)
        self._writer.add_scalar('User_acc/val', self._history_df.loc[(self.current_epoch, 'val'), 'user_acc'], self.current_epoch)
        self._writer.add_scalar('F1/train', self._history_df.loc[(self.current_epoch, 'train'), 'f1'], self.current_epoch)
        self._writer.add_scalar('F1/val', self._history_df.loc[(self.current_epoch, 'val'), 'f1'], self.current_epoch)
        if self.task == 'multiclass':
            self._writer.add_scalar('IoU/train', self._history_df.loc[(self.current_epoch, 'train'), 'iou'], self.current_epoch)
            self._writer.add_scalar('IoU/val', self._history_df.loc[(self.current_epoch, 'val'), 'iou'], self.current_epoch)
        
    def _update_best_epoch(self):
        if self._history_df.loc[(self.current_epoch, 'val'), 'loss'] < self.best_val_loss:
            self.best_epoch = self.current_epoch
            self.best_val_loss = self._history_df.loc[(self.current_epoch, 'val'), 'loss']
    
    def minibatch_update(self, loss, y_hat, y):
        loss = self.temp_loss.update(loss.item())
        
        if self.task == 'multiclass': y_hat = torch.argmax(y_hat, dim=1)
        
        acc = self.acc_metric(y_hat, y)
        pa = self.prod_acc_metric(y_hat, y)
        ua = self.user_acc_metric(y_hat, y)
        f1 = self.f1_metric(y_hat, y)
        if self.task == 'multiclass': 
            iou = self.iou_metric(y_hat, y)
            return loss, acc, pa, ua, f1, iou
        return loss, acc, pa, ua, f1
    
    def update(self):
        loss = self.temp_loss.compute().item()
        self._history_dict['epoch'].append(self.current_epoch)
        self._history_dict['phase'].append(self.current_phase)
        self._history_dict['loss'].append(loss)
        self._history_dict['overall_acc'].append(self.acc_metric.compute().item())
        self._history_dict['prod_acc'].append(self.prod_acc_metric.compute().item())
        self._history_dict['user_acc'].append(self.user_acc_metric.compute().item())
        self._history_dict['f1'].append(self.f1_metric.compute().item())
        if self.task == 'multiclass': self._history_dict['iou'].append(self.iou_metric.compute().item())
                
        self.temp_loss.reset()
        self.acc_metric.reset()
        self.prod_acc_metric.reset()
        self.user_acc_metric.reset()
        self.f1_metric.reset()
        if self.task == 'multiclass': self.iou_metric.reset()
        
        if self.current_phase == 'val': 
            self._update_tensorboard()
            self._update_best_epoch()

        self._next_phase()
        return loss
    
    def save(self, path):
        df = pd.DataFrame(self._history_dict)
        df.to_csv(path, index=False)

# https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# def visualize_inference(X, y, y_hat):
#     X = X[0].cpu().numpy()
    
#     X_r = X[0]
#     X_g = X[1]
#     X_b = X[2]
#     X_nir = X[3]
    
#     X_rgb = np.stack([X_r, X_g, X_b], axis=2)
#     X_

class MultiLabelClassificationModel(nn.Module):
    def __init__(self, encoder=None, n_classes=7):
        super().__init__()
        self.encoder = encoder
        
        # get size of output channel from encoder
        self.encoder.eval()
        dry_x = torch.randn(1, 4, 256, 256)
        dry_x = self.encoder(dry_x)[-1].detach().cpu().numpy()
        out_channels = np.prod(dry_x.shape)
        print(out_channels, dry_x.shape)
        self.encoder.train()
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
            # nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.classification_head(x)
        return x

class AccuracyMetrics:
    
    def __init__(self, n_classes, device) -> None:
        
        tm_kwargs = {
            'task': 'multiclass',
            'num_classes': n_classes,
            'average': 'macro',
        }
        
        self.acc_metric = torchmetrics.Accuracy(**tm_kwargs).to(device)
        self.prod_acc_metric = torchmetrics.Precision(**tm_kwargs).to(device)
        self.user_acc_metric = torchmetrics.Recall(**tm_kwargs).to(device)
        self.f1_metric = torchmetrics.F1Score(**tm_kwargs).to(device)
        self.iou_metric = torchmetrics.JaccardIndex(**tm_kwargs).to(device)
    
    def update(self, y_hat, y):
        
        y_hat = torch.argmax(y_hat, dim=1)
        
        acc = self.acc_metric(y_hat, y)
        pa = self.prod_acc_metric(y_hat, y)
        ua = self.user_acc_metric(y_hat, y)
        f1 = self.f1_metric(y_hat, y)
        iou = self.iou_metric(y_hat, y)
    
    def save(self, path):
        
        metrics_dict = {
            'acc': self.acc_metric.compute().item(),
            'prod_acc': self.prod_acc_metric.compute().item(),
            'user_acc': self.user_acc_metric.compute().item(),
            'f1': self.f1_metric.compute().item(),
            'iou': self.iou_metric.compute().item(),
        }
        metrics_df = pd.DataFrame(metrics_dict, index=[0])
        
        with open(path, 'wb') as f:
            metrics_df.to_csv(f, index=False)
        
        
def final_accuracy_evaluation(y_hat, y, save_dir, n_classes, device):
    
    y_hat = torch.argmax(y_hat, dim=1)
    
    tm_kwargs = {
        'task': 'multiclass',
        'num_classes': n_classes,
        'num_labels': n_classes,
    }
    acc_metric = torchmetrics.Accuracy(**tm_kwargs).to(device)
    prod_acc_metric = torchmetrics.Precision(**tm_kwargs).to(device)
    user_acc_metric = torchmetrics.Recall(**tm_kwargs).to(device)
    f1_metric = torchmetrics.F1Score(**tm_kwargs).to(device)
    iou_metric = torchmetrics.JaccardIndex(**tm_kwargs).to(device)
    
    acc = acc_metric(y_hat, y)
    pa = prod_acc_metric(y_hat, y)
    ua = user_acc_metric(y_hat, y)
    f1 = f1_metric(y_hat, y)
    iou = iou_metric(y_hat, y)
    
    metrics_dict = {
        'acc': acc.compute().item(),
        'prod_acc': acc.compute().item(),
        'user_acc': acc.compute().item(),
        'f1': acc.compute().item(),
        'iou': acc.compute().item(),
    }
    
    metrics_dict = {
        'acc': acc.compute().item(),
        'prod_acc': acc.compute().item(),
        'user_acc': acc.compute().item(),
        'f1': acc.compute().item(),
        'iou': acc.compute().item(),
    }
    
    with open(save_dir, 'wb') as f:
        metrics_df = pd.DataFrame(metrics_dict, index=[0])
        metrics_df.to_csv(f, index=False)