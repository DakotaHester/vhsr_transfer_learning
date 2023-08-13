import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt



def main():
    path = "./"

    all_metrics_file_paths = glob.glob(path + "*n_samples_128/*metrics.csv")
    all_metrics_df = pd.DataFrame()

    for file_path in all_metrics_file_paths:
        
        run_info_dict = get_run_info_dict_from_file_path(file_path)
        if run_info_dict['resolution'] == '1024': continue
        
        metrics_df = pd.read_csv(file_path)
        for key, value in run_info_dict.items():
            metrics_df[key] = value
            
        all_metrics_df = pd.concat([all_metrics_df, metrics_df])
    
    all_metrics_df.set_index(['dataset_name', 'model', 'encoder', 'loss', 'resolution', 'n_samples'], inplace=True)
    all_metrics_df.sort_index(inplace=True)
    all_metrics_df.to_csv('all_metrics.csv')
    
    all_history_file_paths = glob.glob(path + "*n_samples_128/*history.csv")
    all_history_df = pd.DataFrame()
    
    for file_path in all_history_file_paths:
        
        run_info_dict = get_run_info_dict_from_file_path(file_path)
        if run_info_dict['resolution'] == '1024': continue
        
        history_df = pd.read_csv(file_path)
        for key, value in run_info_dict.items():
            history_df[key] = value
            
        all_history_df = pd.concat([all_history_df, history_df])
    
    all_history_df.set_index(['dataset_name', 'model', 'encoder', 'loss', 'resolution', 'n_samples', 'epoch', 'phase'], inplace=True)
    all_history_df.to_csv('all_history.csv')
    # print(all_metrics_df.sort_values(by=['iou'], ascending=False).head(10))
    
    
    
    # print(all_metrics_df.groupby(['model', 'encoder', 'loss']).mean()[['acc', 'f1', 'iou']].plot.bar())
    # print(all_metrics_df.groupby(['model']).mean()[['acc', 'f1', 'iou']])
    # print(all_metrics_df.groupby(['encoder']).mean()[['acc', 'f1', 'iou']])
    # print(all_metrics_df.groupby(['loss']).mean()[['acc', 'f1', 'iou']])
    # print(all_metrics_df.groupby(['resolution']).mean()[['acc', 'f1', 'iou']])
    
    all_metrics_df.groupby(['model', 'encoder', 'loss']).mean()[['acc', 'f1', 'iou']].plot.bar(subplots=True)
    plt.savefig('model_encoder_loss.png')
    
    all_metrics_df.reset_index(inplace=True)
    
    fig, ax = plt.subplots(3, 3, figsize=(7.5, 7.5), sharey=True)
    for i, model in enumerate(all_metrics_df['model'].unique()):
        
        df1 = all_metrics_df[(all_metrics_df['model'] == model)]
        # df = df.groupby(['encoder', 'loss']).mean(numeric_only=True)[['acc', 'f1', 'iou']]
        print(df1)
        
        ax[i][0].set_ylabel(model, size='large')
        ax[i][0].set_ylim([0, 1])
        ax[i][0].set_yticks(np.arange(0, 1.1, 0.2))
        
        for j, encoder in enumerate(all_metrics_df['encoder'].unique()):
            
            ax[0][j].set_title(encoder, size='large')
            
            df2 = df1.loc[df1['encoder'] == encoder]
            df2 = df2.groupby(['loss']).mean(numeric_only=True)[['acc', 'f1', 'iou']]
            print(f'{model}_{encoder}')
            print(df2.index)
            
            x = np.arange(len(df2.index))
            width = 0.25
            
            ax[i][j].bar(x - width, df2['acc'], width, label='acc')
            ax[i][j].bar(x, df2['f1'], width, label='f1')
            ax[i][j].bar(x + width, df2['iou'], width, label='iou')
            
            ax[i][j].set_xticks(range(len(df2.index)), df2.index)
            # ax[i][j].hlines([0.2, 0.4, 0.6, 0.8], 0-width, 2+width, colors='black', linestyles='solid', linewidth=0.25)
            
            
            # for k, loss in enumerate(all_metrics_df['loss'].unique()):
                
            #     data = df.loc[encoder, loss]
            #     print(data)
                
            #     ax[i][j].set_title(f'{encoder} {loss}')
            #     data.plot.bar(ax=ax[i][j])
                
            #     # df = df.groupby(['resolution']).mean()[['acc', 'f1', 'iou']]

    ax[0][2].legend()
    
    fig.tight_layout()
    plt.savefig(f'model_performance_aggregate.png')
    plt.close()
    
    all_metrics_df.set_index(['dataset_name', 'model', 'encoder', 'loss', 'resolution', 'n_samples'], inplace=True)
    all_metrics_df.sort_index(inplace=True)
    
    print(all_metrics_df.groupby(['model']).mean()[['acc', 'f1', 'iou']].sort_values(by=['iou', 'f1', 'acc'], inplace=False, ascending=False))
    
    all_metrics_df.groupby(['model', 'encoder', 'loss']).mean()[['acc', 'f1', 'iou']].sort_values(by=['iou', 'f1', 'acc'], inplace=False, ascending=False).to_csv('model_encoder_loss.csv')
    
    all_metrics_df.groupby(['model']).mean()[['acc', 'f1', 'iou']].sort_values(by=['iou', 'f1', 'acc'], inplace=False, ascending=False).to_csv('model.csv')
    all_metrics_df.groupby(['encoder']).mean()[['acc', 'f1', 'iou']].sort_values(by=['iou', 'f1', 'acc'], inplace=False, ascending=False).to_csv('encoder.csv')
    all_metrics_df.groupby(['loss']).mean()[['acc', 'f1', 'iou']].sort_values(by=['iou', 'f1', 'acc'], inplace=False, ascending=False).to_csv('loss.csv')
    all_metrics_df.groupby(['resolution']).mean()[['acc', 'f1', 'iou']].sort_values(by=['iou', 'f1', 'acc'], inplace=False, ascending=False).to_csv('resolution.csv')
    all_metrics_df.groupby(['dataset_name']).mean()[['acc', 'f1', 'iou']].sort_values(by=['iou', 'f1', 'acc'], inplace=False, ascending=False).to_csv('dataset.csv')
    
    all_metrics_df.groupby(['model', 'encoder']).mean()[['acc', 'f1', 'iou']].sort_values(by=['iou', 'f1', 'acc'], inplace=False, ascending=False).to_csv('model_encoder.csv')
    all_metrics_df.groupby(['model', 'loss']).mean()[['acc', 'f1', 'iou']].sort_values(by=['iou', 'f1', 'acc'], inplace=False, ascending=False).to_csv('model_loss.csv')
    all_metrics_df.groupby(['encoder', 'loss']).mean()[['acc', 'f1', 'iou']].sort_values(by=['iou', 'f1', 'acc'], inplace=False, ascending=False).to_csv('encoder_loss.csv')
    

def get_run_info_dict_from_file_path(path_to_file: str) -> dict:
    
    filename = os.path.basename(path_to_file)
    filename_components_list = filename.split('_')
    
    run_info_dict = {
        'dataset_name': filename_components_list[0], 
        'model': filename_components_list[1], 
        'encoder': filename_components_list[2], 
        'loss': filename_components_list[3], 
        'resolution': filename_components_list[6], 
        'n_samples': filename_components_list[9], 
    }
    
    return run_info_dict

# def make_bar_plot(df, model):
    
#     barwidth = 0.25
#     fig, ax = plt.subplots(3, 3, figsize=(20, 20))
#     fig.suptitle(f'Model Metrics for {model}', fontsize=20)
    
#     for i, encoder in enumerate(['resnet101', 'tu-xception71', 'efficientnet-b7']):
#         print(encoder)
    
#         print(br1, br2, br3)

if __name__ == '__main__':
    main()