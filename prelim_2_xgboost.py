import xgboost as xgb
import numpy as np
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import rasterio as rio
import os
import random
import pickle
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

np.random.seed(1701)
random.seed(1701)

DATASETS = {
    'cpblulc': '/scratch/chesapeake_bay_lulc/sampled/',    
    'nyclc': '/scratch/nyc_lc/sampled/'
}
RESOLUTIONS = [224]
N_SAMPLES = 128
BANDS = [4, 1, 2]

def main():

    for dataset_name, dataset_path in DATASETS.items():
        
        n_classes = 8 if dataset_name == 'nyclc' else 6
        
        for resolution in RESOLUTIONS:
            
            NAME = f'XGBOOST_{dataset_name}_n_samples_{N_SAMPLES}_resolution_{resolution}'
            os.makedirs(NAME, exist_ok=True)
            
            if os.path.exists(f'./{NAME}/X_train.pkl') and \
                os.path.exists(f'./{NAME}/X_val.pkl') and \
                os.path.exists(f'./{NAME}/X_test.pkl') and \
                os.path.exists(f'./{NAME}/Y_train.pkl') and \
                os.path.exists(f'./{NAME}/Y_val.pkl') and \
                os.path.exists(f'./{NAME}/Y_test.pkl'):
                
                with open(f'./{NAME}/X_train.pkl', 'rb') as f:
                    X_train = pickle.load(f)
                with open(f'./{NAME}/X_val.pkl', 'rb') as f:
                    X_val = pickle.load(f)
                with open(f'./{NAME}/X_test.pkl', 'rb') as f:
                    X_test = pickle.load(f)
                with open(f'./{NAME}/Y_train.pkl', 'rb') as f:
                    Y_train = pickle.load(f)
                with open(f'./{NAME}/Y_val.pkl', 'rb') as f:
                    Y_val = pickle.load(f)
                with open(f'./{NAME}/Y_test.pkl', 'rb') as f:
                    Y_test = pickle.load(f)
                
            else:
            
                root_dir = os.path.join(dataset_path, str(resolution))
                
                if dataset_name == 'cpblulc': dataset_files = get_cpblulc_file_paths(root_dir)
                if dataset_name == 'nyclc': dataset_files = get_nyc_file_paths(root_dir)
                X, y = load_dataset(dataset_files, bands=BANDS, n_samples=3*N_SAMPLES, nodata_value=15)
                print(X.shape, y.shape)
                
                if dataset_name == 'cpblulc': y = y - 1 # cpblulc labels are 1-8, subtract 1 to make them 0-7
                
                n_points = resolution**2 * N_SAMPLES
                
                X_train, X_val, X_test = X[:n_points], X[n_points:2*n_points], X[2*n_points:3*n_points]
                Y_train, Y_val, Y_test = y[:n_points], y[n_points:2*n_points], y[2*n_points:3*n_points]
            
                with open(f'./{NAME}/X_train.pkl', 'wb') as f:
                    pickle.dump(X_train, f)
                with open(f'./{NAME}/X_val.pkl', 'wb') as f:
                    pickle.dump(X_val, f)
                with open(f'./{NAME}/X_test.pkl', 'wb') as f:
                    pickle.dump(X_test, f)
                with open(f'./{NAME}/Y_train.pkl', 'wb') as f:
                    pickle.dump(Y_train, f)
                with open(f'./{NAME}/Y_val.pkl', 'wb') as f:
                    pickle.dump(Y_val, f)
                with open(f'./{NAME}/Y_test.pkl', 'wb') as f:
                    pickle.dump(Y_test, f)
                    
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('xgb', xgb.XGBClassifier(
                    objective='multi:softmax',
                    num_class=n_classes,
                    verbosity=2,
                    # tree_method='gpu_hist',
                    # enable_categorical=True,
                    # n_jobs=2,
                    # device='gpu',
                    random_state=1701,
                    early_stopping_rounds=5,
                    tree_method='gpu_hist',
                    gpu_id=0,
                    predictor='gpu_predictor',
                ))
            ])
            
            # following params found using bayeesian optimization
            params = dict([('xgb__colsample_bytree', 1.0), ('xgb__gamma', 0.01), ('xgb__grow_policy', 'lossguide'), ('xgb__learning_rate', 0.18695810922200368), ('xgb__max_depth', 5), ('xgb__min_child_weight', 4), ('xgb__n_estimators', 1000), ('xgb__reg_alpha', 0.8596388382402865), ('xgb__reg_lambda', 0.2093415313907045), ('xgb__subsample', 0.8219059071873525)])
            pipe.set_params(**params)

            # param_grid = {
            #     'xgb__n_estimators': Integer(100, 1000),
            #     'xgb__max_depth': Integer(3, 10),
            #     'xgb__grow_policy': Categorical(['depthwise', 'lossguide']),
            #     'xgb__learning_rate': Real(0.01, 0.5),
            #     'xgb__subsample': Real(0.5, 1.0),
            #     'xgb__colsample_bytree': Real(0.5, 1.0),
            #     'xgb__gamma': Real(0.01, 1.0),
            #     'xgb__min_child_weight': Integer(1, 10),
            #     'xgb__reg_alpha': Real(0.01, 1.0),
            #     'xgb__reg_lambda': Real(0.01, 1.0),
            #     # 'xgb__eval_set': [(X_val, Y_val)],
            #     # 'xgb__scale_pos_weight': Real(1, 10),
            # }
            
            print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)
            
            if False: pass
            # if os.path.exists(f'./{NAME}/model.pkl'):
            #     pass
                # with open(f'./{NAME}/model.pkl', 'rb') as f:
                    # opt = pickle.load(f)
            else:
            
                # opt = BayesSearchCV(
                #     pipe,
                #     param_grid,
                #     verbose=1,
                #     n_jobs=5,
                #     n_points=10,
                #     n_iter=50,
                # )
                
                # _ = opt.fit(X_train, Y_train, xgb__eval_set=[(X_val, Y_val)])
                # print(NAME)
                # print(opt.score(X_val, Y_val))
                # print(opt.score(X_test, Y_test))
                # print(opt.best_params_)
                # with open(f'./{NAME}/model.pkl', 'wb') as f:
                #     pickle.dump(opt, f)
                _ = pipe.fit(X_train, Y_train, xgb__eval_set=[(X_val, Y_val)])
                print(NAME)
                print(pipe.score(X_val, Y_val))
                print(pipe.score(X_test, Y_test))
                # print(pipe.best_params_)
                with open(f'./{NAME}/model.pkl', 'wb') as f:
                    pickle.dump(pipe, f)
            
            y_test_preds = pipe.predict(X_test)
            print(y_test_preds.shape, X_test.shape)
            
            
            xgb.plot_importance(pipe['xgb'])
            plt.savefig(f'./{NAME}/feature_importance.png')
            plt.close()
            xgb.plot_tree(pipe['xgb'])
            plt.savefig(f'./{NAME}/tree.png')
            plt.close()
            
            

def get_cpblulc_file_paths(root_dir):
    
    # directory structure:
    # | dataset
    # | | resolution (224, 299, 600) <- root_dir (you are here)
    # | | | sample_id <- parent_patch_id
    # | | | | input
    # | | | | | 00000.tif <- child_patch_id
    # | | | | | 00001.tif
    # | | | | | 00002.tif
    # | | | | target
    # | | | | | 00000.tif <-child_patch_id
    # | | | | | 00001.tif
    # | | | | | 00002.tif
    filepaths = []
    for parent_patch_id in os.listdir(root_dir):
        for child_patch_id in os.listdir(os.path.join(root_dir, parent_patch_id, 'input')):
            if not child_patch_id.endswith('.tif'): continue
            filepaths.append(
                (
                    os.path.join(root_dir, parent_patch_id, 'input', child_patch_id),
                    os.path.join(root_dir, parent_patch_id, 'target', child_patch_id)
                )
            )
    return filepaths

def get_nyc_file_paths(root_dir):
    
    # directory structure:
    # | dataset
    # | | resolution (224, 299, 600) <- root_dir (you are here)
    # | | |  input
    # | | | | 00000.tif <- child_patch_id
    # | | | | 00001.tif
    # | | | | 00002.tif
    # | | | target
    # | | | | 00000.tif <-child_patch_id
    # | | | | 00001.tif
    # | | | | 00002.tif
    filepaths = []
    for child_patch_id in os.listdir(os.path.join(root_dir, 'sampled', 'input')):
        if not child_patch_id.endswith('.tif'): continue
        filepaths.append(
            (
                os.path.join(root_dir, 'sampled', 'input', child_patch_id),
                os.path.join(root_dir, 'sampled', 'target', child_patch_id)
            )
        )
    return filepaths

def load_dataset(
    filepaths: list[tuple[str, str]], 
    bands: list[int]=[4, 1, 2], 
    n_samples: int=0, 
    nodata_value: int=15
) -> np.ndarray:
    
    if n_samples == 0: n_samples = len(filepaths) # if n_samples == 0 load all samples
    random.shuffle(filepaths) # shuffle filepaths
    
    i = 0
    X, y = np.empty((0, 3)), np.empty(0)
    for filepath in filepaths:
        
        if not os.path.exists(filepath[0]): raise FileNotFoundError(f'Input file {filepath[0]} does not exist')
        if not os.path.exists(filepath[1]): raise FileNotFoundError(f'Target file {filepath[1]} does not exist')
        
        y_sample = rio.open(filepath[1]).read(1)
        if np.any(y_sample == nodata_value): continue
        
        X_sample = rio.open(filepath[0]).read(bands)
        
        X_sample = X_sample.transpose(1, 2, 0).reshape(-1,3)
        y_sample = y_sample.reshape(-1)
        
        X = np.append(X, X_sample, axis=0)
        y = np.append(y, y_sample, axis=0)
        
        i+= 1
        if i == n_samples: break
        
    return X, y

if __name__ == '__main__':
    main()
