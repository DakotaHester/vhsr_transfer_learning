import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import pickle
import pandas as pd

def main():
    
    data_path = 'XGBOOST_nyclc_n_samples_128_resolution_224/'
    
    xgboost_model = pickle.load(open(data_path + 'model.pkl', 'rb'))
    print(xgboost_model)
    
    X_test = pickle.load(open(data_path + 'X_test.pkl', 'rb'))
    Y_test = pickle.load(open(data_path + 'Y_test.pkl', 'rb'))
    
    print(f'X_test - type: {type(X_test)}, shape: {X_test.shape}, range: ({X_test.min()}, {X_test.max()}), dtype: {X_test.dtype}')
    print(f'Y_test - type: {type(Y_test)}, shape: {Y_test.shape}, range: ({Y_test.min()}, {Y_test.max()}), dtype: {Y_test.dtype}')
    
    Y_pred = xgboost_model.predict(X_test)
    print(f'Y_pred - type: {type(Y_pred)}, shape: {Y_pred.shape}, range: ({Y_pred.min()}, {Y_pred.max()}), dtype: {Y_pred.dtype}')
    
    metrics_dict = {
        'acc': accuracy_score(Y_test, Y_pred),
        'f1': f1_score(Y_test, Y_pred, average='micro'),
        'jaccard': jaccard_score(Y_test, Y_pred, average='micro')
    }
    
    print(metrics_dict)
    pd.DataFrame(metrics_dict, index=[0]).to_csv(data_path + 'metrics.csv', index=False)
    
if __name__ == '__main__':
    main()