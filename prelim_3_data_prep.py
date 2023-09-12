import random
import os
import pickle
import data
import cupy as cp
import rasterio

PRE_TRAINING_SAMPLES = 10000 * 3
FINE_TUNING_SAMPLES = 1000 * 3
VALIDATION_SAMPLES = 250 * 3
TEST_SAMPLES = 250 * 3
TOTAL_SAMPLES = PRE_TRAINING_SAMPLES + FINE_TUNING_SAMPLES + VALIDATION_SAMPLES + TEST_SAMPLES

root_file_path = os.path.join('/', 'scratch', 'chesapeake_bay_lulc', 'sampled', '224')

def main() -> None:
    # set seed for reproducibility
    random.seed(1701)
    
    if os.path.exists(f'CPBLULC_224_filepaths.pkl'):
        print('loading data paths from pickle file')
        data_paths = pickle.load(open(f'CPBLULC_224_filepaths.pkl', 'rb'))
    else:
        print('getting data paths')
        data_paths = data.get_cpblulc_file_paths(root_file_path)    
        pickle.dump(data_paths, open(f'CPBLULC_224_filepaths.pkl', 'wb'))
      
    print(f'found {len(data_paths)} total samples')
    random.shuffle(data_paths)
    if len(data_paths) < TOTAL_SAMPLES:
        raise ValueError(f'only found {len(data_paths)} samples, need {TOTAL_SAMPLES}')
    
    # look at test label distribution
    # if os.path.exists(f'CPBLULC_224_test_labels.pkl'):
    if False:
        print('loading test labels from pickle file')
        test_labels = pickle.load(open(f'CPBLULC_224_test_labels.pkl', 'rb'))
    else:
        test_labels = cp.array(
            [rasterio.open(path[1]).read(1) for path in data_paths[:TOTAL_SAMPLES]]
        )
        pickle.dump(test_labels, open(f'CPBLULC_224_test_labels.pkl', 'wb'))
    
    print(test_labels.shape, test_labels.min(), test_labels.max(), test_labels.dtype)
    
    # look at test label distribution
    test_labels_hist, bin_edges = cp.histogram(test_labels, bins=6)
    test_labels_dist = test_labels_hist / test_labels_hist.sum()
    print(test_labels_hist, test_labels_dist, bin_edges, sep='\n')
    
    del test_labels
    
    pretrain_samples = data_paths[:PRE_TRAINING_SAMPLES]
    finetune_samples = data_paths[PRE_TRAINING_SAMPLES:PRE_TRAINING_SAMPLES+FINE_TUNING_SAMPLES]
    val_samples = data_paths[PRE_TRAINING_SAMPLES+FINE_TUNING_SAMPLES:PRE_TRAINING_SAMPLES+FINE_TUNING_SAMPLES+VALIDATION_SAMPLES]
    test_samples = data_paths[PRE_TRAINING_SAMPLES+FINE_TUNING_SAMPLES+VALIDATION_SAMPLES:TOTAL_SAMPLES]
    
    pickle.dump(pretrain_samples, open(f'CPBLULC_224_pretrain_samples.pkl', 'wb'))
    pickle.dump(finetune_samples, open(f'CPBLULC_224_finetune_samples.pkl', 'wb'))
    pickle.dump(val_samples, open(f'CPBLULC_224_val_samples.pkl', 'wb'))
    pickle.dump(test_samples, open(f'CPBLULC_224_test_samples.pkl', 'wb'))
    
    data_paths = data.remove_nodata_samples(data_paths, nodata_value=15, n_samples=TOTAL_SAMPLES, shuffle=False)
    # print(f'found {len(data_paths)} samples after removing nodata')

if __name__ == '__main__':
    main()