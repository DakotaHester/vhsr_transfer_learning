from radiant_mlhub import Dataset
import os
from multiprocessing import Pool

def main():
    dataset_names = [
        # 'na',
        # 'eu',
        # 'af',
        # 'as',
        # 'sa',
        'au',
    ]

    with Pool(len(dataset_names)) as p:
        p.map(get_dataset, dataset_names)

def get_dataset(dataset_name):
    output_dir = f'/scratch/landcovernet_{dataset_name}'
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    collection_filter = {
        f'ref_landcovernet_{dataset_name}_v1_source_sentinel_2': ['B02', 'B03', 'B04', 'B08', 'CLD'], # R, G, B, NIR
        f'ref_landcovernet_{dataset_name}_v1_labels': ['labels'],
    }

    dataset = Dataset.fetch(f'ref_landcovernet_{dataset_name}_v1')
    dataset.download(output_dir=output_dir, collection_filter=collection_filter)

if __name__ == "__main__":
    main()