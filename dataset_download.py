from radiant_mlhub import Dataset
import os


filter = dict(
    ref_landcovernet_eu_v1_source_sentinel_2=['B02', 'B03', 'B04', 'B08', 'CLD'], # R, G, B, NIR
    ref_landcovernet_eu_v1_labels=['labels'],
)
dataset = Dataset.fetch('ref_landcovernet_eu_v1')
if not os.path.exists('/scratch/landcovernet_eu'): os.mkdir('/scratch/landcovernet_eu')
dataset.download(output_dir='/scratch/landcovernet_eu', collection_filter=filter)