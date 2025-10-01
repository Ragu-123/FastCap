 
#!/bin/bash

# This script triggers the download of the 'yerevann/coco-karpathy' dataset from Hugging Face.
# The `datasets` library will automatically handle caching, so you only need to run this once.
# The CocoKarpathyDataset class in datasets.py will then use this cached data.

echo "--- Starting Hugging Face dataset download and caching ---"
echo "Dataset: yerevann/coco-karpathy"
echo "This may take a while depending on your internet connection..."

# Use python to leverage the `datasets` library for robust downloading and caching.
# We download both the 'train' and 'test' splits as they are needed for
# the train/validation/test splits defined in the Karpathy annotations.
python -c "
from datasets import load_dataset
print('\nDownloading the train/validation split...')
load_dataset('yerevann/coco-karpathy', split='train')
print('\nDownloading the test split...')
load_dataset('yerevann/coco-karpathy', split='test')
print('\nAll necessary splits have been downloaded and cached.')
"

echo "--- Download and caching process complete. ---"
