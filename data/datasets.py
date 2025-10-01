# enhanced-fastcap/src/fastcap/data/datasets.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import random

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: `datasets` library not found. This script will not work.")
    print("Please install it: `pip install datasets`")
    load_dataset = None

try:
    from transformers import BertTokenizer
except ImportError:
    print("Warning: `transformers` library not found. This script will not work.")
    print("Please install it: `pip install transformers`")
    BertTokenizer = None


class CocoKarpathyDataset(Dataset):
    """
    A PyTorch Dataset wrapper for the yerevann/coco-karpathy dataset from Hugging Face.
    It handles loading from the HF cache, subsetting, and preprocessing of images
    and captions.
    """
    def __init__(self, split='train', tokenizer=None, pad_token_id=0, max_len=50, subset_percentage=0.25):
        """
        Args:
            split (str): The dataset split to use ('train', 'validation', or 'test').
            tokenizer (object, optional): A pre-initialized tokenizer. If None, a default
                                          BERT tokenizer will be created.
            pad_token_id (int): The ID for the padding token from the tokenizer.
            max_len (int): The maximum length for caption padding.
            subset_percentage (float): The fraction of the dataset to use.
        """
        if load_dataset is None or BertTokenizer is None:
            raise ImportError("Required libraries `datasets` or `transformers` are not installed.")

        self.split = split
        self.max_len = max_len
        self.tokenizer = tokenizer if tokenizer is not None else self.build_tokenizer()
        self.pad_token_id = pad_token_id

        print(f"Loading '{split}' split of yerevann/coco-karpathy from cache...")
        hf_split = 'test' if split == 'test' else 'train'
        full_dataset = load_dataset("yerevann/coco-karpathy", split=hf_split)

        # Filter the dataset based on the Karpathy split definition
        if split == 'validation':
            self.hf_dataset = full_dataset.filter(lambda ex: ex['split'] == 'val')
        elif split == 'train':
            self.hf_dataset = full_dataset.filter(lambda ex: ex['split'] == 'train')
        else: # test
            self.hf_dataset = full_dataset

        # Handle subsetting for low-resource training
        original_size = len(self.hf_dataset)
        if subset_percentage < 1.0:
            print(f"Subsetting data to {subset_percentage*100:.0f}% of its original size.")
            subset_size = int(original_size * subset_percentage)
            self.hf_dataset = self.hf_dataset.shuffle(seed=42).select(range(subset_size))
            print(f"Original size: {original_size}, New size: {len(self.hf_dataset)}")

        # Standard image transformations for vision models
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        record = self.hf_dataset[idx]
        
        # Robust image loading: try direct 'image' field, fallback to 'url'
        try:
            image = record['image'].convert("RGB")
        except (AttributeError, KeyError):
            try:
                response = requests.get(record['url'])
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not load image for record {idx}. Error: {e}")
                return self.transform(Image.new('RGB', (224, 224))), torch.zeros(self.max_len, dtype=torch.long), torch.ones(self.max_len, dtype=torch.bool)
            
        image_tensor = self.transform(image)
        
        captions = record['sentences']
        
        # For training, randomly select one of the 5 captions (data augmentation)
        # For validation/testing, use the first caption for consistent evaluation
        caption_text = random.choice(captions) if self.split == 'train' else captions[0]
        
        # Tokenize the caption using the provided tokenizer
        tokenized = self.tokenizer.encode_plus(
            caption_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = tokenized['input_ids'].squeeze(0)
        
        # CORRECTED LOGIC: Padding mask is now explicitly created using the pad_token_id.
        # It should be True for padded tokens, False otherwise.
        padding_mask = (input_ids == self.pad_token_id)
        
        return image_tensor, input_ids, padding_mask

    @staticmethod
    def build_tokenizer(model_name='bert-base-uncased'):
        """Builds and returns a standard tokenizer."""
        print(f"Building tokenizer: {model_name}")
        return BertTokenizer.from_pretrained(model_name)

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to correctly stack batch tensors."""
        images, captions, masks = zip(*batch)
        return torch.stack(images, 0), torch.stack(captions, 0), torch.stack(masks, 0)

# Example usage:
if __name__ == '__main__':
    print("--- Testing CocoKarpathyDataset ---")
    
    # 1. Instantiate a tokenizer first to get its pad_token_id
    tokenizer = CocoKarpathyDataset.build_tokenizer()
    pad_id = tokenizer.pad_token_id
    
    # 2. Instantiate the training dataset, passing the tokenizer and pad_id
    train_dataset = CocoKarpathyDataset(
        split='train', 
        tokenizer=tokenizer,
        pad_token_id=pad_id,
        subset_percentage=0.01 # Use 1% for a quick test
    )
    
    # 3. Inspect a single sample
    print("\n--- Inspecting a single sample ---")
    img, cap, mask = train_dataset[0]
    print(f"Image tensor shape: {img.shape}")
    print(f"Caption tensor shape: {cap.shape}")
    print(f"Padding mask shape: {mask.shape}")
    assert img.shape == (3, 224, 224)
    assert cap.shape == (train_dataset.max_len,)
    
    # 4. Test with a DataLoader
    print("\n--- Testing with DataLoader ---")
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        collate_fn=CocoKarpathyDataset.collate_fn,
        shuffle=True
    )
    
    img_batch, cap_batch, mask_batch = next(iter(train_loader))
    print(f"Batched image tensor shape: {img_batch.shape}")
    print(f"Batched caption tensor shape: {cap_batch.shape}")
    print(f"Batched padding mask shape: {mask_batch.shape}")
    assert img_batch.shape == (4, 3, 224, 224)
    
    print("\nDataset and DataLoader are working correctly with the updated logic.")

