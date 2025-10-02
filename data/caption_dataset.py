# enhanced-fastcap/src/fastcap/data/datasets.py (Corrected and Final Version)

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
    This version is updated to return the image_id for correct evaluation.
    """
    def __init__(self, split='train', tokenizer=None, pad_token_id=0, max_len=50, subset_percentage=1.0):
        if load_dataset is None or BertTokenizer is None:
            raise ImportError("Required libraries `datasets` or `transformers` are not installed.")

        self.split = split
        self.max_len = max_len
        self.tokenizer = tokenizer if tokenizer is not None else self.build_tokenizer()
        self.pad_token_id = pad_token_id

        print(f"Loading '{split}' split of yerevann/coco-karpathy from cache...")
        # The 'train' split on Hugging Face contains both Karpathy 'train' and 'val'
        # The 'test' split on Hugging Face corresponds to Karpathy 'test'
        hf_split_name = 'test' if split == 'test' else 'train'
        full_dataset = load_dataset("yerevann/coco-karpathy", split=hf_split_name)

        # The 'split' column in the dataset tells us which Karpathy split it belongs to.
        if split == 'validation':
            self.hf_dataset = full_dataset.filter(lambda ex: ex['split'] == 'val')
        elif split == 'train':
            self.hf_dataset = full_dataset.filter(lambda ex: ex['split'] == 'train')
        else: # test
            self.hf_dataset = full_dataset

        original_size = len(self.hf_dataset)
        if subset_percentage < 1.0 and self.split == 'train': # Usually only subset training data
            print(f"Subsetting data to {subset_percentage*100:.0f}% of its original size.")
            subset_size = int(original_size * subset_percentage)
            self.hf_dataset = self.hf_dataset.shuffle(seed=42).select(range(subset_size))
            print(f"Original size: {original_size}, New size: {len(self.hf_dataset)}")

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
        # CRITICAL FIX: Get the image_id for evaluation purposes.
        image_id = record['imgid']
        
        # Robust image loading from either PIL object or URL
        try:
            image = record['image'].convert("RGB")
        except (AttributeError, KeyError):
            try:
                response = requests.get(record['url'])
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                # Return a placeholder for a failed image load
                print(f"Warning: Could not load image for record {idx} (ID: {image_id}). Error: {e}")
                # Return a dummy image and a special image_id (-1) to be skipped in collate_fn if needed
                return self.transform(Image.new('RGB', (224, 224))), torch.zeros(self.max_len, dtype=torch.long), torch.ones(self.max_len, dtype=torch.bool), -1

        image_tensor = self.transform(image)
        captions = record['sentences']
        
        # During training, randomly select one of the 5 captions as a form of data augmentation.
        # During validation, consistently use the first one.
        caption_text = random.choice(captions) if self.split == 'train' else captions[0]
        
        tokenized = self.tokenizer.encode_plus(
            caption_text, 
            max_length=self.max_len, 
            padding='max_length',
            truncation=True, 
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].squeeze(0)
        padding_mask = (input_ids == self.pad_token_id)
        
        # CORRECTED: Return image_id along with other data.
        return image_tensor, input_ids, padding_mask, image_id

    @staticmethod
    def build_tokenizer(model_name='bert-base-uncased'):
        """Helper static method to build a standard tokenizer."""
        print(f"Building tokenizer: {model_name}")
        return BertTokenizer.from_pretrained(model_name)

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to correctly stack batch tensors and handle image_ids.
        """
        # Filter out any samples that failed to load (where image_id is -1)
        batch = [b for b in batch if b[3] != -1]
        if not batch:
            return None, None, None, None

        # Unzip the batch into separate lists
        images, captions, masks, image_ids = zip(*batch)
        
        # Stack tensors and collect image_ids into a list
        return torch.stack(images, 0), torch.stack(captions, 0), torch.stack(masks, 0), list(image_ids)

# Example usage to verify the changes
if __name__ == '__main__':
    print("--- Testing CocoKarpathyDataset ---")
    
    tokenizer = CocoKarpathyDataset.build_tokenizer()
    pad_id = tokenizer.pad_token_id
    
    # Use a small subset of the validation set for a quick test
    val_dataset = CocoKarpathyDataset(
        split='validation', 
        tokenizer=tokenizer,
        pad_token_id=pad_id,
        subset_percentage=0.05 # Use 5% for a quick test
    )
    
    print(f"\nDataset size: {len(val_dataset)}")
    
    # Inspect a single sample to check the new output format
    print("\n--- Inspecting a single sample ---")
    img, cap, mask, img_id = val_dataset[0]
    print(f"Image tensor shape: {img.shape}")
    print(f"Caption tensor shape: {cap.shape}")
    print(f"Padding mask shape: {mask.shape}")
    print(f"Image ID: {img_id} (Type: {type(img_id)})")
    assert isinstance(img_id, (int, str))
    
    # Test with a DataLoader using the updated collate_fn
    print("\n--- Testing with DataLoader ---")
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=CocoKarpathyDataset.collate_fn,
        shuffle=False
    )
    
    img_batch, cap_batch, mask_batch, id_batch = next(iter(val_loader))
    print(f"Batched image tensor shape: {img_batch.shape}")
    print(f"Batched caption tensor shape: {cap_batch.shape}")
    print(f"Batched padding mask shape: {mask_batch.shape}")
    print(f"Image ID batch: {id_batch} (Length: {len(id_batch)})")
    assert len(id_batch) == 4
    
    print("\nDataset and DataLoader are working correctly with the updated logic.")
