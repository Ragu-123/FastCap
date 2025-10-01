# enhanced-fastcap/src/fastcap/data/build_vocab.py

import argparse
import pickle
from collections import Counter
import re

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: `datasets` library not found. This script will not work.")
    print("Please install it: `pip install datasets`")
    load_dataset = None

from tqdm import tqdm

class Vocabulary:
    """A simple vocabulary wrapper that maps words to indices."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def tokenize(text):
    """A simple tokenizer that splits on non-alphanumeric characters."""
    text = text.lower()
    # Keep alphanumeric characters and spaces
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text.split()

def build_vocabulary(threshold=5):
    """
    Builds a vocabulary from the training captions of the Karpathy dataset.

    Args:
        threshold (int): The minimum frequency for a word to be included in the vocabulary.
    
    Returns:
        Vocabulary: The constructed vocabulary object.
    """
    if load_dataset is None:
        raise ImportError("`datasets` library is required to run this script.")

    print("Loading Karpathy training split to build vocabulary...")
    # We only need the 'train' part of the 'train' split from HF
    full_train_dataset = load_dataset("yerevann/coco-karpathy", split='train')
    train_dataset = full_train_dataset.filter(lambda ex: ex['split'] == 'train')
    
    counter = Counter()
    print("Tokenizing captions and counting word frequencies...")
    for record in tqdm(train_dataset, desc="Processing captions"):
        for sentence in record['sentences']:
            tokens = tokenize(sentence)
            counter.update(tokens)

    # Filter words with frequency below the threshold
    words = [word for word, count in counter.items() if count >= threshold]

    # Create the vocabulary object and add special tokens
    vocab = Vocabulary()
    vocab.add_word('<pad>')  # Padding token
    vocab.add_word('<start>') # Start of sentence token
    vocab.add_word('<end>')   # End of sentence token
    vocab.add_word('<unk>')   # Unknown word token

    # Add the filtered words to the vocabulary
    print(f"Adding {len(words)} words to the vocabulary (frequency >= {threshold})...")
    for word in words:
        vocab.add_word(word)
        
    return vocab

def main(args):
    vocab = build_vocabulary(threshold=args.threshold)
    
    # Save the vocabulary to a file
    with open(args.output_path, 'wb') as f:
        pickle.dump(vocab, f)
        
    print(f"\nVocabulary built successfully.")
    print(f"Total vocabulary size: {len(vocab)}")
    print(f"Vocabulary saved to: {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a vocabulary for the EnhancedFastCap model.")
    parser.add_argument('--threshold', type=int, default=5,
                        help="Minimum word frequency to be included in the vocabulary.")
    parser.add_argument('--output-path', type=str, default='./vocab.pkl',
                        help="Path to save the generated vocabulary file.")
    args = parser.parse_args()
    
    main(args)
 
