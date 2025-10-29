import torch
from torch.utils.data import Dataset

def get_max_label(dataset):
    """Traverse the dataset to get the maximum label value"""
    max_label = 0
    for _, label in dataset:
        if isinstance(label, torch.Tensor):
            label = label.item()
        max_label = max(max_label, label)
    return max_label

def get_bit_length(max_label):
    """Calculate required number of binary bits for a given maximum label value"""
    return max_label.bit_length() if max_label > 0 else 1

class BitwiseDataset(Dataset):
    def __init__(self, original_dataset, bit_position, max_label=None):
        self.original_dataset = original_dataset
        self.bit_position = bit_position
        
        # Automatically calculate maximum label value (if not provided)
        if max_label is None:
            self.max_label = get_max_label(original_dataset)
        else:
            self.max_label = max_label
        
        # Calculate required number of binary bits
        self.bit_length = get_bit_length(self.max_label)
        
        # Validate bit_position validity
        assert 0 <= bit_position < self.bit_length, f"bit_position must be 0~{self.bit_length-1}"

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        data, label = self.original_dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        
        # Convert label to binary string (pad with leading zeros)
        binary_str = format(label, f'0{self.bit_length}b')
        
        # Extract value at bit_position
        bit_index = self.bit_length - 1 - self.bit_position  # 0 corresponds to least significant bit
        return data, 2*int(binary_str[bit_index])-1