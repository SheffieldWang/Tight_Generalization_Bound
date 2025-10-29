import torch
import jax.numpy as jnp
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

def is_dataloader_shuffled(dataloader: DataLoader) -> bool:
    """
    Check if DataLoader has shuffle=True enabled
    
    Args:
        dataloader (DataLoader): PyTorch DataLoader object to check
    
    Returns:
        bool: True if data will be shuffled, False otherwise
    
    Raises:
        AttributeError: When DataLoader does not have sampler attribute
    """
    if not hasattr(dataloader, 'sampler'):
        raise AttributeError("DataLoader missing 'sampler' attribute")

    sampler = dataloader.sampler

    # Handle nested BatchSampler case
    if isinstance(sampler, BatchSampler):
        sampler = sampler.sampler  # Extract underlying sampler

    # Check sampler type
    if isinstance(sampler, RandomSampler):
        return True
    elif isinstance(sampler, SequentialSampler):
        return False
    elif isinstance(sampler, DistributedSampler):
        return sampler.shuffle  # Check internal shuffle attribute for distributed case
    else:
        print("⚠️ Warning: Custom sampler detected, cannot directly determine shuffle status")
        return False
    
    
def numpy_collate(batch):
    """Convert batch data to jax.array format"""
    if isinstance(batch[0], tuple):
        return tuple(map(numpy_collate, zip(*batch)))
    return jnp.array(batch)