import numpy as np
import torch

def convert_typing_to_native(data: dict) -> dict:
    """
    Convert a dictionary containing high-level types into an identical dictionary containing only native types
    
    Args:
        data : The dictionnary to process

    Returns:
        The native types dictionary
    """
    if isinstance(data, dict):
        return {k: convert_typing_to_native(v) for k, v in data.items()}

    elif isinstance(data, tuple):
        return convert_typing_to_native(list(data))

    elif isinstance(data, torch.Tensor):
        return convert_typing_to_native(data.item()) if data.numel()==1 else convert_typing_to_native(data.cpu().numpy().tolist())

    elif isinstance(data, np.ndarray):
        return convert_typing_to_native(data.item()) if len(data)==1 else convert_typing_to_native(data.tolist())

    elif isinstance(data, list):
        return [convert_typing_to_native(elem) for elem in data]

    elif isinstance(data, (np.int64, np.int32)):
        return int(data)

    elif isinstance(data, (np.float64, np.float32)):
        return float(data)

    else:
        return data