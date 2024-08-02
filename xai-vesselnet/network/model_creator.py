import torch
from torch.nn import Module

from models import instanciate_model


def init_inference_model(
    model_name: str,
    weights_path: str,
    in_channels: int,
    out_channels: int,
    device: torch.device,
) -> Module:
    """
    Load a trained model and switch to evaluation mode.

    Args:
        model_name      : The name of the model
        weights_path    : The patch to the trained weights
        in_channels     : The number of input channels
        device          : The device to store the model

    Returns:
        model (torch.nn.Module) : The trained model in evaluation mode.
    """
    # Model
    model = instanciate_model.instanciate_model(
        model_name, in_channels=in_channels, out_channels=out_channels
    ).to(device)
    model.load_state_dict(torch.load(weights_path)["model_state_dict"])
    model.eval()

    return model
