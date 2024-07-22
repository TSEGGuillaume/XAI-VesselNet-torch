import argparse
import logging
import os

import torch
from torch.nn import Module

from monai.data.meta_tensor import MetaTensor
from monai.inferers import Inferer, SimpleInferer

from monai.data import DataLoader, ArrayDataset
from monai.transforms import (
    Transform,
    Compose,
    Activations,
    AsDiscrete,
    RemoveSmallObjects,
    SpatialCrop,
    SpatialPad,
)
from monai.transforms import SaveImage

from monai.inferers import SlidingWindowInferer
from monai.data.utils import decollate_batch, first

from models import instanciate_model
from datasets.instanciate_dataset import instanciate_image_dataset
from utils.prebuilt_logs import log_hardware
from utils.load_hyperparameters import load_hyperparameters

from datasets.ImageDataset import ImageDatasetd
from network.model_creator import init_inference_model

logger = logging.getLogger("app")


_CONST_BATCH_SIZE = 1  # `batch_size`==1 as data does not have equal spatial dims. See `sw_batch_size` variable instead


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        metavar=("MODEL"),
        choices=instanciate_model._all_models,
        default=instanciate_model._all_models[0],
        help="Model name",
    )
    parser.add_argument(
        "weights",
        type=str,
        metavar=("WEIGHTS_PATH"),
        help="Path to the model's weights",
    )
    parser.add_argument(
        "data",
        type=str,
        metavar=("DATA_PATH"),
        help="Path to the data to infer (*.nii.gz, *.csv)",
    )
    parser.add_argument(
        "--hyperparameters",
        "-p",
        type=str,
        metavar=("HYPERPARAM_PATH"),
        default="./resources/default_hyperparameters.json",
        help="Path to the hyperparameters file (*.json)",
    )

    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args


def infer_patch(
    model: Module,
    data: MetaTensor,
    device: torch.device,
    patch_pos: list,
    input_size: tuple,
    postprocessing: Transform = None,
):
    """
    Create a patch from a provided spatial position and infer this patch.
    The position of the patch must be a tuple list in the following format [ (start, ), (end, ) ].
    The input size (patch size) is used to pad the patch if the end position of the roi exceed the image size.
    To save the result of the inferences, compose the post-processing with a `SaveImage` object.

    Args
        model           : The trained model
        data            : The image data
        device          : The device to store the model
        patch_pos       : The [ (start, ), (end, ) ] coordinate of the patch
        input_size      : The input size (patch size)
        postprocessing  : the post transforms to apply

    Returns
        predictions (list[MetaTensor]) : The list of infered patch.
    """

    if isinstance(data, MetaTensor) == False:
        raise TypeError(f"Expected `MetaTensor`, got {type(data)}")

    # Pre-processing
    start_roi, end_roi = patch_pos

    pre_T_composer = Compose(
        [
            # Crop the volume to patch at roi position, and pad the patch if patch size is inferior to input size.
            SpatialCrop(roi_start=start_roi, roi_end=end_roi),
            SpatialPad(spatial_size=input_size),
        ]
    )

    # Manage the data
    logger.debug("MetaTensor provided. Collate the data through a DataLoader...")
    _ds = ArrayDataset(img=[data], img_transform=pre_T_composer)
    data = DataLoader(_ds, batch_size=_CONST_BATCH_SIZE, num_workers=0)

    inferer = SimpleInferer()

    predictions = infer(
        model=model,
        data=data,
        device=device,
        inferer=inferer,
        postprocessing=postprocessing,
    )

    return predictions


def infer(
    model: Module,
    data: DataLoader,
    device: torch.device,
    inferer: Inferer = None,
    postprocessing: Transform = None,
) -> list[MetaTensor]:
    """
    Infer the data included in the data loader.
    The `DataLoader` should be constructed from a `ImageDatasetd` or `ArrayDataset` dataset type. Otherwise, an exception is thrown.
    If a patch-based model is used, infer the entire volume by supplying a `SlindingWindowInferer`, configured accordingly, as inferer.
    To infer a single data / a single patch, provide a `SimpleInferer` as inferer. If inferer is None, `SimpleInferer` is used as default.
    To save the result of the inferences, compose the post-processing with a `SaveImage` object.

    Args:
        model           : The trained model
        data            : The data loader
        device          : The device to store the model
        inferer         : The inferer to use. `None` as default
        postprocessing  : the post transforms to apply

    Returns:
        predictions (list[MetaTensor]) : The list of infered data. `predictions` contains `ceil(count_data/batch_size)` MetaTensors (B,C,H,W,[D]) where `B=batch_size` except the last where `B=count_data%batch_size`
    """

    if isinstance(data, DataLoader) == False:
        raise TypeError(f"Expected data type `DataLoader`, got {type(data)}")

    # Default behaviour
    if inferer is None:
        inferer = SimpleInferer()

    if postprocessing is None:
        postprocessing = Compose([])

    # Prediction
    predictions = []

    if isinstance(data.dataset, ImageDatasetd):

        for item in data:
            x = item["img"].to(device)

            logger.debug(f"Infer a {x.shape} tensor.")

            y = inferer(inputs=x, network=model)
            prediction = torch.stack([postprocessing(i) for i in decollate_batch(y)])

            predictions.append(prediction)

    elif isinstance(data.dataset, ArrayDataset):

        for item in data:

            if isinstance(item, MetaTensor) == False:
                # If this exception is thrown, check the dataset construction. Unexpected segmentation data may have been provided
                raise TypeError(f"Expected data type `MetaTensor`, got {type(item)}")

            x = item.to(device)

            logger.debug(f"Infer a {x.shape} tensor.")

            y = inferer(inputs=x, network=model)
            prediction = torch.stack([postprocessing(i) for i in decollate_batch(y)])

            predictions.append(prediction)
    else:
        raise TypeError(
            f"The `DataLoader` should be constructed from a `ImageDatasetd` or `ArrayDataset`."
        )

    return predictions


def main():
    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    # Define variables from user args
    model_name = args.model
    weigths_path = args.weights
    dataset_path = args.data

    # Load hyperparameters for training
    hyperparameters = load_hyperparameters(args.hyperparameters)

    in_channels = hyperparameters["in_channels"]
    out_channels = hyperparameters["out_channels"]

    # Load the trained model
    model = init_inference_model(
        model_name=model_name,
        weigths_path=weigths_path,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
    )

    # Load the data to infer
    infer_ds = instanciate_image_dataset(dataset_path, image_only=False)
    infer_loader = DataLoader(infer_ds, batch_size=_CONST_BATCH_SIZE, num_workers=0)

    # Verify that the provided `in_channels` in the setting file matches the actual data channels
    # Assume the same input channels through the whole dataset
    assert (
        in_channels == first(infer_loader)["img"].shape[1]
    ), "Provided `in_channels` in hyperparamaeters file does not match the actual image channel"

    # Prepare the inferer
    sw_batch_size = hyperparameters["batch_size"]
    sw_shape = hyperparameters["patch_size"]
    sw_overlap = hyperparameters["patch_overlap"]

    inferer = SlidingWindowInferer(
        sw_shape, sw_batch_size=sw_batch_size, overlap=sw_overlap
    )

    # To save the image. Deported from the inference function to allow customization
    save_seg = SaveImage(
        output_dir=os.path.join(cfg.result_dir, "inferences"),
        output_ext=".nii.gz",
        output_postfix=f"seg_{model_id}",
        resample=False,
        separate_folder=False,
    )

    transforms = Compose(
        [
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
            RemoveSmallObjects(),
            save_seg,
        ]
    )

    predictions = infer(
        model=model,
        data=infer_loader,
        inferer=inferer,
        device=device,
        postprocessing=transforms,
    )

    logger.info(f"End prediction for {len(predictions)} data.")
    logger.info(f"Save directory : {save_seg.folder_layout.output_dir}")


if __name__ == "__main__":
    from utils.configuration import Configuration

    cfg = Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    # Logger
    model_id, ext = os.path.splitext(os.path.basename(args.weights))

    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": "{}/infer_{}_{}.log".format(cfg.log_dir, args.model, model_id),
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    logger.debug(args)

    main()
