"""
TODO:
  * Add metrics (clDice, TP, FP, FN, TN)
  * Compute the average metrics
"""

import argparse
import logging
import os

import torch

from monai.data.meta_tensor import MetaTensor
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    SaveImage,
    LoadImage,
)
from monai.metrics import (
    DiceMetric,
    SurfaceDistanceMetric,
)

from monai.data.utils import decollate_batch, first
from monai.networks.utils import one_hot as OneHotEncoding


from models import instanciate_model
from network.model_creator import init_inference_model
from datasets.instanciate_dataset import instanciate_image_dataset
from utils.load_hyperparameters import load_hyperparameters
from utils.dataset_reader import parse_csv
from utils.prebuilt_logs import log_hardware
from metrics.cldice import clDiceMetric

from infer import infer


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
        help="Path to the data to infer (*.csv)",
    )
    parser.add_argument(
        "--hyperparameters",
        "-p",
        type=str,
        metavar=("HYPERPARAMETERS_JSON_PATH"),
        default=os.path.join(
            cfg.workspace, "resources", "default_hyperparameters.json"
        ),
        help="Path to the hyperparameters JSON",
    )
    parser.add_argument(
        "--mask",
        "-m",
        type=str,
        metavar=("MASK_CSV_PATH"),
        default=None,
        help="Path to the masks CSV (*.csv)",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args


def parse_mask_file(csv_path: str) -> dict:
    """
    Read a CSV file containing various masks to be applied during evaluation. The header (first line) of the CSV file should define the mask names.

    Args:
        csv_path : The path to the CSV file

    Returns:
        model (dict) : The dictionnary of mask filenames list, indexed by the mask type (first line of the CSV)
    """
    masks_filenames = parse_csv(csv_path)

    # Assume the first line contains mask identifier (csv header)
    header = masks_filenames[0]

    masks_dict = {}

    for line in masks_filenames[1:]:
        for mask_type, mask in zip(header, line):
            masks_dict.setdefault(mask_type, []).append(mask)

    return masks_dict


def evaluate(ys_pred: list[MetaTensor], ys_true: list[MetaTensor]) -> list[dict]:
    """
    Evaluate the predictions.
    Both predictions and ground-truths must be one-hot encoded.
    WARNING : Elements from ys_pred and ys_true must be on the same device !

    Available metrics : Dice, ASSD

    Args:
        ys_pred : The list MetaTensors (N,C,H,W,[D]) (predictions data)
        ys_true : The list MetaTensors (N,C,H,W,[D]) (ground-truths data)

    Returns:
        res_metrics (list[dict]) : The list of dictionnaries of computed metrics, indexed by the name (see available metrics).
    """
    assert (
        len(ys_pred) == len(ys_true)
    ), "Prediction and ground truth arrays must be the same length !"

    res_metrics = []

    # The metrics needs batch dimension (N, C, H, W, [D]) and one-hot-encoding (MONAI v1.1)
    mapping_name_metrics = {
        "Dice": DiceMetric(include_background=False, reduction="mean"),
        "clDice":clDiceMetric(include_background=False, reduction="mean"),
        "ASSD": SurfaceDistanceMetric(
            include_background=False,
            symmetric=True,
            distance_metric="euclidean",
            reduction="mean",
        ),
    }

    for y_pred, y_true in zip(ys_pred, ys_true):

        metrics_val = {}

        for metric_name, metric_fun in mapping_name_metrics.items():
            metric_fun(y_pred=y_pred, y=y_true)

            metric_val = metric_fun.aggregate().item()

            metrics_val[metric_name] = metric_val

            metric_fun.reset()

        res_metrics.append(metrics_val)

    return res_metrics


def print_evaluation_results(
    sample_fnames: list[str],
    res_metrics: dict,
    mask_type: str = None,
    mask_fnames: list[str] = None,
) -> None:
    """
    Print the computed metrics in a hierarchical view.

    Args:
        sample_fnames : The array of predictions
        res_metrics : The array of ground-truths
        mask_type : The array of ground-truths
        mask_fnames : The array of ground-truths

    Returns:
        res_metrics (list[dict]) : The dictionnary of computed metrics.
    """
    if (
        mask_type is not None
        and mask_fnames is None
        or mask_type is None
        and mask_fnames is not None
    ):
        raise ValueError("Both mask type and mask file list should be provided.")

    logger.info(f"Mask: {mask_type}")
    for sample_idx in range(len(sample_fnames)):
        logger.info(f"   Image: {sample_fnames[sample_idx]}")

        # if mask_fnames is not None:
        #     logger.debug(f"M: {mask_fnames[sample_idx]} :") # Only on debug

        for k, v in res_metrics[sample_idx].items():
            logger.info(f"      * {k}:\t{v}")


def main():
    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    # Define variables from user args
    model_name      = args.model
    weigths_path    = args.weights
    dataset_path    = args.data

    masks_path      = args.mask

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

    infer_ds = instanciate_image_dataset(dataset_path, image_only=True)
    infer_loader = DataLoader(
        infer_ds, batch_size=_CONST_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify that the provided `in_channels` in the setting file matches the actual data channels
    # Assume the same input channels through the whole dataset
    assert (
        in_channels == first(infer_loader)["img"].shape[1]
    ), "Provided `in_channels` in hyperparamaeters file does not match the actual image channel"

    # Prepare the inferer
    sw_batch_size   = hyperparameters["batch_size"]
    sw_shape        = hyperparameters["patch_size"]
    sw_overlap      = hyperparameters["patch_overlap"]

    inferer = SlidingWindowInferer(
        sw_shape, sw_batch_size=sw_batch_size, overlap=sw_overlap
    )

    # To save the image. Deported from the inference function to allow customization
    save_seg = SaveImage(
        output_dir=os.path.join(cfg.result_dir, "inferences"),
        output_ext=".nii",
        output_postfix=f"seg_{model_id}",
        resample=False,
        separate_folder=False,
    )

    # Post-transform
    transforms = Compose(
        [
            Activations(sigmoid=True) if out_channels == 1 else Activations(softmax=True),
            AsDiscrete(threshold=0.5),
            # RemoveSmallObjects(),
            save_seg,
        ]
    )

    ys_pred = infer(
        model=model,
        data=infer_loader,
        inferer=inferer,
        device=device,
        postprocessing=transforms,
    )

    # TODO: write a more pythonic data preparation
    # Prepare the prediction data
    # 1/ From a list of batch MetaTensors (B,C,H,W,[D]) to list of MetaTensors (1,C,H,W,[D]) -> just a security, see _CONST_BATCH_SIZE
    # 2/ Encode to One-Hot
    _mem_ys_pred = ys_pred
    ys_pred = []

    if out_channels > 1: 
        # We assume ys_pred elements are in one-hot format
        for batch_pred in _mem_ys_pred:
            ys_pred += [
                torch.unsqueeze(tensor, axis=0) for tensor in decollate_batch(batch_pred)
            ]
    elif out_channels == 1 :
        # We assume ys_pred elements are in single-channel class indices
        for batch_pred in _mem_ys_pred:
            ys_pred += [
                OneHotEncoding(torch.unsqueeze(tensor, axis=0), num_classes=2, dim=1)
                for tensor in decollate_batch(batch_pred)
            ]
    else:  
        raise RuntimeError("Tensors must have a channel dimension.")

    # Prepare the ground-truth data
    # 1/ Get the ground-truth in the right sample order given the loader
    # 2/ Add the batch dim
    # 3/ Encode to One-Hot
    # 4/ Transfert to device
    # 5/ Get the associated filenames
    ys_true = []
    sample_fnames = []

    for idx_sample in infer_loader.sampler:
        ys_true.append(
            OneHotEncoding(
                torch.unsqueeze(infer_ds[idx_sample]["seg"], axis=0),
                num_classes=2,
                dim=1,
            ).to(device)
        )
        sample_fnames.append(infer_ds.image_files[idx_sample])

    assert (
        len(ys_pred) == len(ys_true)
    ), f"Lenghts of predictions ({len(ys_pred)}) and ground-truth ({len(ys_true)}) array must match."

    for y_pred, y_true in zip(ys_pred, ys_true):
        assert (
            y_pred.shape == y_true.shape
        ), f"Shape of prediction {y_pred.shape} and ground-truth {y_true.shape} must match."

    # Evaluate the global data
    res_metrics = evaluate(ys_pred, ys_true)

    # Print the metrics
    print_evaluation_results(sample_fnames, res_metrics)

    # Evaluate masked data if provided
    if masks_path is not None:
        sorted_masklists = parse_mask_file(args.mask)

        logger.info(
            f"Masks detected : {[ mask_type for mask_type in sorted_masklists.keys() ]}"
        )

        # Masking task
        for mask_type, mask_list in sorted_masklists.items():
            assert (
                len(ys_pred) == len(mask_list)
            ), f"Lenghts of predictions ({len(ys_pred)}) and masks ({len(mask_list)}) array must match."

            # Work on copies ; TODO : dig inside the world of cloning, deepcopy, shallow copy, etc,...
            ys_pred_cpy = [y_pred.clone() for y_pred in ys_pred]
            ys_true_cpy = [y_true.clone() for y_true in ys_true]

            # Would be better in a function but type(ys_pred_cpy) == type(ys_true_cpy) =/= type(mask_list) and that's not pretty
            for idx_data, fname_mask in enumerate(mask_list):
                M = Compose(
                    [
                        LoadImage(image_only=True, ensure_channel_first=True),
                        AsDiscrete(threshold=0.5),
                    ]
                )(fname_mask).to(
                    device
                )  # Read mask file, binarize the mask and send it to the device

                # Points to watch out for: y_pred and y_true are (1,C,H,W,[D]) ; M is (1,H,W,[D]) --> broadcasting
                ys_pred_cpy[idx_data] = ys_pred_cpy[idx_data] * M
                ys_true_cpy[idx_data] = ys_true_cpy[idx_data] * M

            res_metrics = evaluate(ys_pred_cpy, ys_true_cpy)

            print_evaluation_results(
                sample_fnames, res_metrics, mask_type=mask_type, mask_fnames=mask_list
            )


if __name__ == "__main__":
    from utils.configuration import Configuration

    cfg = Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    # Logger
    model_id, ext = os.path.splitext(os.path.basename(args.weights))

    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": "{}/evaluation_{}_{}.log".format(
                cfg.log_dir, args.model, model_id
            ),
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    logger.debug(args)

    main()
