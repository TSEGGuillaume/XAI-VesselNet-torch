import argparse
import logging
import os

import torch
import monai
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.data.utils import decollate_batch, first
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    RemoveSmallObjects,
    SaveImage,
    LoadImage
)
from monai.metrics import DiceMetric

import models.instanciate_model

"""
TODO:
  * Add metrics (clDice, SurfaceDistanceMetric, TP, FP, FN, TN)
  * Compute the average metrics
  * The script is very csv-dependant (its structure), it is not very clear how to use mask. Find a better way (JSON ?)
  * General note: play with decollate batch to understand how it works
"""


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        metavar=("MODEL"),
        choices=models.instanciate_model._all_models,
        default=models.instanciate_model._all_models[0],
        help="Path to the training CSV",
    )
    parser.add_argument(
        "weights",
        type=str,
        metavar=("WEIGHTS_PATH"),
        help="Path to the model's weights",
    )
    parser.add_argument(
        "csv_eval",
        type=str,
        metavar=("EVAL_CSV_PATH"),
        help="Path to the evaluation CSV",
    )
    parser.add_argument(
        "--mask",
        "-m",
        type=str,
        metavar=("MASK_CSV_PATH"),
        default=None,
        help="Path to the masks CSV",
    )

    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args


def main():
    from utils.dataset_reader import parse_csv
    from utils.prebuilt_logs import log_hardware
    from models.instanciate_model import instanciate_model
    from datasets.instanciate_dataset import instanciate_image_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    infer_ds = instanciate_image_dataset(args.csv_eval, image_only=False)
    infer_loader = DataLoader(infer_ds, batch_size=1, num_workers=0) # Batch_size must be equal to 1 as input tensors does not have equal spatial dims. See sw_batch_size for parallelization.

    saver = SaveImage(
        output_dir=cfg.result_dir,
        output_ext=".nii",
        output_postfix=f"seg_{model_id}",
        resample=False,
        separate_folder=False,
    )

    # Post-processing
    post_transforms = Compose(
        [Activations(sigmoid=True), AsDiscrete(threshold=0.5), RemoveSmallObjects()]
    )

    # Model
    in_channels = first(infer_loader)["img"].shape[1]  # We suppose the same input channels through the whole dataset

    model = instanciate_model(args.model, in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(args.weights)["model_state_dict"])
    model.eval()

    # Inference parameters
    sw_shape = (64, 64, 64)  # TODO: Read from the --hyperparameters
    sw_batch_size = 64  # TODO: Move into a cfg file
    sw_overlap = 0.25  # TODO: Move into a cfg file

    inferer = SlidingWindowInferer(
        sw_shape,
        sw_batch_size=sw_batch_size,
        overlap=sw_overlap
    )

    metrics = [
        # The metrics needs batch dimension (N, C, H, W, [D]) and one-hot-encoding (MONAI v1.1)
        DiceMetric(include_background=False, reduction="mean")
    ]

    if args.mask is not None:
        masks_arr = parse_csv(args.mask)
        assert len(infer_loader) == len(masks_arr) # One line of per evaluation volume. Multiple columns per line for multiple masks
    else:
        masks_arr = []

    for i, batch in enumerate(infer_loader):
        xs, ys, meta = batch["img"].to(device), batch["seg"].to(device), batch["img_meta"]

        predictions = inferer(inputs=xs, network=model)
        predictions = [post_transforms(i) for i in decollate_batch(predictions)]

        ys = [y for y in decollate_batch(ys)]

        for j, prediction in enumerate(predictions):
            logger.info(f"Validation: {meta['filename_or_obj'][j]}")

            saver(prediction)

            ohe_pred = torch.unsqueeze(
                monai.networks.utils.one_hot(prediction, num_classes=2, dim=0),
                axis=0
            )
            ohe_y = torch.unsqueeze(
                monai.networks.utils.one_hot(ys[j], num_classes=2, dim=0),
                axis=0
            )

            for metric in metrics:
                metric(y_pred=ohe_pred, y=ohe_y)
                logger.info("\t- {} = {}".format(type(metric).__name__, metric.aggregate().item()))
                metric.reset()

            # If mask has been provided, for each mask
            if masks_arr:
                for mask in masks_arr[i * infer_loader.batch_size + j]:
                    logger.info(os.path.basename(mask).split("_")[1])

                    mask = torch.unsqueeze(
                        LoadImage(ensure_channel_first=True, image_only=True)(mask)>0, axis=0
                    ).to(device)

                    # Convert to one-hot format, to generalized use amoung metrics and add the batch dimension
                    ohe_masked_pred = torch.unsqueeze(
                        monai.networks.utils.one_hot(prediction*mask, num_classes=2, dim=0),
                        axis=0
                    )
                    ohe_masked_y = torch.unsqueeze(
                        monai.networks.utils.one_hot(ys[j]*mask, num_classes=2, dim=0),
                        axis=0
                    )

                    for metric in metrics:
                        metric(y_pred=ohe_masked_pred, y=ohe_masked_y)
                        logger.info("\t- {} = {}".format(type(metric).__name__, metric.aggregate().item()))
                        metric.reset()


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
