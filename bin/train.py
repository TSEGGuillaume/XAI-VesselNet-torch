import argparse
import logging

from datetime import datetime
import time
import os

import numpy as np
import random
from math import ceil

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch import device
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
)
from monai.metrics import DiceMetric
from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference, SimpleInferer
from monai.data.utils import decollate_batch
from monai.networks.utils import one_hot as OneHotEncoding
from monai.visualize.img2tensorboard import plot_2d_or_3d_image

from utils.configuration import Configuration
from utils.prebuilt_logs import log_hardware
from utils.load_hyperparameters import load_hyperparameters
from models import instanciate_model
from datasets.instanciate_dataset import create_training_loaders


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        metavar=("MODEL"),
        choices=instanciate_model._all_models,
        default=instanciate_model._all_models[0],
        help="Name of the model to train",
    )

    parser.add_argument(
        "csv_train",
        type=str,
        metavar=("TRAIN_CSV_PATH"),
        help="Path to the training CSV",
    )
    parser.add_argument(
        "csv_val",
        type=str,
        metavar=("VAL_CSV_PATH"),
        help="Path to the validation CSV",
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

    args = parser.parse_args()
    return args


def fit(model: Module, train_loader: DataLoader, val_loader: DataLoader, hyperparameters: dict, device:device="cpu") -> None:
    """
    Fit the model

    Args:
        model           : The model
        train_loader    : The dataloader for training
        val_loader      : The dataloader for validation
        hyperparameters : The dictionary of hyperparameters
        device          : The device. Default to "cpu"

    Returns:
        None
    """
    # Training hyperparameters
    start_lr    = hyperparameters["lr"]
    max_epoch   = hyperparameters["epoch"]

    out_channels = hyperparameters["out_channels"]

    if out_channels == 1:
        to_onehot_y = False
        kwargs = { "sigmoid": True, }
    elif out_channels > 1:
        to_onehot_y = True
        kwargs = { "softmax": True, }
    else:
        raise ValueError(f"Expected out channels >= 1, got {out_channels}")
    
    include_background = False

    optimizer = torch.optim.Adam(
        model.parameters(), start_lr
    )  # TODO: Move to a cfg file and make a factory

    loss_function = DiceFocalLoss(
        include_background=include_background, to_onehot_y=to_onehot_y, reduction="mean", **kwargs,
    )  # TODO: Move to a cfg file and make a factory

    # Validation hyperparameters
    val_interval = 1

    sw_batch_size = hyperparameters["batch_size"] # How many sliding windows processed at the same time
    sw_patch_size = hyperparameters["patch_size"]
    
    val_metric = DiceMetric(include_background=include_background, reduction="mean")

    # Define the post-processing (apply to compute val_metric)
    post_transforms = Compose([Activations(**kwargs), AsDiscrete(threshold=0.5)])

    # Instanciate SummaryWriter to track the training process
    writer = SummaryWriter(
        log_dir="logs/runs/{}__{}__LR_{:.2e}__BATCH_{}__{}".format(
            timestamp,
            args.model,
            start_lr,
            train_loader.batch_size,
            type(loss_function).__name__,
        )
    )

    # Start a typical PyTorch training
    best_metric_value       = -1
    best_metric_idx_epoch   = -1

    """
    Approximation, len(dataloader) does not work because we are working on an IterableDataset.
    Manual says that for IterableDataset, len(dataloader) return an approximation of len(dataset) / batch_size, with proper rounding but error here.
    """
    epoch_len = ceil(len(list(train_loader.dataset)) / train_loader.batch_size)

    for idx_epoch in range(1, max_epoch + 1):
        model.train()

        logger.info("\n{}\nEpoch {}/{}".format("-" * 10, idx_epoch, max_epoch))

        steps_count     = 0
        epoch_avg_loss  = 0

        stime = time.time() # Start a timer

        for batch_data in train_loader:
            steps_count += 1

            optimizer.zero_grad()

            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_avg_loss += loss.item()

            logger.debug(f"[{steps_count}/~{epoch_len}] Train loss : {loss.item():.4f}")

        etime = time.time() # Stop the timer

        epoch_avg_loss /= steps_count # Average epoch loss

        writer.add_scalar(
            f"train_{loss_function.reduction}_loss", epoch_avg_loss, idx_epoch
        )  # Prefered to a log per step

        logger.info(
            f"Epoch {idx_epoch} ({etime - stime} s) | average loss: {epoch_avg_loss:.4f}"
        )
        logger.debug("Optimizer: {}".format(optimizer.state_dict()["param_groups"]))

        # If validation requiered
        if (idx_epoch) % val_interval == 0:
            model.eval()

            with torch.no_grad():
            
                stime = time.time() # Start a timer

                val_count       = 0
                val_avg_loss    = 0

                for val_data in val_loader:
                    val_count += 1

                    val_inputs, val_labels = val_data["img"].to(device), val_data["seg"].to(device)

                    if np.unique(sw_patch_size) == -1:
                        val_outputs = SimpleInferer()(val_inputs, model)
                    else:

                        val_outputs = sliding_window_inference(
                            val_inputs, sw_patch_size, sw_batch_size, model
                        )

                    val_loss = loss_function(val_outputs, val_labels) # Compute the validation loss for current iteration
                    logger.debug(f"Current val loss : {val_loss.item()}")

                    val_avg_loss += val_loss.item()

                    val_outputs = torch.stack(
                        [post_transforms(i) for i in decollate_batch(val_outputs)]
                    )  # Decolate to post-process and re-stack

                    # Always encode to One Hot format for metrics
                    if val_outputs.shape[1] == 1:
                        val_labels  = OneHotEncoding(labels=val_labels, num_classes=2)
                        val_outputs = OneHotEncoding(labels=val_outputs, num_classes=2)
                    elif val_outputs.shape[1] >= 1:
                        val_labels  = OneHotEncoding(labels=val_labels, num_classes=val_outputs.shape[1])

                    val_metric(y_pred=val_outputs, y=val_labels) # Compute the metrics for current validation

                val_avg_loss /= val_count

                etime = time.time() # Stop the timer

                reduced_metric = val_metric.aggregate().item() # Execute reduction and aggregation logic
                val_metric.reset() # Reset the metric for next validation round

                if reduced_metric > best_metric_value:
                    best_metric_value = reduced_metric
                    best_metric_idx_epoch = idx_epoch

                    model_save_path = os.path.join(cfg.result_dir, "weights", f"model_{timestamp}.pth")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        model_save_path
                    ) # Save the model
                    logger.debug(f"New best performing model, saved at {model_save_path}")

                logger.info(
                    "Validation step for current epoch: {} ({} s) | Validation loss: {:.4f} | Current {} dice: {:.4f} | Best {} dice: {:.4f} at epoch {}".format(
                        idx_epoch,
                        etime - stime,
                        val_avg_loss,
                        val_metric.reduction,
                        reduced_metric,
                        val_metric.reduction,
                        best_metric_value,
                        best_metric_idx_epoch,
                    )
                )
                writer.add_scalar(f"val_{val_metric.reduction}_dice", reduced_metric, idx_epoch)
                writer.add_scalar(f"val_{loss_function.reduction}_loss", val_avg_loss, idx_epoch)

                # Plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(
                    val_labels, idx_epoch, writer, index=0, max_channels=val_outputs.shape[1], tag="label_"
                )

                plot_2d_or_3d_image(
                    val_outputs, idx_epoch, writer, index=0, max_channels=val_outputs.shape[1], tag="output_"
                )

    logger.info(
        f"Training completed. Best metric: {best_metric_value:.4f} at epoch: {best_metric_idx_epoch}"
    )
    writer.close()


def main():
    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    # Load hyperparameters for training
    hyperparameters = load_hyperparameters(args.hyperparameters)

    # Define variables
    batch_size      = hyperparameters["batch_size"]
    input_shape     = hyperparameters["patch_size"]
    spatial_dims    = len(input_shape)
    in_channels     = hyperparameters["in_channels"]
    out_channels    = hyperparameters["out_channels"]
    logger.debug(f"Input channels : {in_channels} | Output channels : {out_channels}")

    train_dataset_path  = args.csv_train
    val_dataset_path    = args.csv_val

    model_type = args.model

    # Create loaders from provided csv
    train_loader, val_loader = create_training_loaders(
        train_dataset_path,
        val_dataset_path,
        input_shape,
        batch_size,
    )

    # Create the model
    model = instanciate_model.instanciate_model(
        model_type, spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels
    ).to(device)

    # Call the training loop
    fit(model, train_loader, val_loader, hyperparameters, device=device)


if __name__ == "__main__":
    cfg = Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    # Logger
    timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")

    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": "{}/{}_train_{}.log".format(cfg.log_dir, timestamp, args.model),
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    logger.info(timestamp)
    logger.debug(args)

    # Reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    main()
