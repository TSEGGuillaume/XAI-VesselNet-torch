import argparse
import logging
from datetime import datetime
import time
import os

import numpy as np
import random

import torch
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
)
from monai.metrics import DiceMetric
from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data.utils import decollate_batch, first
from monai.visualize.img2tensorboard import plot_2d_or_3d_image

from models import instanciate_model


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
        default=None,
        help="Path to the hyperparameters JSON",
    )

    args = parser.parse_args()
    return args


def load_training_hyperparameters(json_path: str):
    import json

    if json_path is None:
        json_path = os.path.join(cfg.workspace, "resources", "default_hyperparameters.json")
    
    with open(json_path, "r") as f:
        training_cfg = json.load(f)
        logger.debug(
            "Loading hyperparameters from {} \n {}".format(json_path, training_cfg)
        )
        return training_cfg


def fit(model, train_loader, val_loader, hyperparameters:dict, device="cpu"):
    # Hyperparameters
    start_lr = hyperparameters["lr"]
    loss_function = DiceFocalLoss(
        include_background=False,
        sigmoid=True
    )  # TODO: Move to a cfg file and make a factory
    optimizer = torch.optim.Adam(
        model.parameters(),
        start_lr
    )  # TODO: Move to a cfg file and make a factory

    # Validation settings
    val_metric = DiceMetric(include_background=False, reduction="mean")
    post_transforms = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5)
    ])

    # Summary writer to track training
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
    epoch_loss_values = []

    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    metric_values = []

    """
    Approximation, len(dataloader) does not work because we are working on an IterableDataset.
    Manual says that for IterableDataset, len(dataloader) return an approximation of len(dataset) / batch_size, with proper rounding but error here.
    """ 
    epoch_len = round(
        len(list(train_loader.dataset)) / train_loader.batch_size
    )  

    loss = 0.0

    for epoch in range(hyperparameters["epoch"]):
        model.train()

        logger.info("{}".format("-" * 10))
        logger.info("Epoch {}/{}".format(epoch + 1, hyperparameters["epoch"]))

        epoch_loss = 0
        step = 0

        stime = time.time()

        for batch_data in train_loader:
            step += 1

            optimizer.zero_grad()

            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            logger.debug(f"[{step}/~{epoch_len}] Train loss : {loss.item():.4f}")

        etime = time.time()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        writer.add_scalar(
            "train_mean_loss", epoch_loss, epoch + 1
        )  # Prefered to a log per step

        logger.info(
            f"Epoch {epoch + 1} ({etime - stime} s) | average loss: {epoch_loss:.4f}"
        )
        logger.debug("Optimizer: {}".format(optimizer.state_dict()["param_groups"]))

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()

            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None

                sw_batch_size = hyperparameters["batch_size"]

                stime = time.time()

                n_val = 0
                global_val_loss = 0
                for val_data in val_loader:
                    val_images = val_data["img"].to(device)
                    val_labels = val_data["seg"].to(device)

                    val_outputs = sliding_window_inference(
                        val_images, hyperparameters["patch_size"], sw_batch_size, model
                    )
                    val_outputs = [
                        post_transforms(i) for i in decollate_batch(val_outputs)
                    ]

                    # Compute loss and metric for current iteration
                    val_metric(y_pred=val_outputs, y=val_labels)

                    for val_output in val_outputs:
                        loss = loss_function(
                            torch.unsqueeze(val_output, axis=0), val_labels
                        )
                        logger.debug(f"Current val loss : {loss.item()}")
                        global_val_loss += loss.item()
                        n_val += 1

                global_val_loss /= n_val

                etime = time.time()

                metric = val_metric.aggregate().item() # Aggregate the final mean dice result
                val_metric.reset()  # Reset the status for next validation round
                metric_values.append(metric)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                        },
                        os.path.join(cfg.result_dir, "weights", f"model_{timestamp}.pth")
                    )
                    logger.debug("Saved new best performing model")

                logger.info(
                    "Validation step for current epoch: {} ({} s) | Validation loss: {:.4f} | Current mean dice: {:.4f} | Best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1,
                        etime - stime,
                        global_val_loss,
                        metric,
                        best_metric,
                        best_metric_epoch,
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                writer.add_scalar("val_mean_loss", global_val_loss, epoch + 1)

                # Plot the last model output as GIF image in TensorBoard with the corresponding image and label
                #plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(
                    val_outputs, epoch + 1, writer, index=0, tag="output"
                )

    logger.info(
        f"Train completed. Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()


def main():
    from datasets.instanciate_dataset import create_training_loaders
    from utils.logging_blocks import log_hardware

    # Training parameters
    hyperparameters = load_training_hyperparameters(args.hyperparameters)

    # Dataset loaders
    train_loader, val_loader = create_training_loaders(
        args.csv_train,
        args.csv_val,
        hyperparameters["patch_size"],
        hyperparameters["batch_size"]
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    # Model
    in_channels = first(train_loader)["img"].shape[1] # We assume the same input channels through the whole datasets ; N, C, H, W, [D]
    logger.debug(f"Input channels : {in_channels}")

    model = instanciate_model.instanciate_model(
        args.model,
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1
    ).to(device)

    fit(
        model,
        train_loader,
        val_loader,
        hyperparameters=hyperparameters,
        device=device
    )


if __name__ == "__main__":
    from utils.configuration import Configuration

    cfg = Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")

    # Logger
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
