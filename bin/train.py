import argparse
from datetime import datetime
import logging
import os
import time

import numpy as np
import random

import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
)
from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data.utils import decollate_batch, first
from monai.visualize.img2tensorboard import plot_2d_or_3d_image

import models.instanciate_model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        metavar=("MODEL"),
        choices=models.instanciate_model._all_models,
        default=models.instanciate_model._all_models[0],
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

    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args


def fit(model, train_loader, val_loader, device="cpu", max_epoch=None, input_size=None):
    MAX_EPOCH = max_epoch
    PATCH_SIZE = input_size

    # Hyperparameters
    start_lr = 1e-4  # TODO: Move to a cfg file
    loss_function = DiceFocalLoss(
        include_background=False, sigmoid=True
    )  # TODO: Move to a cfg file and make a factory
    optimizer = torch.optim.Adam(
        model.parameters(), start_lr
    )  # TODO: Move to a cfg file and make a factory

    val_metric = DiceMetric(include_background=False, reduction="mean")
    post_transforms = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Start a typical PyTorch training
    epoch_loss_values = list()

    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    metric_values = list()

    writer = SummaryWriter(
        log_dir="logs/runs/{}__{}__LR_{:.2e}__BATCH_{}__{}".format(
            timestamp,
            args.model,
            start_lr,
            train_loader.batch_size,
            type(loss_function).__name__,
        )
    )

    # len(dataloader) does not work because we are working on an IterableDataset.
    # Manual says that for IterableDataset, len(dataloader) return an approximation of len(dataset) / batch_size, with proper rounding but error here.
    epoch_len = round(len(list(train_loader.dataset)) / train_loader.batch_size) # Approx.

    loss = 0.0

    for epoch in range(MAX_EPOCH):
        model.train()

        logger.info("{}".format("-" * 10))
        logger.info("Epoch {}/{}".format(epoch + 1, MAX_EPOCH))

        epoch_loss = 0
        step = 0

        stime = time.time()

        for batch_data in train_loader:
            step += 1

            optimizer.zero_grad()

            inputs, labels = batch_data["image"].to(device), batch_data["mask"].to(
                device
            )
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

            logger.info(f"[{step}/~{epoch_len}] Train loss : {loss.item():.4f}")

        etime = time.time()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        writer.add_scalar("train_mean_loss", epoch_loss, epoch + 1) # Prefered to a log per step

        logger.info(
            f"Epoch {epoch + 1} ({etime - stime} s) | average loss: {epoch_loss:.4f}"
        )

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()

            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None

                sw_batch_size = train_loader.batch_size

                stime = time.time()

                n_val = 0
                global_val_loss = 0
                for val_data in val_loader:
                    n_val += 1
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["mask"].to(device)

                    val_outputs = sliding_window_inference(
                        val_images, PATCH_SIZE, sw_batch_size, model
                    )
                    val_outputs = [
                        post_transforms(i) for i in decollate_batch(val_outputs)
                    ]

                    # Compute loss and metric for current iteration
                    val_metric(y_pred=val_outputs, y=val_labels)

                    loss = loss_function(
                        torch.unsqueeze(val_outputs[0], axis=0), val_labels
                    ) # Bad ! TODO : reimplement properly
                    logger.info(f"Current val loss : {loss.item()}")
                    global_val_loss += loss.item()

                global_val_loss /= n_val

                etime = time.time()

                metric = (
                    val_metric.aggregate().item()
                )  # Aggregate the final mean dice result
                val_metric.reset()  # Reset the status for next validation round
                metric_values.append(metric)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(), "results/model_{}.pth".format(timestamp)
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
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(
                    val_outputs, epoch + 1, writer, index=0, tag="output"
                )

    logger.info(
        f"Train completed. Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()


def main():
    from datasets.instanciate_dataset import create_patch_loader
    from models.instanciate_model import instanciate_model

    # Training parameters
    PATCH_SIZE = (64, 64, 64)  # TODO: Move to a cfg file
    BATCH_SIZE = 16 # TODO: Move to a cfg file
    MAX_EPOCH = 300  # TODO: Move to a cfg file

    # Dataset loader
    train_loader, val_loader = create_patch_loader(
        args.csv_train, args.csv_val, PATCH_SIZE, BATCH_SIZE
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hardware log
    logger.info("{}".format("_" * 10))
    logger.info("HARDWARE")
    logger.info(
        "CUDA available : {}  [ :{} device(s) ]".format(
            torch.cuda.is_available(), torch.cuda.device_count()
        )
    )
    logger.info(
        "Current device : {}:{} ({})".format(
            device, torch.cuda.current_device(), torch.cuda.get_device_name(device)
        )
    )

    # Model
    in_channels = first(train_loader)["image"].shape[1]  # We assume the same input channels through the whole datasets ; N, C, H, W, [D]
    logger.info(f"Input channels : {in_channels}")

    model = instanciate_model(args.model, in_channels=in_channels).to(device)
    fit(
        model,
        train_loader,
        val_loader,
        device=device,
        max_epoch=MAX_EPOCH,
        input_size=PATCH_SIZE,
    )


if __name__ == "__main__":
    import utils.configuration as appcfg

    cfg = appcfg.CConfiguration(p_filename="./resources/default.ini")

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

    # Reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    # Logs
    logger.debug(timestamp)
    logger.debug(args)

    main()
