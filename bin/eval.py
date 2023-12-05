import argparse
import logging
import os
import json


import torch

import monai
from monai.transforms import (
    AsDiscrete,
    LoadImage
)
from monai.metrics import DiceMetric

from metrics.cldice import clDiceMetric

import infer
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
        help="Name of the model",
    )
    parser.add_argument(
        "weights",
        type=str,
        metavar=("WEIGHTS_PATH"),
        help="Path to the model's weights",
    )
    parser.add_argument(
        "schedule",
        type=str,
        metavar=("JSON_SCHEDULING_PATH"),
        help="Path to the JSON file for the evaluation scheduling",
    )

    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args


def parse_eval_scheduling_json_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def main(model, weights, json_scheduler):
    eval_schedule = parse_eval_scheduling_json_file(json_scheduler)

    infer.infer(model, weights, eval_schedule) # infer data and update prediction field 
    with open(json_scheduler, 'w') as f:
        json.dump(eval_schedule, f)

    y_true_paths = [elem["reference"] for elem in eval_schedule["schedule"]]
    y_pred_paths = [elem["prediction"] for elem in eval_schedule["schedule"]]

    if len(y_true_paths) != len(y_pred_paths):
        raise AssertionError("The number of references and prediction must be equal.")
    
    for k in range(len(y_true_paths)):
        logger.info(f"Evaluate sample {y_true_paths[k]}")

        y_true = LoadImage(image_only=True, ensure_channel_first=True)(y_true_paths[k])
        y_pred = LoadImage(image_only=True, ensure_channel_first=True)(y_pred_paths[k])

        ohe_true = torch.unsqueeze(
            monai.networks.utils.one_hot(y_true, num_classes=2, dim=0),
            axis=0
        )
        logger.debug(f"{y_true_paths[k]} : encode to one-hot... {ohe_true.shape}")

        ohe_pred = torch.unsqueeze(
            monai.networks.utils.one_hot(y_pred, num_classes=2, dim=0),
            axis=0
        )
        logger.debug(f"{y_pred_paths[k]} : encode to one-hot... {ohe_pred.shape}")

        # TODO : Gérer les métriques en fonction de ce qui est décrit dans le JSON

        metrics = [
            # The metrics needs batch dimension (N, C, H, W, [D]) and one-hot-encoding (MONAI v1.1)
            DiceMetric(include_background=False, reduction="mean"),
            clDiceMetric(include_background=False, reduction="mean")
            #SurfaceDistanceMetric(include_background=False, symmetric=True, distance_metric="euclidean")
        ]

        # Classic global metrics
        logger.info("Global :")
        for metric in metrics:
            metric(y_pred=ohe_pred, y=ohe_true)
            logger.info("\t- {} = {}".format(type(metric).__name__, metric.aggregate().item()))
            metric.reset()

        # Apply mask if exists
        masks = eval_schedule["schedule"][k]["mask"]

        for mask in masks:
            mask_data = LoadImage(image_only=True, ensure_channel_first=True)(mask)            

            # Prepare the mask : 1/ binarize 2/ one-hot encoding 3/ add the batch channel
            mask_data = AsDiscrete(threshold=0.5)(mask_data)
            ohe_mask = torch.unsqueeze(
                monai.networks.utils.one_hot(mask_data, num_classes=2, dim=0),
                axis=0
            )
            logger.debug(f"{mask} : encode to one-hot... {ohe_mask.shape}")

            masked_ref = ohe_true * ohe_mask
            masked_pred = ohe_pred * ohe_mask

            # Metrics on partitions
            logger.info(f"Mask : {mask}")
            for metric in metrics:
                metric(y_pred=masked_ref, y=masked_pred)
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

    main(args.model, args.weights, args.schedule)
