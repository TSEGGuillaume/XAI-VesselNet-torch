import argparse
import logging
import os

import torch
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete
)
from monai.transforms import (
    SaveImage
)
from monai.data.utils import decollate_batch, first

from datasets.instanciate_dataset import instanciate_image_dataset
from models.instanciate_model import instanciate_model
import models.instanciate_model
from utils.configuration import Configuration
from utils.logging_blocks import log_hardware


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
        help="Path to the JSON file for inference scheduling",
    )

    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args


def parse_scheduler_json_file(json_path):
    import json

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def infer(model, weights, scheduler, device=None):
    # Repeat some general info because this function can be called either from __main__ or eval.py
    cfg = Configuration(p_filename="./resources/default.ini")
    model_id, ext = os.path.splitext(os.path.basename(weights))
    logger = logging.getLogger("app") 

    # Inference parameters
    BATCH_SIZE = 1 # Do not touch this
    sw_shape = (64, 64, 64) # TODO: Read from the --hyperparameters
    sw_batch_size = 64 # TODO: Move to a cfg file
    sw_overlap = 0.25 # TODO: Move to a cfg file

    # Classic variables
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saver = SaveImage(
        output_dir=cfg.result_dir,
        output_ext=".nii.gz",
        output_postfix=f"seg_{model_id}",
        resample=False,
        separate_folder=False
    )

    post_transforms = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5)
    ])

    # Prepare the data
    xs = [elem["data"] for elem in scheduler["schedule"]]

    infer_ds = instanciate_image_dataset(xs, xs, image_only=False)
    infer_loader = DataLoader(infer_ds, batch_size=BATCH_SIZE, num_workers=0) # Note: batch_size must be equal to 1 because input tensors does not have equal spatial dims. See sw_batch_size instead

    # Model
    in_channels = first(infer_loader)["img"].shape[1] # We suppose the same input channels through the whole dataset ; N, C, H, W, [D]

    model = instanciate_model(model, in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(weights)["model_state_dict"])
    model.eval()

    # Inference
    inferer = SlidingWindowInferer(
        sw_shape,
        sw_batch_size=sw_batch_size,
        overlap=sw_overlap
    )

    # Note that this script was tested for a batch size of 1. Increasing the batch size may result in a crash
    for k, data in enumerate(infer_loader):
        x_batch, meta_batch = data["img"].to(device), data["img_meta"]

        preds = inferer(inputs=x_batch, network=model)
        preds = [post_transforms(i) for i in decollate_batch(preds)]

        for idx, prediction in enumerate(preds):
            saver(prediction)
            logger.info("Save prediction for {}".format(os.path.basename(meta_batch["filename_or_obj"][idx])))

            scheduler["schedule"][k*BATCH_SIZE+idx]["prediction"] = os.path.join(
                saver.folder_layout.output_dir, 
                "{}_{}{}".format(
                    os.path.basename(meta_batch["filename_or_obj"][idx]).split('.')[0], # TODO : This is crap
                    saver.folder_layout.postfix, 
                    saver.folder_layout.ext
                )
            )        

    # It should great to return all predictions directly in case of evaluation, instead of reading saved predictions on disk 


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    inference_scheduler = parse_scheduler_json_file(args.schedule)

    infer(args.model, args.weights, inference_scheduler, device=device)


if __name__ == "__main__":
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
