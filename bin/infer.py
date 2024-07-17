import argparse
import logging
import os

from utils.load_hyperparameters import load_hyperparameters

import models.instanciate_model

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
        "csv_infer",
        type=str,
        metavar=("INFER_CSV_PATH"),
        help="Path to the inference CSV",
    )
    parser.add_argument(
        "--hyperparameters",
        "-p",
        type=str,
        metavar=("HYPERPARAMETERS_JSON_PATH"),
        default=os.path.join(cfg.workspace, "resources", "default_hyperparameters.json"),
        help="Path to the hyperparameters JSON",
    )

    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args


def main():
    import torch
    from monai.data import DataLoader
    from monai.inferers import SlidingWindowInferer
    from monai.transforms import (
        Compose,
        Activations,
        AsDiscrete,
        RemoveSmallObjects,
        SaveImage
    )
    from monai.data.utils import decollate_batch, first


    from datasets.instanciate_dataset import instanciate_image_dataset
    from models.instanciate_model import instanciate_model
    from utils.prebuilt_logs import log_hardware

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    hyperparameters = load_hyperparameters(args.hyperparameters)

    infer_ds = instanciate_image_dataset(args.csv_infer, image_only=False)
    infer_loader = DataLoader(infer_ds, batch_size=1, num_workers=0) # Note: batch_size must be equal to 1 because input tensors does not have equal spatial dims. See sw_batch_size instead

    saver = SaveImage(
        output_dir=cfg.result_dir,
        output_ext=".nii",
        output_postfix=f"seg_{model_id}",
        resample=False,
        separate_folder=False
    )

    # Post-processing
    post_transforms = Compose(
        [
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
            RemoveSmallObjects()
        ]
    )

    # Model
    in_channels = first(infer_loader)["img"].shape[1] # We suppose the same input channels through the whole dataset ; N, C, H, W, [D]

    model = instanciate_model(args.model, in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(args.weights)["model_state_dict"])
    model.eval()

    # Inference parameters
    sw_shape = hyperparameters["patch_size"]
    sw_batch_size = 64 # TODO: Move to a cfg file
    sw_overlap = 0.25 # TODO: Move to a cfg file

    inferer = SlidingWindowInferer(
        sw_shape,
        sw_batch_size=sw_batch_size,
        overlap=sw_overlap
    )

    for data in infer_loader:
        x, meta = data["img"].to(device), data["img_meta"]

        pred = inferer(inputs=x, network=model)
        pred = [post_transforms(i) for i in decollate_batch(pred)]

        idx = 0
        for prediction in pred:
            
            saver(prediction)

            logger.info("Save prediction for {}".format(os.path.basename(meta["filename_or_obj"][idx])))
            idx+=1


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
