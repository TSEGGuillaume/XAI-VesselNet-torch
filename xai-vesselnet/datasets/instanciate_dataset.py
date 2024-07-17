import logging

from datasets.ImageDataset import ImageDatasetd, BalancedGridPatchDataset


def instanciate_image_dataset(csv_path:str, image_only=True):
    from monai.data.utils import first
    from utils.dataset_reader import parse_dataset_csv

    image, mask = parse_dataset_csv(csv_path)
    ds = ImageDatasetd(
        image_files=image,
        seg_files=mask,
        ensure_channel_first=True,
        image_only=image_only,
    )

    logger.debug("Number of data: {}".format(len(ds)))
    logger.debug(
        "Shape of first data : [ {}  -  {} ]".format(
            first(ds)["img"].shape,
            first(ds)["seg"].shape
        )
    )

    return ds


def create_training_loaders(csv_train_path:str, csv_val_path:str, input_size:tuple, batch_size=16):
    """
    Instantiate patch loaders used in the training pipeline (train, validation) from CSV files.
    Patchs are generated using monai.data.PatchIterd (see https://docs.monai.io/en/stable/data.html#patchiterd) and are then 
    yielded using BalancedGridPatchDataset.
    
    See function instanciate_dataset() to learn more about expected CSV.

    Args:
        csv_train_path: path to CSV file containing training data paths.
        csv_val_path: path to CSV file containing validation data paths.
        input_size: size of the patches
        batch_size: batch size. Defaults to 16.

    Notes:
    1. The preprocessing is operated across patches.
    2. Applied transform are RandRotate90d, RandFlipd, RandGaussianSmoothd

    TODO:
    1. Externalize the transforms (either in a cfg file or in parameters at least)

    """
    from monai.data import DataLoader, PatchIterd
    from monai.transforms import (
        Compose,
        RandRotate90d,
        RandFlipd,
        RandGaussianSmoothd
    )

    train_ds    = instanciate_image_dataset(csv_train_path)
    val_ds      = instanciate_image_dataset(csv_val_path)

    train_T = Compose(
        [
            RandRotate90d(keys=(["img", "seg"])),
            RandFlipd(keys=(["img", "seg"])),
            RandGaussianSmoothd(keys=(["img", "seg"])),
        ]
    )

    patch_iterator = PatchIterd(
        ["img", "seg"], patch_size=input_size, start_pos=(0, 0, 0)
    )

    train_patch_ds = BalancedGridPatchDataset(
        train_ds,
        patch_iterator,
        with_coordinates=False,
        transform=train_T,
        foreground_ratio=0.85,
    )

    # Loaders
    train_loader    = DataLoader(train_patch_ds, batch_size=batch_size, num_workers=0)
    val_loader      = DataLoader(val_ds, batch_size=1, num_workers=0)

    return train_loader, val_loader


logger = logging.getLogger("app")