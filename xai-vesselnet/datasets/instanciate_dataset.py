import logging

from datasets.ImageDataset import CImageDatasetd, CCustomGridPatchDataset


def instanciate_dataset(csv_train_path:str, csv_val_path:str):
    """
    Instantiate 3 datasets (train, validation, test) from CSV files.

    The CSV file must contain a list of data/ground-truth pairs. A line (pair) must comply with the following format:
    path/to/data.nii;path/to/associated/ground-truth.nii

    The test dataset is created from a part of the validation dataset.

    Args:
        csv_train_path: path to CSV file containing training data paths.
        csv_val_path: path to CSV file containing validation data paths.

    Notes :
     1. The pipeline does not processes the random folds generation to simplify reproductibility.
     This means that the whole dataset has to be manually shuffled and splitted by the experimenter beforhand.
     2. At the moment, the test set is limited to a single volume (hardcoded ; due to the lack of data of the IRCAD dataset).

    TODO:
    1. Allow the automatic k-fold generation (and saving to a CSV file) if no CSV provided.
    2. Design a factory to clean the code and allow the extension to other datasets.

    """

    from monai.data.utils import first
    from utils.dataset_reader import parse_dataset_csv as parse_dataset

    train_f_img, train_f_mask = parse_dataset(csv_train_path)
    val_f_img, val_f_mask = parse_dataset(csv_val_path)

    train_ds = CImageDatasetd(
        image_files=train_f_img,
        seg_files=train_f_mask,
        ensure_channel_first=True,
        image_only=True,
    )

    val_ds = CImageDatasetd(
        image_files=val_f_img[:-1],
        seg_files=val_f_mask[:-1],
        ensure_channel_first=True,
        image_only=True,
    )

    test_ds = CImageDatasetd(
        image_files=[val_f_img[-1]],
        seg_files=[val_f_mask[-1]],
        ensure_channel_first=True,
        image_only=True,
    )

    logger.info("Number of train data: {}".format(len(train_ds)))
    logger.info("Number of validation data: {}".format(len(val_ds)))
    logger.info("Number of test data: {}".format(len(test_ds)))

    logger.debug(
        "Shape of first train data : [ {}  -  {} ]".format(
            first(train_ds)["image"].shape, first(train_ds)["mask"].shape
        )
    )
    logger.debug(
        "Shape of first validation data : [ {}  -  {} ]".format(
            first(val_ds)["image"].shape, first(val_ds)["mask"].shape
        )
    )
    logger.debug(
        "Shape of first test data : [ {}  -  {} ]".format(
            first(test_ds)["image"].shape, first(test_ds)["mask"].shape
        )
    )

    return train_ds, val_ds, test_ds


def create_patch_loader(csv_train_path:str, csv_val_path:str, input_size:tuple, batch_size=16):
    """
    Instantiate 2 patch loaders (train, validation) from CSV files.
    Patchs are generated using monai.data.PatchIterd (see https://docs.monai.io/en/stable/data.html#patchiterd) and are then 
    yielded using CCustomGridPatchDataset.
    
    See function instanciate_dataset() to learn more about expected CSV.

    Args:
        csv_train_path: path to CSV file containing training data paths.
        csv_val_path: path to CSV file containing validation data paths.
        input_size: size of the patches
        batch_size: batch size. Defaults to 16.

    Notes:
    1. The preprocessing is operated on the patches.
    2. Applied transform are RandRotate90d, RandFlipd, RandGaussianSmoothd

    TODO:
    1. Externalize the transforms (either in a cfg file or in parameters at least)

    """
    from monai.data import DataLoader, PatchIterd
    from monai.transforms import Compose, RandRotate90d, RandFlipd, RandGaussianSmoothd

    train_ds, val_ds, test_ds = instanciate_dataset(csv_train_path, csv_val_path)

    train_T = Compose(
        RandRotate90d(keys=(["image", "mask"])),
        RandFlipd(keys=(["image", "mask"])),
        RandGaussianSmoothd(keys=(["image", "mask"])),
    )

    patch_iterator = PatchIterd(
        ["image", "mask"], patch_size=input_size, start_pos=(0, 0, 0)
    )

    train_patch_ds = CCustomGridPatchDataset(
        train_ds,
        patch_iterator,
        with_coordinates=False,
        transform=train_T,
        foreground_ratio=0.85,
    )

    # Loaders
    train_loader = DataLoader(train_patch_ds, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    return train_loader, val_loader


logger = logging.getLogger("app")
