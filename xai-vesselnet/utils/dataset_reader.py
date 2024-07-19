import logging
import csv

logger = logging.getLogger("app")


def parse_csv(csv_path: str, delimiter: str = ";") -> list[list[str]]:
    """
    Read a CSV file (read-only).

    Args:
        csv_path    : Path to the CSV file
        delimiter   : The delimiter used in the CSV. The semicolon character is used by default.

    Returns:
        data (array) : The array of lines contained in the CSV.
    """
    with open(csv_path, mode="r") as csv_file:
        dataset_reader = csv.reader(csv_file, delimiter=delimiter)

        data = [line for line in dataset_reader]

        return data


def parse_dataset_csv(csv_path: str, delimiter: str = ";") -> tuple[list]:
    """
    Read a CSV file (read-only) built as a dataset.
    Each line of the file must contain a paired [image, segmentation] separated by the delimiter, for training or evaluation, or a single image, for inference.
    Additional elements per line are ignored.

    Args:
        csv_path    : Path to the CSV file
        delimiter   : The delimiter used in the CSV. The semicolon character is used by default.

    Returns:
        (x, y) (tuple) : A tuple of array. `x` is the array of images while `y` is the array of segmentations. If no segmentation provided, `y` is returned empty.
    """
    x = []
    y = []

    with open(csv_path, mode="r") as csv_file:
        dataset_reader = csv.reader(csv_file, delimiter=delimiter)

        for line in dataset_reader:
            x.append(line[0])

            if len(line) > 1:
                if len(line) > 2:
                    logger.warning(
                        "Too many elements in the CSV. Only the first two elements per line are taken into account, the others are ignored."
                    )

                y.append(line[1])

    return x, y
