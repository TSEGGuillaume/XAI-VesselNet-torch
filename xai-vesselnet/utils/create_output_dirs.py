import os

def create_output_dir(out_path: str, attribution_id: str) -> str:
    """
    Create the directory, if it does not exist, to store the output files.

    Args:
        out_path (str): The path to the output directory to create.
        attribution_id (str): The ID of the attribution.

    Returns:
        str: The path to the output directory.
    """
    out_path = os.path.join(out_path, attribution_id)
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    return out_path


def create_output_dirs(out_paths: list[str], attribution_id: str) -> list[str]:
    """
    Cretae the directories, if they do not exist, to store the output files.

    Args:
        out_paths (list[str]): The paths to the output directories to create.
        attribution_id (str): The ID of the attribution.

    Returns:
        list[str]: The paths to the output directories.
    """
    out_out_paths = []

    for out_path in out_paths:
        out_out_paths.append(
            create_output_dir(out_path, attribution_id)
        )

    return out_out_paths