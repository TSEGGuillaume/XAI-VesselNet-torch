import logging

import os
import json

logger = logging.getLogger("app")


def load_hyperparameters(json_path: str) -> dict:
    """
    Load hyperparameters from a JSON file.
    See default_hyperparameters.json for the file structure

    Args:
        json_path   : Path to the JSON file

    Returns:
        hyperparameters_cfg (dict) : The dictionnary containing hyperparameters
    """
    with open(json_path, "r") as f:
        hyperparameters_cfg = json.load(f)

        logger.debug(
            "Loading hyperparameters from {} \n {}".format(
                json_path, hyperparameters_cfg
            )
        )

        return hyperparameters_cfg
