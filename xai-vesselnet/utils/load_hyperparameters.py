import logging

import os
import json

logger = logging.getLogger("app")


def load_hyperparameters(json_path: str):

    with open(json_path, "r") as f:
        hyperparameters_cfg = json.load(f)

        logger.debug(
            "Loading hyperparameters from {} \n {}".format(
                json_path, hyperparameters_cfg
            )
        )

        return hyperparameters_cfg
