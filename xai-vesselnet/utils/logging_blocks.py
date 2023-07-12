import logging

def log_hardware(device):
    import torch

    logger.info("{}".format("-" * 10))
    logger.info("HARDWARE INFORMATIONS")
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
    logger.info("{}".format("-" * 10))

logger = logging.getLogger("app")
