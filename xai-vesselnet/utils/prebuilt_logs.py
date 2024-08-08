import logging

def log_hardware(device):
    import torch

    logger.info("{}".format("-" * 10))

    logger.info("HARDWARE INFORMATIONS")
    
    logger.info(
        f"CUDA available : {torch.cuda.is_available()}  [ :{torch.cuda.device_count()} device(s) ]" if torch.cuda.is_available() else f"CUDA available : {torch.cuda.is_available()}" 
    )
    logger.info(
        f"Current device : {device}:{torch.cuda.current_device()} ({torch.cuda.get_device_name(device)})" if device == "cuda" else f"Current device : {device}"
    )

    logger.info("{}".format("-" * 10))

logger = logging.getLogger("app")
