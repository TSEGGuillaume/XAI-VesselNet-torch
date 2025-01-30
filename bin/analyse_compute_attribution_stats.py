import argparse
import logging

import os
import json
from multiprocessing import Process

from tqdm import tqdm

import numpy as np
from monai.transforms import LoadImage

from metrics.descriptive_statistics import univariate_analysis
from metrics.total_variation import image_total_variation as TotalVariation
from utils.create_output_dirs import create_output_dir
from utils.json_format import convert_typing_to_native

logger = logging.getLogger("app")


def parse_arguments():
    parser = argparse.ArgumentParser()
        
    parser.add_argument(
        "attributions_dir",
        type=str,
        metavar=("ATTRIBUTION_MAPS_DIRECTORY"),
        help="Path to the directory containing attribution maps (*.nii, *.nii.gz)",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        metavar=("OUTPUT_DIR"),
        help="output directory",
        default=None,
    )

    parser.add_argument(
        "--thread",
        "-t",
        type=int,
        metavar=("THREAD_COUNT"),
        help="Number of threads to use",
        default=8,
    )
        
    args = parser.parse_args()
    return args


def task_compute_attribution_stats(thread_idx, files, in_path, out_path):

    image_loader = LoadImage(ensure_channel_first=False, image_only=False)

    for file in tqdm(files):
        fpath = os.path.join(in_path, file)

        I_attr, meta = image_loader(fpath)

        stats = univariate_analysis(I_attr.get_array().flatten())
        
        #tv = TotalVariation(I_attr.get_array(), neighborhood="N26", norm="L1")
        #stats["total_variation"] = tv

        f_basename = "{}".format(file.split(".")[0])
        with open(os.path.join(out_path, f"{f_basename}_stats.json"), "w") as out_file:
            json.dump(convert_typing_to_native(stats), out_file, indent=4)
     
    logger.info(f"Thread {thread_idx}: finished")


def main(in_attributions_dir, out_path, threads_count):
    # I/O   
    if out_path is None:
        out_path = os.path.join(cfg.result_dir, "stats")

    attribution_id = os.path.normpath(in_attributions_dir).split(os.sep)[-1]

    out_dirs = create_output_dir(out_path, attribution_id)

    # List attribution files, for which no blobs mask already exists
    files = [f for f in os.listdir(in_attributions_dir) if f.endswith(".nii.gz")]

    logger.info(f"Files count : {len(files)}")
    nfiles = len(files)

    if nfiles > 0:

        if nfiles < threads_count:
            threads_count = nfiles
        
        files_chunks = np.array_split(files, threads_count)

        threads = []
        for id_chunk, chunk in enumerate(files_chunks):
            chunk = chunk.tolist()

            logger.debug(f"Chunk {id_chunk} : {len(chunk)} files.")

            threads.append(
                Process(target=task_compute_attribution_stats, args=(id_chunk, chunk, in_attributions_dir, out_dirs))
            )
            threads[-1].start()
           
        for thread in threads:
            thread.join()

    else:
        raise RuntimeError("The given directory is empty !")


if __name__ == "__main__":
    import utils.configuration as appcfg

    cfg = appcfg.Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    # Quick and dirty
    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": f"{cfg.log_dir}/attribution_stats.log",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    # Shut the mouth of console handler and LoadImage logger
    for handler in logger.handlers:
        if type(handler) == logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    logging.getLogger('LoadImage').setLevel(logging.ERROR)

    main(args.attributions_dir, args.output, args.thread)