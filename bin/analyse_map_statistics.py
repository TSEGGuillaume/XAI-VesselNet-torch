import argparse
import logging

import os
import json
from multiprocessing import Queue, Process, cpu_count
from queue import Empty

from tqdm import tqdm

import numpy as np
from monai.transforms import LoadImage

from metrics.descriptive_statistics import univariate_analysis
from metrics.total_variation import image_total_variation as TotalVariation
from utils.create_output_dirs import create_output_dir
from utils.json_format import convert_typing_to_native
from utils.template_filename import CXAIVesselNetFilename as Filename

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


def get_attribution_id(in_path):
    return os.path.normpath(in_path).split(os.sep)[-1]


def pbar_listener(queue, ntasks):
    pbar = tqdm(total=ntasks)
    for _ in iter(queue.get, None):
        pbar.update()


def create_map_stats_json(file, in_dir_attribution, image_loader, output_dir):

    struct_fname = Filename(filename=file)

    attr_map = image_loader(os.path.join(in_dir_attribution, file))

    stats = univariate_analysis(attr_map.get_array().flatten())
    #tv = TotalVariation(I_attr.get_array(), neighborhood="N26", norm="L1")
    #stats["total_variation"] = tv

    with open(os.path.join(output_dir, f"{struct_fname.get_filename()}_stats.json"), "w") as out_file:
        json.dump(convert_typing_to_native(stats), out_file, indent=4)


def task_create_map_stats_json(files_queue, files_finished_queue, in_dir_attribution, image_loader, output_dir):
    while True:
        try:
            task_file = files_queue.get()
            
            if task_file is None:
                raise Empty

        except Empty:
            break

        else:
            # No exception raised, process the job and add the task completion to finished_queue
            create_map_stats_json(task_file, in_dir_attribution, image_loader, output_dir)
            files_finished_queue.put(task_file)
    
    return True


def distribute_stats_json_creation(files, in_dir_attribution, image_loader, output_dir, process_count):
    nfiles = len(files)

    process_count = min(min(process_count, cpu_count()), nfiles)

    files_to_process = Queue()
    files_finished = Queue()

    processes = []

    for file in files:
        files_to_process.put(file)

    # Start the progress bar process
    proc = Process(target=pbar_listener, args=(files_finished, nfiles))
    proc.start()

    # creating processes
    for k in range(process_count):
        files_to_process.put(None) # Stop condition
        processes.append(
            Process(target=task_create_map_stats_json, args=(files_to_process, files_finished, in_dir_attribution, image_loader, output_dir))
        )
        processes[-1].start()

    # completing process
    for p in processes:
        p.join()

    files_finished.put(None)
    proc.join()


def main(in_dir_attribution, output_dir, process_count=None):
    attribution_id = get_attribution_id(in_dir_attribution)

    if output_dir is None:
        output_dir = cfg.result_dir

    output_dir = create_output_dir(os.path.join(output_dir, "stats"), attribution_id)

    image_loader = LoadImage(ensure_channel_first=False, image_only=True)

    files = [f for f in os.listdir(in_dir_attribution) if f.endswith(".nii.gz")]

    distribute_stats_json_creation(files, in_dir_attribution, image_loader, output_dir, process_count)


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

    main(args.attributions_dir, args.output, args.thread)