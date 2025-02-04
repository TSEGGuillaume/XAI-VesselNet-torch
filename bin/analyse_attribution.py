import argparse
import logging
import logging.config # Mandatory if MONAI is not imported

import os
import json
from tqdm import tqdm

from multiprocessing import Queue, Process, cpu_count
from queue import Empty

import utils.configuration as AppCfg
from utils.json_format import convert_typing_to_native
from utils.create_output_dirs import create_output_dir
from utils.template_filename import CXAIVesselNetFilename as Filename


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "attribution",
        type=str,
        metavar=("ATTRIBUTION_DIR"),
        help="Path to the attribution maps directory",
    )

    parser.add_argument(
        "output",
        type=str,
        metavar=("OUTPUT_DIR"),
        help="Output directory",
    )

    parser.add_argument(
        "--hyperparameters",
        type=str,
        metavar=("HYPERPARAMETERS_PATH"),
        default="./resources/default_hyperparameters.json",
        help="Path to the hyperparameters file (*.json)",
    )

    parser.add_argument("--thread",
        "-t",
        type=int,
        metavar=("NUMBER_OF_THREADS"),
        help="The number of threads to be used.",
        default=1,
    )

    args = parser.parse_args()
    return args


def get_attribution_id(in_path):
    return os.path.normpath(in_path).split(os.sep)[-1]


def pbar_listener(queue, ntasks):
    pbar = tqdm(total=ntasks)
    for _ in iter(queue.get, None):
        pbar.update()


def create_attribution_json(file, input_json_dirs, output_dir):
    struct_fname = Filename(filename=file)

    fname_prefix = struct_fname.get_prefix()
    fname_basename = struct_fname.get_filename()

    with open(os.path.join(input_json_dirs["landmark"], f"{fname_prefix}_landmark.json"), "r") as json_file:
        landmark_data = json.load(json_file)
            
    with open(os.path.join(input_json_dirs["tubularity"], f"{fname_prefix}_tubularity.json"), "r") as json_file:
        tubularity_data = json.load(json_file)

    with open(os.path.join(input_json_dirs["connectivity"], f"{fname_prefix}_connectivity.json"), "r") as json_file:
        connectivity_data = json.load(json_file)

    with open(os.path.join(input_json_dirs["thickness"], f"{fname_prefix}_eedt.json"), "r") as json_file:
        thickness_data = json.load(json_file)

    with open(os.path.join(input_json_dirs["patch"], f"{fname_prefix}_patch.json"), "r") as json_file:
        inference_data = json.load(json_file)

    # Depends on channels, use f_basename
    with open(os.path.join(input_json_dirs["stats"], f"{fname_basename}_stats.json"), "r") as json_file:
        stats_data = json.load(json_file)

    with open(os.path.join(input_json_dirs["blobs"], f"{fname_basename}_blobs.json"), "r") as json_file:
        blobs_data = json.load(json_file)

    out_dict = {
        "point"         : None,
        "inference"     : None,
        "attribution"   : None,
    }

    # Information about the landmark
    out_dict["point"] = landmark_data | connectivity_data | thickness_data | { "tubularity_probs": tubularity_data }
    out_dict["inference"] = inference_data
    out_dict["attribution"] = stats_data | { "blobs": blobs_data }
        
    with open(os.path.join(output_dir, f"res_{fname_basename}.json"), "w") as json_file:
        json.dump(convert_typing_to_native(out_dict), json_file, indent=4)


def task_create_attribution_json(files_queue, files_finished_queue, input_json_dirs, output_dir):
    while True:
        try:
            task_file = files_queue.get()
            
            if task_file is None:
                raise Empty

        except Empty:
            break

        else:
            # No exception raised, process the job and add the task completion to finished_queue
            create_attribution_json(task_file, input_json_dirs, output_dir)
            files_finished_queue.put(task_file)
    
    return True


def distribute_attribution_json_creation(files, in_json_dirs, output_dir, process_count):
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
            Process(target=task_create_attribution_json, args=(files_to_process, files_finished, in_json_dirs, output_dir))
        )
        processes[-1].start()

    # completing process
    for p in processes:
        p.join()

    files_finished.put(None)
    proc.join()


def main(in_dir_attribution, output_dir=None, process_count=1):
    attribution_id = get_attribution_id(in_dir_attribution)

    if output_dir is None:
        output_dir = cfg.result_dir

    output_dir = create_output_dir(os.path.join(output_dir, "attributions", "json"), attribution_id)

    input_json_dirs = {
        "landmark":     os.path.join(cfg.result_dir, "landmark",    "json", attribution_id),
        "tubularity":   os.path.join(cfg.result_dir, "tubularity",  "json", attribution_id),
        "connectivity": os.path.join(cfg.result_dir, "connectivity","json", attribution_id),
        "thickness":    os.path.join(cfg.result_dir, "thickness",   "json", attribution_id),
        "patch":        os.path.join(cfg.result_dir, "patch",       "json", attribution_id),
        "stats":        os.path.join(cfg.result_dir, "stats",       "json", attribution_id),
        "blobs":        os.path.join(cfg.result_dir, "blobs",       "json", attribution_id),
    }

    logger.info("Checking the file environment:")
    for k_path, v_path in input_json_dirs.items():
        path_ok = os.path.exists(v_path)
        
        logger.info(f"\t* {k_path}: {v_path} --> {path_ok}")

        if not path_ok:
            raise EnvironmentError(f"Environment error: {v_path} directory is missing")
        
    files = [f for f in os.listdir(in_dir_attribution) if f.endswith(".nii.gz")]

    distribute_attribution_json_creation(files, input_json_dirs, output_dir, process_count)


if __name__ == "__main__":

    cfg = AppCfg.Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()
    print(args)

    print(logging.__file__)

    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": "{}/last_analyse.log".format(cfg.log_dir),
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    main(args.attribution, args.output, args.thread)