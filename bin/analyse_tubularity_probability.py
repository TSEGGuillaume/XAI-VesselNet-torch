import argparse
import logging
import logging.config # Mandatory if MONAI is not imported

import os
import json
from multiprocessing import Queue, Process, cpu_count
from queue import Empty

from tqdm import tqdm

from monai.transforms import LoadImage

from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2ImageGraph
from utils.create_output_dirs import create_output_dir
from utils.json_format import convert_typing_to_native
from utils.get_landmark_from_args import get_landmark_obj as GetLandmark
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
        "filters_dir",
        type=str,
        metavar=("FILTERS_DIRECTORY"),
        help="Path to the dataset containing vesselness filters (*.nii, *.nii.gz)",
    )
    parser.add_argument(
        "graphs_dir",
        type=str,
        metavar=("GRAPHS_DIRECTORY"),
        help="Path to the vessel graphs (*.vvg)",
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

"""
TODO
We can continue the optimization by computing only once per patch as the tubularity props are found in the image given the landmark
This does not really matter at the moment, the entire process is not so long

In terms of writing, main() and distribute_tubularity_probs_json_creation() should be merged, unless distribute_tubularity_probs_json_creation() -> distribute_task_json_creation(), with distribute_task_json_creation a common thread manager for all called analysis processes (tubularity, blobs, ...) using Callback and kwargs as parameters.
"""

def pbar_listener(queue, ntasks):
    pbar = tqdm(total=ntasks)
    for _ in iter(queue.get, None):
        pbar.update()


def get_attribution_id(in_path):
    return os.path.normpath(in_path).split(os.sep)[-1]


def create_tubularity_probs_json(filename, graph, filters_map: dict, output_dir: str):

    out_dict = {data["name"]: None for _, data in filters_map.items()}

    fname = Filename(filename=filename)

    for _, data_filter in filters_map.items():
        I =  data_filter["image"]

        landmark = GetLandmark(graph=graph, landmark_type=fname.landmark_type, landmark_id=fname.landmark_id)

        out_dict[data_filter["name"]] = I[landmark.pos]
    

    json_path = os.path.join(output_dir, "{}_tubularity.json".format(fname.get_prefix()))
            
    with open(json_path, "w") as json_file:
        json.dump(convert_typing_to_native(out_dict), json_file, indent=4)


def task_create_tubularity_probs_json(files_queue, files_finished_queue, graph, filters_map: dict, output_dir: str):
    while True:
        try:
            task_file = files_queue.get()
            
            if task_file is None:
                raise Empty

        except Empty:
            break

        else:
            # No exception raised, process the job and add the task completion to finished_queue
            create_tubularity_probs_json(task_file, graph, filters_map, output_dir)
            files_finished_queue.put(task_file)
    
    return True


def distribute_tubularity_probs_json_creation(files: list[str], graph, map_data: dict, output_dir: str, process_count: None|int):
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
            Process(target=task_create_tubularity_probs_json, args=(files_to_process, files_finished, graph, map_data, output_dir))
        )
        processes[-1].start()

    # completing process
    for p in processes:
        p.join()

    files_finished.put(None)
    proc.join()


def main(in_dir_attribution: str, in_dir_filters: str, in_dir_graphs: str, output_dir: str, process_count=None):

    attribution_id = get_attribution_id(in_dir_attribution)
    output_dir = create_output_dir(
        os.path.join(output_dir, "tubularity"),
        attribution_id
    )

    map_data = {
        "0000": { "name" : "scan", },
        "0001": { "name" : "frangi", },
        "0002": { "name" : "jerman", },
        "0003": { "name" : "sato", },
        "0004": { "name" : "zhang", },
        "0005": { "name" : "meijering", },
        "0006": { "name" : "rorpo", },
    }

    image_loader = LoadImage(ensure_channel_first=False, image_only=False)

    files = [f for f in os.listdir(in_dir_attribution) if f.endswith(".nii.gz")]

    unique_samples = { "_".join(file.split('_')[:2]) for file in files }

    for sample in unique_samples:
        logger.info(f"Process sample {sample}")
        graph = LoadVesselGraph(os.path.join(in_dir_graphs, f"{sample}_graph.vvg"))

        mem_affine = None

        for id_filter, _ in map_data.items():
            image, meta = image_loader(os.path.join(in_dir_filters, f"{sample}_{id_filter}.nii.gz"))

            if mem_affine is None:
                mem_affine = meta["original_affine"]
            else:
                assert (mem_affine == meta["original_affine"]).all(), f"Affines don't match for sample {sample}"        

            map_data[id_filter]["image"] = image
            map_data[id_filter]["meta"] = meta

        graph = Anatomic2ImageGraph(graph, mem_affine)

        sample_files = [f for f in files if f.startswith(sample)]

        distribute_tubularity_probs_json_creation(sample_files, graph, map_data, output_dir, process_count)


if __name__ == "__main__":
    import utils.configuration as appcfg

    cfg = appcfg.Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": f"{cfg.log_dir}/analyse_tubularity.log",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    main(args.attributions_dir, args.filters_dir, args.graphs_dir, cfg.result_dir, args.thread)