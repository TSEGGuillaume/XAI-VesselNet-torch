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
from utils.load_patch_position import read_path_position_from_file as ReadPositionFile
from utils.distances import distance
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
        "graphs_dir",
        type=str,
        metavar=("GRAPHS_DIRECTORY"),
        help="Path to the vessel graphs (*.vvg)",
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


def pbar_listener(queue, ntasks):
    pbar = tqdm(total=ntasks)
    for _ in iter(queue.get, None):
        pbar.update()


def get_attribution_id(in_path):
    return os.path.normpath(in_path).split(os.sep)[-1]


def create_landmark_info_json(file, in_dir_attribution, graph, affine, output_dir):
        struct_fname = Filename(filename=file)

        # Retrieves info
        landmark = GetLandmark(graph, landmark_type=struct_fname.landmark_type, landmark_id=struct_fname.landmark_id)

        patch_position_path = os.path.join(in_dir_attribution, f"{struct_fname.get_prefix()}_pos.txt")
        patch_pos = ReadPositionFile(patch_position_path)
        relative_landmark_pos = tuple(lpos - ppos for lpos, ppos in zip(landmark.pos, patch_pos[0]))

        input_half_shape = [(w_e - w_s)/2 for w_s, w_e in zip(patch_pos[0], patch_pos[1])]
        dist_landmark_from_patch_center = distance(
            relative_landmark_pos, input_half_shape, norm="L2"
        )

        dict_output = {
            "landmark_type": struct_fname.landmark_type,
            "landmark_id": struct_fname.landmark_id,
            
            "absolute_position": landmark.pos,
            "relative_position": relative_landmark_pos,
            "distance_from_center": dist_landmark_from_patch_center,

            "affine": affine,
        }

        with open(os.path.join(output_dir, f"{struct_fname.get_prefix()}_landmark.json"), "w") as json_file:
            json.dump(convert_typing_to_native(dict_output), json_file, indent=4)


def task_create_landmark_info_json(files_queue, files_finished_queue, in_dir_attribution, graph, affine, output_dir):
    while True:
        try:
            task_file = files_queue.get()
            
            if task_file is None:
                raise Empty

        except Empty:
            break

        else:
            # No exception raised, process the job and add the task completion to finished_queue
            create_landmark_info_json(task_file, in_dir_attribution, graph, affine, output_dir)
            files_finished_queue.put(task_file)
    
    return True


def distribute_landmark_info_json_creation(files: list[str], in_dir_attribution: str, graph, affine, output_dir: str, process_count: None|int):
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
            Process(target=task_create_landmark_info_json, args=(files_to_process, files_finished, in_dir_attribution, graph, affine, output_dir))
        )
        processes[-1].start()

    # completing process
    for p in processes:
        p.join()

    files_finished.put(None)
    proc.join()


def main(in_dir_attribution: str, in_dir_graphs: str, output_dir: str=None, process_count=None):

    attribution_id = get_attribution_id(in_dir_attribution)

    if output_dir is None:
        output_dir = cfg.result_dir

    out_path = create_output_dir(
        os.path.join(output_dir, "landmark", "json"),
        attribution_id
    )

    image_loader = LoadImage(ensure_channel_first=False, image_only=False)

    files = [f for f in os.listdir(in_dir_attribution) if f.endswith(".nii.gz")]

    unique_samples = { "_".join(file.split('_')[:2]) for file in files }

    for sample in unique_samples:
        logger.info(f"Process sample {sample}")
        graph = LoadVesselGraph(os.path.join(in_dir_graphs, f"{sample}_graph.vvg"))

        sample_files = [f for f in files if f.startswith(sample)]
        _, meta = image_loader(os.path.join(in_dir_attribution, sample_files[0])) # We assume affine of all attirbution maps are equal
        affine = meta["original_affine"]

        graph = Anatomic2ImageGraph(graph, affine)

        distribute_landmark_info_json_creation(sample_files, in_dir_attribution, graph, affine, out_path, process_count)


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

    main(args.attributions_dir, args.graphs_dir, args.output, args.thread)