import argparse
import logging
import logging.config # Mandatory if MONAI is not imported

import os
import json
from multiprocessing import Queue, Process, cpu_count
from queue import Empty

from copy import deepcopy

import monai.data
from tqdm import tqdm

import numpy as np
import monai
from monai.transforms import LoadImage, SaveImage

from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2ImageGraph
from utils.load_patch_position import read_path_position_from_file

from utils.create_output_dirs import create_output_dirs
from utils.json_format import convert_typing_to_native
from utils.get_landmark_from_args import get_landmark_obj as GetLandmark
from image.relative_connectivity import compute_relative_degree
from utils.template_filename import CXAIVesselNetFilename as Filename


logger = logging.getLogger("app")


def parse_arguments():
    parser = argparse.ArgumentParser()
        
    parser.add_argument(
        "attributions_dir",
        type=str,
        metavar=("MAP_POSITION_DIRECTORY"),
        help="Path to the directory containing attribution maps positions (*.txt)",
    )
    parser.add_argument(
        "ytrue_dir",
        type=str,
        metavar=("GROUNDTRUTH_DIRECTORY"),
        help="Path to the directory containing ground-truth volumes (*.nii, *.nii.gz)",
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


def get_attribution_id(in_path):
    return os.path.normpath(in_path).split(os.sep)[-1]


def pbar_listener(queue, ntasks):
    pbar = tqdm(total=ntasks)
    for _ in iter(queue.get, None):
        pbar.update()


def create_vessel_connectivity_json(file, in_dir_attribution, ytrue, meta, graph, mask_saver, out_dir_json):

    struct_fname = Filename(filename=file)
    fname_prefix = struct_fname.get_prefix()

    # patch_pos = read_path_position_from_file(os.path.join(in_dir_attribution, f"{fname_prefix}_pos.txt"))
    patch_pos = read_path_position_from_file(os.path.join(in_dir_attribution, file))
        
    images_dict = None

    if struct_fname.landmark_type == "node":
        landmark_id = int(struct_fname.landmark_id)

        new_degree, images_dict = compute_relative_degree(int(landmark_id), graph, ytrue, patch_pos, return_images=True)
        abs_degree = graph.nodes[landmark_id].degree

    elif struct_fname.landmark_type == "centerline":
        # No change, not good but the process is not adapted for lines/skeletons. Deserves more work.
        landmark_id = int(struct_fname.landmark_id)

        new_degree = 2
        abs_degree = 2

    elif struct_fname.landmark_type == "position":
        landmark = GetLandmark(graph, struct_fname.landmark_type, struct_fname.landmark_id)    
        landmark_id = struct_fname.landmark_id

        new_degree = 0
        abs_degree = 0

        check_value = ytrue[landmark.pos].item()
        if check_value != 0:
            new_degree = -1
            abs_degree  -1
            raise RuntimeWarning(f"Position {landmark._id} for sample {file} belong to vessel class ({check_value}) The degree can't be computed.")
            
    if images_dict is not None:
        meta_cpy = deepcopy(meta)
        image_space_new_origin = np.array(patch_pos[0])
        world_space_new_origin = np.matmul(meta["affine"][:-1, :-1], np.atleast_2d(image_space_new_origin).T)
        meta_cpy["affine"][:-1,-1:] = meta["affine"][:-1,-1:] + world_space_new_origin # [., -1:] to keep 2D dimension of the returned array

        meta_cpy["filename_or_obj"] = f"{fname_prefix}_mask_landmark.nii.gz"
        mask_saver(monai.data.MetaTensor(images_dict["mask_landmark"], meta=meta_cpy))

        meta_cpy["filename_or_obj"] = f"{fname_prefix}_skel_cc_landmark.nii.gz"
        mask_saver(monai.data.MetaTensor(images_dict["skel_cc_landmark"], meta=meta_cpy))
            
        meta_cpy["filename_or_obj"] = f"{fname_prefix}_skel_exclude_mask_landmark.nii.gz"
        mask_saver(monai.data.MetaTensor(images_dict["skel_exclude_mask_landmark"], meta=meta_cpy))

        meta_cpy["filename_or_obj"] = f"{fname_prefix}_patch.nii.gz"
        mask_saver(monai.data.MetaTensor(images_dict["patch"], meta=meta_cpy))

    with open(os.path.join(out_dir_json, f"{fname_prefix}_connectivity.json"), "w") as json_f:
        json.dump(
            convert_typing_to_native(
                {
                    "landmark_type": struct_fname.landmark_type,
                    "landmark_id": landmark_id,
                    "absolute_degree": abs_degree,
                    "relative_degree":  new_degree
                }
            ),
            json_f
        )


def task_create_vessel_connectivity_json(files_queue, files_finished_queue, in_dir_attribution, I, meta, graph, image_saver, out_dir_json):
    while True:
        try:
            task_file = files_queue.get()
            
            if task_file is None:
                raise Empty

        except Empty:
            break

        else:
            # No exception raised, process the job and add the task completion to finished_queue
            create_vessel_connectivity_json(task_file, in_dir_attribution, I, meta, graph, image_saver, out_dir_json)
            files_finished_queue.put(task_file)
    
    return True


def distribute_vessel_connectivity_json_creation(files: list[str], in_dir_attribution: str, I, meta, graph, image_saver, out_dir_json, process_count=None):
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
            Process(target=task_create_vessel_connectivity_json, args=(files_to_process, files_finished, in_dir_attribution, I, meta, graph, image_saver, out_dir_json))
        )
        processes[-1].start()

    # completing process
    for p in processes:
        p.join()

    files_finished.put(None)
    proc.join()


def main(in_dir_attribution, in_dir_ytrue, in_dir_graphs, output_dir, process_count=None):
    attribution_id = get_attribution_id(in_dir_attribution)

    if output_dir is None:
        output_dir = cfg.result_dir

    out_dir_masks, out_dir_json = create_output_dirs(
        [
            os.path.join(output_dir, "connectivity", "mask"),
            os.path.join(output_dir, "connectivity", "json"),
        ],
        attribution_id
    )

    image_saver = SaveImage(
        output_dir=out_dir_masks,
        output_ext=".nii.gz",
        output_postfix="",
        output_dtype=np.uint8,
        resample=False,
        separate_folder=False,
    )

    image_loader = LoadImage(ensure_channel_first=False, image_only=False)

    # files = [f for f in os.listdir(in_dir_attribution) if f.endswith(".nii.gz")]
    files = [f for f in os.listdir(in_dir_attribution) if f.endswith(".txt")]

    unique_samples = { "_".join(file.split('_')[:2]) for file in files }

    for sample in unique_samples:
        logger.info(f"Process sample {sample}")

        I, meta = image_loader(os.path.join(in_dir_ytrue, f"{sample}.nii.gz"))

        graph = LoadVesselGraph(os.path.join(in_dir_graphs, f"{sample}_graph.vvg"))
        graph = Anatomic2ImageGraph(graph, meta["original_affine"])

        sample_files = [f for f in files if f.startswith(sample)]

        distribute_vessel_connectivity_json_creation(sample_files, in_dir_attribution, I, meta, graph, image_saver, out_dir_json, process_count)


if __name__ == "__main__":
    import utils.configuration as appcfg

    cfg = appcfg.Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": f"{cfg.log_dir}/analyse_connectivity.log",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    main(args.attributions_dir, args.ytrue_dir, args.graphs_dir, args.output, args.thread)