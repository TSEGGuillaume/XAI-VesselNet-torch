import argparse
import logging
import logging.config # Mandatory if MONAI is not imported

import os
import json
from multiprocessing import Process

from tqdm import tqdm

import numpy as np
from monai.transforms import LoadImage

from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2ImageGraph

from utils.create_output_dirs import create_output_dir
from utils.json_format import convert_typing_to_native
from utils.get_landmark_from_args import get_landmark_obj as GetLandmark
from utils.load_patch_position import read_path_position_from_file as ReadPositionFile
from utils.distances import distance


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


def task_compute_landmark_info(thread_id, files, in_path_attr, in_path_graphs, out_path):
    for file in tqdm(files):
        f_basename = os.path.basename(file).split(".")[0] # The blob mask share the exact attribution name + postfix "_blobs_label"
        
        decompose_fname = f_basename.split("_")

        sample_id = "_".join(decompose_fname[:2])
        f_prefix = "_".join(decompose_fname[:-2]) # Shared by every files that do not depend on input channel's attribution

        landmark_type = decompose_fname[6]
        landmark_id = decompose_fname[7]

        # I/O
        _, meta = LoadImage(ensure_channel_first=False, image_only=False)(os.path.join(in_path_attr, file))
        affine = meta["original_affine"]

        graph = LoadVesselGraph(os.path.join(in_path_graphs, f"{sample_id}_graph.vvg"))        
        graph = Anatomic2ImageGraph(graph, affine)
        
        # Retrieves info
        landmark = GetLandmark(graph, landmark_type=landmark_type, landmark_id=landmark_id)

        patch_position_path = os.path.join(in_path_attr, f"{f_prefix}_pos.txt")
        patch_pos = ReadPositionFile(patch_position_path)
        relative_landmark_pos = tuple(lpos - ppos for lpos, ppos in zip(landmark.pos, patch_pos[0]))

        input_half_shape = [(w_e - w_s)/2 for w_s, w_e in zip(patch_pos[0], patch_pos[1])]
        dist_landmark_from_patch_center = distance(
            relative_landmark_pos, input_half_shape, norm="L2"
        )

        dict_output = {
            "type": landmark_type,
            "id": landmark_id,
            
            "absolute_position": landmark.pos,
            "relative_position": relative_landmark_pos,
            "distance_from_center": dist_landmark_from_patch_center,

            "affine": affine,
        }

        with open(os.path.join(out_path, f"{f_prefix}_landmark.json"), "w") as json_file:
            json.dump(convert_typing_to_native(dict_output), json_file, indent=4)

    logger.info(f"Thread {thread_id} finished. Result file saved at {out_path}")


def main(in_path_attr, in_path_graphs, out_path, threads_count=1):
    # I/O
    attribution_id = get_attribution_id(in_path_attr)
    out_path = create_output_dir(
        os.path.join(out_path, "landmark"),
        attribution_id
    )

    files = [f for f in os.listdir(in_path_attr) if f.endswith(".nii.gz")]
    nfiles = len(files)
    logger.info(f"Files count : {nfiles}")

    if nfiles > 0:

        if nfiles < threads_count:
            threads_count = nfiles
        
        files_chunks = np.array_split(files, threads_count)

        threads = []
        for id_chunk, chunk in enumerate(files_chunks):
            chunk = chunk.tolist()

            logger.debug(f"Chunk {id_chunk} : {len(chunk)} files.")

            threads.append(
                Process(target=task_compute_landmark_info, args=(id_chunk, chunk, in_path_attr, in_path_graphs, out_path))
            )
            threads[-1].start()
            logger.info(f"Thread {id_chunk}: start")

           
        for thread_idx, thread in enumerate(threads):
            thread.join()
            logger.info(f"Thread {thread_idx}: finished")

    else:
        logger.error("No files found in the directory.")


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

    for handler in logger.handlers:
        if type(handler) == logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    main(args.attributions_dir, args.graphs_dir, cfg.result_dir, args.thread)