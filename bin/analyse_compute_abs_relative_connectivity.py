import argparse
import logging
import logging.config # Mandatory if MONAI is not imported

import os
import json
from multiprocessing import Process

from copy import deepcopy

import monai.data
from tqdm import tqdm

import numpy as np
import monai
from monai.transforms import Compose, LoadImage, SaveImage, SpatialCrop, SpatialPad, AsDiscrete, ToTensor

from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2ImageGraph
from image.vessel_thickness import distance_map
from utils.load_patch_position import read_path_position_from_file

from utils.create_output_dirs import create_output_dirs
from utils.json_format import convert_typing_to_native
from utils.get_landmark_from_args import get_landmark_obj as GetLandmark
from image.relative_connectivity import compute_relative_degree

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


def task_compute_relative_connectivity(files, in_path_attr, in_path_ytrue, in_path_graphs, out_paths):

    mask_out_path, json_out_path = out_paths

    mask_saver = SaveImage(
        output_dir=mask_out_path,
        output_ext=".nii.gz",
        output_postfix="",
        output_dtype=np.uint8,
        resample=False,
        separate_folder=False,
    )

    # Get the sample
    sample_id = "_".join(files[0].split("_")[:2])

    # TODO: assert that every files in the list are from the same sample
    
    # Load the ground-truth volume
    ytrue, meta = LoadImage(ensure_channel_first=False, image_only=False)(os.path.join(in_path_ytrue, f"{sample_id}.nii.gz"))

    # Load the vascular structure graph
    graph = LoadVesselGraph(os.path.join(in_path_graphs, f"{sample_id}_graph.vvg"))
    graph = Anatomic2ImageGraph(graph, meta["original_affine"])
        
    # Loop over attributions
    for file in tqdm(files):
        decompose_fname = file.split("_")
        basename = "_".join(decompose_fname[:-2])

        patch_pos_file = f"{basename}_pos.txt"
        patch_pos = read_path_position_from_file(os.path.join(in_path_attr, patch_pos_file))

        # Graph and JSON     
        landmark_type = decompose_fname[6]
        landmark_id = decompose_fname[7]
        
        images_dict = None

        if landmark_type == "node":
            landmark_id = int(landmark_id)

            new_degree, images_dict = compute_relative_degree(landmark_id, graph, ytrue, patch_pos, return_images=True)
            abs_degree = graph.nodes[landmark_id].degree

        elif landmark_type == "centerline":
            # No change, not good but the process is not adapted for lines/skeletons. Deserves more work.
            landmark_id = int(landmark_id)

            new_degree = 2
            abs_degree = 2

        elif landmark_type == "position":

            landmark = GetLandmark(graph, landmark_type, landmark_id)       

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

            meta_cpy["filename_or_obj"] = f"{basename}_mask_landmark.nii.gz"
            mask_saver(monai.data.MetaTensor(images_dict["mask_landmark"], meta=meta_cpy))

            meta_cpy["filename_or_obj"] = f"{basename}_skel_cc_landmark.nii.gz"
            mask_saver(monai.data.MetaTensor(images_dict["skel_cc_landmark"], meta=meta_cpy))
            
            meta_cpy["filename_or_obj"] = f"{basename}_skel_exclude_mask_landmark.nii.gz"
            mask_saver(monai.data.MetaTensor(images_dict["skel_exclude_mask_landmark"], meta=meta_cpy))

            meta_cpy["filename_or_obj"] = f"{basename}_patch.nii.gz"
            mask_saver(monai.data.MetaTensor(images_dict["patch"], meta=meta_cpy))

        with open(os.path.join(json_out_path, f"{basename}_connectivity.json"), "w") as json_f:
            json.dump(
                {
                    "landmark_type": landmark_type,
                    "landmark_id": landmark_id,
                    "absolute_degree": abs_degree,
                    "relative_degree":  new_degree
                },
                json_f
            )


def main(in_path_attr, in_path_ytrue, in_path_graphs, out_path, threads_count=1):
    # I/O
    attribution_id = get_attribution_id(in_path_attr)
    out_path = create_output_dirs(
        [
            os.path.join(out_path, "connectivity", "mask"),
            os.path.join(out_path, "connectivity", "json"),
        ],
        attribution_id
    )

    files = [f for f in os.listdir(in_path_attr) if f.endswith(".nii.gz")]
    nfiles = len(files)
    logger.info(f"Files count : {nfiles}")

    if nfiles > 0:

        samples = {}

        # Search for samples to process
        for file in files:
            decompose_fname = file.split("_")
            sample_id = "_".join(decompose_fname[:2])

            if sample_id not in samples.keys():
                samples[sample_id] = [file]
            else:
                samples[sample_id].append(file)

        for sample_id, sample_files in samples.items():

            logger.info(f"Processing sample {sample_id}")

            nfiles = len(sample_files)

            if nfiles < threads_count:
                threads_count = nfiles

            sample_files_chunks = np.array_split(sample_files, threads_count)

            threads = []
            for id_sample_chunk, sample_chunk in enumerate(sample_files_chunks):
                sample_chunk = sample_chunk.tolist()

                logger.debug(f"Chunk {id_sample_chunk} : {len(sample_chunk)} files.")

                threads.append(
                    Process(
                        target=task_compute_relative_connectivity,
                        args=(
                            sample_chunk,
                            in_path_attr,
                            in_path_ytrue,
                            in_path_graphs,
                            out_path
                        )
                    )
                )
                threads[-1].start()
                logger.info(f"Thread {id_sample_chunk}: start")

           
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
            "filename": f"{cfg.log_dir}/analyse_connectivity.log",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    for handler in logger.handlers:
        if type(handler) == logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    main(args.attributions_dir, args.ytrue_dir, args.graphs_dir, cfg.result_dir, args.thread)