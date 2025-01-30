import argparse
import logging
import logging.config # Mandatory if MONAI is not imported

import os
import json
from multiprocessing import Process

from copy import deepcopy

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


def extract_patch(I: monai.data.MetaTensor, patch_pos: tuple[tuple], patch_size: tuple) -> monai.data.MetaTensor:
    """
    Extract a patch from a volume at a given position and size.

    Args:
        I : The volume to extract the patch from.
        patch_pos : The position of the patch in the volume. Tuple of two tuples (start, end).
        patch_size : The size of the patch to extract.

    Returns:
        The extracted patch.
    """
    start_roi, end_roi = patch_pos

    pipeline_T = Compose(
        [
            # Crop the volume to patch at roi position, and pad the patch if patch size is inferior to input size.
            SpatialCrop(roi_start=start_roi, roi_end=end_roi),
            SpatialPad(spatial_size=patch_size),
        ]
    )

    # Not supposed to be batched but more safe
    I = pipeline_T(I)

    return I


def task_compute_relative_thickness(files, in_path_attr, in_path_ytrue, in_path_graphs, out_paths):

    pre_T = Compose([
        AsDiscrete(threshold=0.5),  # Ensure the ground-truth is binary
        ToTensor(),
    ])

    global_eedt_out_path, patch_eedt_out_path, json_out_path = out_paths

    image_saver = SaveImage(
        output_dir=global_eedt_out_path,
        output_ext=".nii.gz",
        output_postfix="eedt_label",
        output_dtype=np.int32,
        resample=False,
        separate_folder=False,
    )

    # Get the sample
    sample_id = "_".join(files[0].split("_")[:2])

    # TODO: assert that every files in the list are from the same sample
    
    # Load the ground-truth volume
    ytrue, meta = LoadImage(ensure_channel_first=True, image_only=False)(os.path.join(in_path_ytrue, f"{sample_id}.nii.gz"))
    ytrue = pre_T(ytrue)

    # Load the vascular structure graph
    graph = LoadVesselGraph(os.path.join(in_path_graphs, f"{sample_id}_graph.vvg"))
    graph = Anatomic2ImageGraph(graph, meta["original_affine"])
        
    # Global EEDT map
    only_spatial = np.squeeze(ytrue.numpy(), axis=0)
    eedt_map = distance_map(only_spatial)

    image_saver(
        monai.data.MetaTensor(eedt_map, meta=meta)
    )

    image_saver.folder_layout.output_dir = patch_eedt_out_path

    # Loop over attributions
    for file in tqdm(files):
        decompose_fname = file.split("_")
        basename = "_".join(decompose_fname[:-2])

        patch_pos_file = f"{basename}_pos.txt"
        patch_pos = read_path_position_from_file(os.path.join(in_path_attr, patch_pos_file))

        patch_ytrue = np.squeeze(extract_patch(ytrue, patch_pos, (64, 64, 64)).numpy(), axis=0) # TODO: patch size is hardcoded, we need to read the config instead

        eedt_map_patch = distance_map(patch_ytrue)

        # Change metadata
        meta_cpy = deepcopy(meta)
        meta_cpy["filename_or_obj"] = f"{basename}_eedt_label.nii.gz"

        image_space_new_origin = np.array(patch_pos[0])
        world_space_new_origin = np.matmul(meta["affine"][:-1, :-1], np.atleast_2d(image_space_new_origin).T)
        meta_cpy["affine"][:-1,-1:] = meta["affine"][:-1,-1:] + world_space_new_origin # [., -1:] to keep 2D dimension of the returned array

        image_saver(
            monai.data.MetaTensor(eedt_map_patch, meta=meta_cpy)
        )

        # Graph and JSON     
        landmark_type = decompose_fname[6]
        landmark_id = decompose_fname[7]
        landmark = GetLandmark(graph=graph, landmark_type=landmark_type, landmark_id=landmark_id)

        relative_position = tuple(ldmrk_pos - p_pos for ldmrk_pos, p_pos in zip(landmark.pos, patch_pos[0]))

        logger.debug(f"Relative position {relative_position} in path {patch_pos} for absolute position {landmark.pos}")

        with open(os.path.join(json_out_path, f"{basename}_eedt.json"), "w") as json_file:
            json_data = {
                "global_thickness": eedt_map[landmark.pos],
                "local_thickness": eedt_map_patch[relative_position],
            }
            json.dump(convert_typing_to_native(json_data), json_file, indent=4)


def main(in_path_attr, in_path_ytrue, in_path_graphs, out_path, threads_count=1):
    # I/O
    attribution_id = get_attribution_id(in_path_attr)
    out_path = create_output_dirs(
        [
            os.path.join(out_path, "thickness"),
            os.path.join(out_path, "thickness", "patch"),
            os.path.join(out_path, "thickness", "json"),
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
                        target=task_compute_relative_thickness,
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
            "filename": f"{cfg.log_dir}/analyse_tubularity.log",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    for handler in logger.handlers:
        if type(handler) == logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    main(args.attributions_dir, args.ytrue_dir, args.graphs_dir, cfg.result_dir, args.thread)