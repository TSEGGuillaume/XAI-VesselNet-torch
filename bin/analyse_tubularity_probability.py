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


def get_attribution_id(in_path):
    return os.path.normpath(in_path).split(os.sep)[-1]


def task_compute_tubularity_prob(files, in_path_filters, in_path_graphs, out_path):

    table_filters_id_name = {
        "0000": "scan",
        "0001": "frangi",
        "0002": "jerman",
        "0003": "sato",
        "0004": "zhang",
        "0005": "meijering",
        "0006": "rorpo",
    }

    graphs = {} # Will contain sample_id: graph_data ; graphs are common, whatever the filter used.

    for id_filter, name_filter in table_filters_id_name.items():

        Is = {} # Will contain sample_id: image_filter_data

        logger.info(f"Processing filter {name_filter}")

        for file in tqdm(files):

            decomp_fname = file.split("_")

            prefix_fname = "_".join(decomp_fname[:-2])
            sample_id = "_".join(decomp_fname[:2])
            landmark_type = decomp_fname[6]
            landmark_id = decomp_fname[7]

            if sample_id in Is.keys() and sample_id in graphs.keys():
                I = Is[sample_id]
                graph = graphs[sample_id]
            else: # First time this sample is processed. Add to the dict.
                I, meta = LoadImage(ensure_channel_first=False, image_only=False)(os.path.join(in_path_filters, f"{sample_id}_{id_filter}.nii.gz"))

                graph = LoadVesselGraph(os.path.join(in_path_graphs, f"{sample_id}_graph.vvg"))
                graph = Anatomic2ImageGraph(graph, meta["original_affine"])

                Is[sample_id] = I
                graphs[sample_id] = graph

            landmark = GetLandmark(graph=graph, landmark_type=landmark_type, landmark_id=landmark_id)

            vesselness_value = I[landmark.pos]

            json_path = os.path.join(out_path, "{}_tubularity.json".format(prefix_fname))
            
            if os.path.isfile(json_path):
                # UPDATE
                with open(json_path, "r") as json_file:
                    json_data = json.load(json_file)
                    json_data.update({name_filter:vesselness_value})
    
                with open(json_path, "w") as json_file:
                    json.dump(convert_typing_to_native(json_data), json_file, indent=4)
            
            else:
                # FIRST TIME ; CREATE
                with open(json_path, "w") as json_file:
                    json.dump(convert_typing_to_native({name_filter:vesselness_value}), json_file, indent=4)
    

def main(in_path_attr, in_path_filters, in_path_graphs, out_path, threads_count=1):
    # I/O
    attribution_id = get_attribution_id(in_path_attr)
    out_path = create_output_dir(
        os.path.join(out_path, "tubularity"),
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
                Process(target=task_compute_tubularity_prob, args=(chunk, in_path_filters, in_path_graphs, out_path))
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

    main(args.attributions_dir, args.filters_dir, args.graphs_dir, cfg.result_dir, args.thread)