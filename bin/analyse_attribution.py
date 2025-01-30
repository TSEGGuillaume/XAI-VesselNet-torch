import argparse
import logging
import logging.config # Mandatory if MONAI is not imported

import os
import json
from tqdm import tqdm

from multiprocessing import Process

import numpy as np

import utils.configuration as AppCfg
from utils.json_format import convert_typing_to_native
from utils.create_output_dirs import create_output_dir


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


def task_loop_analyse_attribution(thread_id, files, input_dir, output_dir):

    attribution_id = get_attribution_id(input_dir)

    input_json_dirs = {
        "landmark":   os.path.join(cfg.result_dir, "json", "landmark", attribution_id),
        "tubularity":   os.path.join(cfg.result_dir, "json", "tubularity", attribution_id),
        "connectivity": os.path.join(cfg.result_dir, "json", "connectivity", attribution_id),
        "thickness":    os.path.join(cfg.result_dir, "json", "thickness", attribution_id),
        "patch":        os.path.join(cfg.result_dir, "json", "patch", attribution_id),
        "stats":        os.path.join(cfg.result_dir, "json", "stats", attribution_id),
        "blobs":        os.path.join(cfg.result_dir, "json", "blobs", attribution_id),
    }

    logger.info("Checking the file environment:")

    for k_path, v_path in input_json_dirs.items():
        path_ok = os.path.exists(v_path)
        
        logger.info(f"\t* {k_path}: {v_path} --> {path_ok}")

        if not path_ok:
            raise EnvironmentError(f"Environment error: {v_path} directory is missing")

    for file in tqdm(files):
        # Get usefull information from the attribution map file name

        # [DATASET]_[DATAID]_[TRAINING_STRATEGY]_model_[MODELID]_[XAIMETHOD]_[LANDMARKTYPE]_[LANDMARKID]_[PATCHID]_ochan[OUTPUTCHANNELID]_ichan[INPUTCHANNELID].nii.gz
        # 3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_ochan0_ichan0.nii.gz
        f_basename = os.path.basename(file).split(".")[0]
        decompose_fname = f_basename.split("_")

        fname_prefix = "_".join(decompose_fname[:-2]) # delete _ochanX_ichanY 

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
        with open(os.path.join(input_json_dirs["stats"], f"{f_basename}_stats.json"), "r") as json_file:
            stats_data = json.load(json_file)

        with open(os.path.join(input_json_dirs["blobs"], f"{f_basename}_blobs.json"), "r") as json_file:
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
        
        with open(os.path.join(output_dir, f"res_{f_basename}.json"), "w") as json_file:
            json.dump(convert_typing_to_native(out_dict), json_file, indent=4)

    logger.info(f"Thread {thread_id} finished. Result file saved at {output_dir}")


def main(attribution_dir, output_dir=None, threads_count=1):
    # Set output
    attribution_id = os.path.normpath(attribution_dir).split(os.sep)[-1]

    if output_dir == None:
        output_dir = cfg.result_dir    
    output_dir = create_output_dir(os.path.join(output_dir, "attributions", "json"), attribution_id)

    logger.info(f"Output directory: {output_dir}")

    # Set inputs
    files_attr = [
        os.path.join(attribution_dir, f) for f in os.listdir(attribution_dir)
        if f.endswith(".nii.gz") and
        not os.path.isfile(os.path.join(output_dir, "res_{}.json".format(os.path.basename(f).split(".")[0])))
    ]

    nfiles = len(files_attr)

    if nfiles > 0:

        if nfiles < threads_count:
            threads_count = nfiles
        
        files_attr_chunks = np.array_split(files_attr, threads_count)

        threads = []
        for id_chunk, chunk in enumerate(files_attr_chunks):
            chunk = chunk.tolist()

            logger.debug(f"Chunk {id_chunk} : {len(chunk)} files.")

            threads.append(
                Process(target=task_loop_analyse_attribution, args=(id_chunk, chunk, attribution_dir, output_dir))
            )
            threads[-1].start()
           
        for thread in threads:
            thread.join()
           
    else:
        logger.info("No files found")

    logger.info("The job is over, my Lord")


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