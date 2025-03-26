import argparse
import logging
import logging.config # Mandatory if MONAI is not imported

import os
import json
from multiprocessing import Queue, Process, cpu_count, current_process
from queue import Empty

from copy import deepcopy

from tqdm import tqdm

import numpy as np
import monai
from monai.transforms import Compose, LoadImage, SaveImage, SpatialCrop, SpatialPad, AsDiscrete, ToTensor, ToNumpy
from skimage.measure import label

from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2ImageGraph
from utils.load_hyperparameters import load_hyperparameters as LoadHyperparameters
from image.vessel_thickness import distance_map
from utils.load_patch_position import read_path_position_from_file
from utils.create_output_dirs import create_output_dirs
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
        "--hyperparameters",
        type=str,
        metavar=("HYPERPARAMETERS_PATH"),
        default="./resources/default_hyperparameters.json",
        help="Path to the hyperparameters file (*.json)",
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


class CProcessThicknessInput:
    def __init__(self, sample_name, in_dir_ytrue, in_dir_graphs, patch_size, output_dir_json, image_saver):

        self.y_true, self.meta = self._read_and_prepare_image(sample_name, in_dir_ytrue)
        self.graph = self._read_and_prepare_graph(sample_name, in_dir_graphs)

        self.output_dir_json = output_dir_json
        self.image_saver = image_saver

        self.patch_size = patch_size


    def _read_and_prepare_image(self, sample_name, in_dir_ytrue):
        image_loader = LoadImage(ensure_channel_first=True, image_only=False)

        pre_T = Compose([
            AsDiscrete(threshold=0.5),  # Ensure the ground-truth is binary
            ToNumpy(),
        ])

        y_true, meta = image_loader(os.path.join(in_dir_ytrue, f"{sample_name}.nii.gz"))
        y_true = pre_T(y_true) # Save y_true as a np.array as Tensor raise WARNING and reach weird behavior when put in the Queue 

        return y_true, meta


    def _read_and_prepare_graph(self, sample_name, in_dir_graphs):
        graph = LoadVesselGraph(
            os.path.join(
                in_dir_graphs, 
                f"{sample_name}_graph.vvg"
            )
        )
        graph = Anatomic2ImageGraph(graph, self.meta["original_affine"])

        return graph


    def perform_global_eedt(self):
        prep_img = np.squeeze(self.y_true, axis=0)
        self.y_true_eedt = distance_map(prep_img)
        self.image_saver(
            monai.data.MetaTensor(self.y_true_eedt, meta=self.meta)
        )


    def _update_metadata(self, filename, patch_pos):
        meta_cpy = deepcopy(self.meta) # I don't think this is necessary as every Process as it's own copy of the obj, but nevermind
        # Update the filename
        meta_cpy["filename_or_obj"] = f"{filename}.nii.gz"

        # Update the affine
        image_space_new_origin = np.array(patch_pos[0])
        world_space_new_origin = np.matmul(self.meta["affine"][:-1, :-1], np.atleast_2d(image_space_new_origin).T)
        meta_cpy["affine"][:-1,-1:] = self.meta["affine"][:-1,-1:] + world_space_new_origin # [., -1:] to keep 2D dimension of the returned array

        return meta_cpy


    def save_results(self, patch_eedt, patch_pos, input_file):
        struct_fname = Filename(input_file)

        patch_meta = self._update_metadata(struct_fname.get_prefix(), patch_pos)

        self.image_saver(
            monai.data.MetaTensor(patch_eedt, meta=patch_meta)
        )

        # Graph and JSON     
        landmark = GetLandmark(graph=self.graph, landmark_type=struct_fname.landmark_type, landmark_id=struct_fname.landmark_id)

        relative_position = tuple(ldmrk_pos - p_pos for ldmrk_pos, p_pos in zip(landmark.pos, patch_pos[0]))

        logger.debug(f"Relative position {relative_position} in patch {patch_pos} for absolute position {landmark.pos}")

        with open(os.path.join(self.output_dir_json, f"{struct_fname.get_prefix()}_eedt.json"), "w") as json_file:
            json_data = {
                "global_thickness": self.y_true_eedt[landmark.pos],
                "local_thickness": patch_eedt[relative_position],
            }
            json.dump(convert_typing_to_native(json_data), json_file, indent=4)


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

    # Crop the volume to patch at roi position, and pad the patch if patch size is inferior to input size.
    pipeline_T = Compose(
        [
            ToTensor(), # Remember that obj.y_true is a np.array as Tensor raise WARNING and reach weird behavior when put in the Queue
            SpatialCrop(roi_start=start_roi, roi_end=end_roi),
            SpatialPad(spatial_size=patch_size),
        ]
    )

    # Not supposed to be batched but more safe
    I = pipeline_T(I)

    return I


def pbar_listener(queue, ntasks):
    pbar = tqdm(total=ntasks)
    for _ in iter(queue.get, None):
        pbar.update()


def create_vessel_thickness_json(file, input_obj):
    patch_pos = read_path_position_from_file(file)

    patch_ytrue = np.squeeze(extract_patch(input_obj.y_true, patch_pos, input_obj.patch_size).numpy(), axis=0)

    eedt_map_patch = distance_map(patch_ytrue)

    # Change metadata
    input_obj.save_results(eedt_map_patch, patch_pos, file)


def task_create_vessel_thickness_json(obj_queue, obj_finished_queue, proc_input_obj):
    while True:
        try:
            task_file = obj_queue.get()
            
            if task_file is None:
                raise Empty

        except Empty:
            break

        else:
            # No exception raised, process the job and add the task completion to finished_queue
            create_vessel_thickness_json(task_file, proc_input_obj)
            obj_finished_queue.put(task_file)
    
    return True


def distribute_vessel_thickness_json_creation(files, in_dir_attribution, proc_input_obj, process_count):
    nfiles = len(files)

    process_count = min(min(process_count, cpu_count()), nfiles)

    obj_to_process = Queue()
    obj_finished = Queue()

    processes = []

    for file in files:
        obj_to_process.put(os.path.join(in_dir_attribution, file))

    # Start the progress bar process
    proc = Process(target=pbar_listener, args=(obj_finished, nfiles))
    proc.start()

    # creating processes
    for k in range(process_count):
        obj_to_process.put(None) # Stop condition
        processes.append(
            Process(target=task_create_vessel_thickness_json, args=(obj_to_process, obj_finished, proc_input_obj))
        )
        processes[-1].start()

    # completing process
    for p in processes:
        
        p.join()

    obj_finished.put(None)
    proc.join()


def main(in_dir_attribution, in_dir_ytrue, in_dir_graphs, hyperparameters_path, output_dir, process_count=None):
    attribution_id = get_attribution_id(in_dir_attribution)

    hyperparameters = LoadHyperparameters(hyperparameters_path)

    is_patch        = hyperparameters["patch"]
    if not is_patch:
        raise ValueError(f"Relative thickness can only be set for patched data. Patched data ? {is_patch}")
    input_shape     = hyperparameters["input_shape"]

    if output_dir is None:
        output_dir = cfg.result_dir

    out_dir_skeleton, out_dir_json = create_output_dirs(
        [
            os.path.join(output_dir, "thickness", "eedt"),
            os.path.join(output_dir, "thickness", "json"),
        ],
        attribution_id
    )

    image_saver = SaveImage(
        output_dir=out_dir_skeleton,
        output_ext=".nii.gz",
        output_postfix="eedt_label",
        output_dtype=np.float32,
        resample=False,
        separate_folder=False,
    )

    files = [f for f in os.listdir(in_dir_attribution) if f.endswith(".txt")]

    unique_samples = { "_".join(file.split('_')[:2]) for file in files }

    for sample in unique_samples:
        logger.info(f"Process sample {sample}")

        input_obj = CProcessThicknessInput(sample, in_dir_ytrue, in_dir_graphs, input_shape, out_dir_json, image_saver) # TODO: patch size is hardcoded, we need to read the config instead
        input_obj.perform_global_eedt()

        sample_files = [f for f in files if f.startswith(sample)]

        distribute_vessel_thickness_json_creation(sample_files, in_dir_attribution, input_obj, process_count)


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

    main(args.attributions_dir, args.ytrue_dir, args.graphs_dir, args.hyperparameters, args.output, args.thread)