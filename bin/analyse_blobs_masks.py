import argparse
import logging

import os
import json
from multiprocessing import Queue, Process, cpu_count
from queue import Empty

from tqdm import tqdm

import numpy as np
import monai
from monai.transforms import LoadImage, SaveImage

from skimage.morphology import remove_small_objects

from image.blobs import detect_bright_and_dark_blobs, compute_blobs_properties

from metrics.descriptive_statistics import univariate_analysis
from metrics.total_variation import image_total_variation as TotalVariation
from metrics.norm import compute_norm
from metrics.fisher import compute_fisher_contrast_noise_ratio as Fisher
from utils.create_output_dirs import create_output_dirs
from utils.json_format import convert_typing_to_native
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


def subtask_compute_blobs_properties(blobs_mask):

    blobs_mask = remove_small_objects(
        blobs_mask>0, min_size=14
    )  # For regionprops that requiere convex hull, we remove objects smaller that (3^3)/2

    selected_props = [
        "label",
        "area",
        "centroid",
        "equivalent_diameter_area",
        "feret_diameter_max",
    ]

    blobs_mask_props, _, _ = compute_blobs_properties(
        blobs_mask, selected_props, is_labeled=True, include_background=True
    ) # Get the props for all blobs together (without labelization) ; with is_labeled==True, labelization not performed

    blobs_props, labeled_blobs, nblobs = compute_blobs_properties(
        blobs_mask, selected_props, is_labeled=False, include_background=False
    )

    out_data = {
        "blobs_count": nblobs,
        "blobs_mask_props": blobs_mask_props,
        "blobs_props": blobs_props,
    }

    return out_data, labeled_blobs, nblobs


def create_blobs_json(file, in_dir_attribution, image_loader, image_saver, out_dir_json):

    fpath = os.path.join(in_dir_attribution, file)

    I_attr, meta = image_loader(fpath)

    blobs_mask = detect_bright_and_dark_blobs(I_attr.numpy(), sigma_min = 1, sigma_max=3, N_sigma=5, threshold="otsu")
    blobs_mask = blobs_mask.astype(np.uint8)

    blobs_props, blobs_labeled, nblobs = subtask_compute_blobs_properties(blobs_mask)

    if nblobs != 0:
        I_attr = I_attr.numpy()

        # Computes features for the background and !background
        # ----------------------------------------------------
        # 
        # Background is the label 0
        label_background = 0
        mask_background = blobs_mask == label_background
        stats_background = univariate_analysis(I_attr[mask_background].flatten())
        #tv_background = TotalVariation(I_attr, neighborhood="N26", norm="L1", mask=mask_background)
        norm_background = compute_norm(I_attr[mask_background].flatten())

        blobs_props["blobs_mask_props"][label_background]["stats"] = stats_background
        blobs_props["blobs_mask_props"][label_background]["stats"]["norm"] = norm_background
        #blobs_props["blobs_mask_props"][label_background]["stats"]["total_variation"] = tv_background

        # \!background --> all_blobs_mask
        label_blobs = 1
        mask_blobs = blobs_mask >= label_blobs # Useless operation, but for clarity ; blobs_mask is already equal to (blobs_mask>=blobs_label) except for type
        stats_blobs = univariate_analysis(I_attr[mask_blobs].flatten())
        #tv_blobs = TotalVariation(I_attr, neighborhood="N26", norm="L1", mask=mask_blobs)
        norm_blobs = compute_norm(I_attr[mask_blobs].flatten())
        fisher_cnr_abs = Fisher(I_attr[mask_blobs].flatten(), I_attr[mask_background].flatten(), abs=True)
        fisher_cnr = Fisher(I_attr[mask_blobs].flatten(), I_attr[mask_background].flatten(), abs=False)

        blobs_props["blobs_mask_props"][label_blobs]["stats"] = stats_blobs 
        blobs_props["blobs_mask_props"][label_blobs]["stats"]["norm"] = norm_blobs
        blobs_props["blobs_mask_props"][label_blobs]["stats"]["fisher_CNR_abs"] = fisher_cnr_abs
        blobs_props["blobs_mask_props"][label_blobs]["stats"]["fisher_CNR"] = fisher_cnr
        #blobs_props["blobs_mask_props"][label_blobs]["stats"]["total_variation"] = tv_blobs

        # Now we have all we need for background and "all_blobs_mask"
        # Hence, if only one blob, we can skip the loop, as all_blobs_mask == blob
        if nblobs == 1:
            # Duplicate blobs_props["blobs_mask_props"][label_blobs] ; label_blobs-1 as background is not included in blobs_props
            blobs_props["blobs_props"][label_blobs-1] = blobs_props["blobs_mask_props"][label_blobs]

        else: # i.e. nblobs > 1
            # Loop over blobs, starting at 1 -> we already have 0 above
            for blob_lbl in range(1, nblobs+1):
                mask_current_blobs = blobs_labeled == blob_lbl
                masked_I_attr = I_attr[mask_current_blobs]

                stats_blob = univariate_analysis(masked_I_attr.flatten())
                #tv_blobs = TotalVariation(I_attr, neighborhood="N26", norm="L1", mask=mask_current_blobs)
                norm_blob = compute_norm(masked_I_attr.flatten())
                fisher_cnr_abs = Fisher(masked_I_attr.flatten(), I_attr[mask_background].flatten(), abs=True)
                fisher_cnr = Fisher(masked_I_attr.flatten(), I_attr[mask_background].flatten(), abs=False)

                stats_blob["area_check"] = np.sum(mask_current_blobs)

                blob_idx = blob_lbl - 1
                blobs_props["blobs_props"][blob_idx]["stats"] = stats_blob
                blobs_props["blobs_props"][blob_idx]["stats"]["norm"] = norm_blob
                blobs_props["blobs_props"][blob_idx]["stats"]["fisher_CNR_abs"] = fisher_cnr_abs
                blobs_props["blobs_props"][blob_idx]["stats"]["fisher_CNR"] = fisher_cnr
                #blobs_props["blobs_props"][blob_idx]["stats"]["total_variation"] = tv_blobs

    else:
        blobs_props["blobs_mask_props"] = []
        blobs_props["blobs_props"] = []

    
    struct_fname = Filename(file)

    blobs_props["blobs_mask"] = os.path.join(
        image_saver.folder_layout.output_dir,
        f"{struct_fname.get_filename()}_{image_saver.folder_layout.postfix}.nii.gz"
    )

    # image_saver(monai.data.MetaTensor(blobs_mask, meta=meta))
    image_saver.folder_layout.postfix = "blobs_label"
    image_saver(monai.data.MetaTensor(blobs_labeled, meta=meta))

    with open(os.path.join(out_dir_json, f"{struct_fname.get_filename()}_blobs.json"), "w") as out_file:
        json.dump(convert_typing_to_native(blobs_props), out_file, indent=4)


def task_create_blobs_json(files_queue, files_finished_queue, in_dir_attribution, image_loader, image_saver, out_dir_json):
    while True:
        try:
            task_file = files_queue.get()
            
            if task_file is None:
                raise Empty

        except Empty:
            break

        else:
            # No exception raised, process the job and add the task completion to finished_queue
            create_blobs_json(task_file, in_dir_attribution, image_loader, image_saver, out_dir_json)
            files_finished_queue.put(task_file)
    
    return True


def distribute_blobs_json_creation(files, in_dir_attribution, image_loader, image_saver, out_dir_json, process_count):
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
            Process(target=task_create_blobs_json, args=(files_to_process, files_finished, in_dir_attribution, image_loader, image_saver, out_dir_json))
        )
        processes[-1].start()

    # completing process
    for p in processes:
        p.join()

    files_finished.put(None)
    proc.join()


def main(in_dir_attribution, output_dir, process_count=None):
    attribution_id = get_attribution_id(in_dir_attribution)

    if output_dir is None:
        output_dir = cfg.result_dir

    out_dir_mask, out_dir_json = create_output_dirs(
        [
            os.path.join(output_dir, "blobs", "mask"),
            os.path.join(output_dir, "blobs", "json")
        ],
        attribution_id
    )

    image_saver = SaveImage(
        output_dir=out_dir_mask,
        output_ext=".nii.gz",
        #output_postfix="blobs_label",
        output_dtype=np.float32,
        resample=False,
        separate_folder=False,
    )

    image_loader = LoadImage(ensure_channel_first=False, image_only=False)

    files = [f for f in os.listdir(in_dir_attribution) if f.endswith(".nii.gz")]

    distribute_blobs_json_creation(files, in_dir_attribution, image_loader, image_saver, out_dir_json, process_count)


if __name__ == "__main__":
    import utils.configuration as appcfg

    cfg = appcfg.Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    # Quick and dirty
    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": f"{cfg.log_dir}/blobs_search.log",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    main(args.attributions_dir, args.output, args.thread)