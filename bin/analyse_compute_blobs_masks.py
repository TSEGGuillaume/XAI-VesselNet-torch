import argparse
import logging

import os
import json
from multiprocessing import Process

from tqdm import tqdm

import numpy as np
import monai
from monai.transforms import LoadImage, SaveImage

from skimage.morphology import remove_small_objects

from image.blobs import detect_bright_and_dark_blobs, compute_blobs_properties

from metrics.descriptive_statistics import univariate_analysis
from metrics.total_variation import image_total_variation as TotalVariation
from utils.create_output_dirs import create_output_dirs
from utils.json_format import convert_typing_to_native

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


def task_compute_blobs_mask(thread_idx, files, in_path, out_path):

    for file in tqdm(files):

        fpath = os.path.join(in_path, file)
        I, meta = LoadImage(ensure_channel_first=False, image_only=False)(fpath)

        blobs_mask = detect_bright_and_dark_blobs(I.numpy(), sigma_max=10, N_sigma=8, threshold="otsu").astype(np.uint8)

        f_basename = file.split(".")[0]
        nib.save(nib.Nifti1Image(blobs_mask, meta["original_affine"]), os.path.join(out_path, f"{f_basename}_blobs.nii.gz"))

    logger.info(f"Thread {thread_idx}: finished")


def task_compute_blobs_properties(thread_idx, files, in_path, out_path):

    attribution_id = os.path.normpath(in_path).split(os.sep)[-1]
    output_dir = os.path.join(out_path, "label", attribution_id)

    image_saver = SaveImage(
        output_dir=output_dir,
        output_ext=".nii.gz",
        output_postfix="label",
        resample=False,
        separate_folder=False,
        output_dtype=np.uint8,
    )

    for file in tqdm(files):

        if not os.path.exists(os.path.join(output_dir, file)):

            fpath = os.path.join(in_path, file)
            blobs_mask, meta = LoadImage(ensure_channel_first=False, image_only=False)(fpath)

            blobs_mask = remove_small_objects(
                (blobs_mask>0).numpy(), min_size=14
            )  # For regionprops that requiere convex hull, we remove objects smaller that (3^3)/2

            selected_props = [
                "label",
                "area",
                "centroid",
                "equivalent_diameter_area",
                "feret_diameter_max",
            ]

            blobs_mask_props, _, _ = compute_blobs_properties(
                blobs_mask, selected_props, is_labeled=True
            ) # Get the props for all blobs together (without labelization) ; with is_labeled==True, labelization not performed

            blobs_props, labeled_blobs, nblobs = compute_blobs_properties(
                blobs_mask, selected_props, is_labeled=False
            )

            image_saver(labeled_blobs, meta)

        else:
            # The labelized image already exists, just load it
            fpath = os.path.join(output_dir, file)

            blobs_mask, meta = LoadImage(ensure_channel_first=False, image_only=False)(fpath)
            selected_props = [
                "label",
                "area",
                "centroid",
                "equivalent_diameter_area",
                "feret_diameter_max",
            ]

            blobs_props, _, nblobs = compute_blobs_properties(
                blobs_mask, selected_props, is_labeled=True
            )

        file_basename = file.split(".")[0]

        out_data = {
            "blobs_count": nblobs,
            "blobs_mask" : os.path.join(output_dir, f"{file_basename}_{image_saver.folder_layout.postfix}.nii.gz"),
            "blobs_mask_props": blobs_mask_props,
            "blobs": blobs_props,
        }

        with open(os.path.join(out_path, "json", attribution_id, f"{file_basename}.json"), "w") as out_file:
            json.dump(convert_typing_to_native(out_data), out_file, indent=4)
     
    logger.info(f"Thread {thread_idx}: finished")


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


def task_compute_blobs_and_properties(thread_idx, files, in_path, out_path):

    out_path_mask, out_path_label, out_path_json = out_path

    image_mask_saver = SaveImage(
        output_dir=out_path_mask,
        output_ext=".nii.gz",
        output_postfix="blobs",
        resample=False,
        separate_folder=False,
        output_dtype=np.uint8,
    )

    image_label_saver = SaveImage(
        output_dir=out_path_label,
        output_ext=".nii.gz",
        output_postfix="blobs_label",
        resample=False,
        separate_folder=False,
        output_dtype=np.uint8,
    )

    image_loader = LoadImage(ensure_channel_first=False, image_only=False)

    for file in tqdm(files):
        fpath = os.path.join(in_path, file)

        I_attr, meta = image_loader(fpath)

        blobs_mask = detect_bright_and_dark_blobs(I_attr.numpy(), sigma_max=10, N_sigma=8, threshold="otsu").astype(np.uint8)

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
            tv_background = TotalVariation(I_attr, neighborhood="N26", norm="L1", mask=mask_background)

            blobs_props["blobs_mask_props"][label_background]["stats"] = stats_background 
            blobs_props["blobs_mask_props"][label_background]["stats"]["total_variation"] = tv_background

            # \!background --> all_blobs_mask
            label_blobs = 1
            mask_blobs = blobs_mask >= label_blobs # Useless operation, but for clarity ; blobs_mask is already equal to (blobs_mask>=blobs_label) except for type
            stats_blobs = univariate_analysis(I_attr[mask_blobs].flatten())
            tv_blobs = TotalVariation(I_attr, neighborhood="N26", norm="L1", mask=mask_blobs)

            blobs_props["blobs_mask_props"][label_blobs]["stats"] = stats_blobs 
            blobs_props["blobs_mask_props"][label_blobs]["stats"]["total_variation"] = tv_blobs

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
                    tv_blobs = TotalVariation(I_attr, neighborhood="N26", norm="L1", mask=mask_current_blobs)

                    stats_blob["area_check"] = np.sum(mask_current_blobs)

                    blob_idx = blob_lbl - 1
                    blobs_props["blobs_props"][blob_idx]["stats"] = stats_blob
                    blobs_props["blobs_props"][blob_idx]["stats"]["total_variation"] = tv_blobs

        else:
            blobs_props["blobs_mask_props"] = []
            blobs_props["blobs_props"] = []

        f_basename = "{}".format(file.split(".")[0])
        blobs_props["blobs_mask"] = os.path.join(
            out_path_mask,
            f"{f_basename}_{image_label_saver.folder_layout.postfix}.nii.gz"
        )

        image_mask_saver(monai.data.MetaTensor(blobs_mask, meta=meta))
        image_label_saver(monai.data.MetaTensor(blobs_labeled, meta=meta))

        with open(os.path.join(out_path_json, f"{f_basename}_blobs.json"), "w") as out_file:
            json.dump(convert_typing_to_native(blobs_props), out_file, indent=4)
     
    logger.info(f"Thread {thread_idx}: finished")


def main(in_attributions_dir, out_path, threads_count):
    # I/O   
    if out_path is None:
        out_path = os.path.join(cfg.result_dir, "blobs")

    attribution_id = os.path.normpath(in_attributions_dir).split(os.sep)[-1]

    out_dirs = create_output_dirs(
        [
            out_path,
            os.path.join(out_path, "label"),
            os.path.join(out_path, "json")
        ],
        attribution_id
    )

    # List attribution files, for which no blobs mask already exists
    files = [f for f in os.listdir(in_attributions_dir)
                if f.endswith(".nii.gz")
                and not (
                    # os.path.isfile(os.path.join(out_path, attribution_id, "{}_blobs.nii.gz".format(f.split(".")[0])))
                    # or 
                    os.path.isfile(os.path.join(out_path, "json", attribution_id, "{}_blobs.json".format(f.split(".")[0])))
                )
            ]
    logger.info(f"Files count : {len(files)}")
    nfiles = len(files)

    if nfiles > 0:

        if nfiles < threads_count:
            threads_count = nfiles
        
        files_chunks = np.array_split(files, threads_count)

        threads = []
        for id_chunk, chunk in enumerate(files_chunks):
            chunk = chunk.tolist()

            logger.debug(f"Chunk {id_chunk} : {len(chunk)} files.")

            threads.append(
                #Process(target=task_compute_blobs_mask, args=(id_chunk, chunk, attributions_path, out_path))
                Process(target=task_compute_blobs_and_properties, args=(id_chunk, chunk, in_attributions_dir, out_dirs))
            )
            threads[-1].start()
           
        for thread in threads:
            thread.join()

    else:
        raise RuntimeError("The given directory is empty !")


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

    # Shut the mouth of console handler and LoadImage logger
    for handler in logger.handlers:
        if type(handler) == logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    logging.getLogger('LoadImage').setLevel(logging.ERROR)

    main(args.attributions_dir, args.output, args.thread)