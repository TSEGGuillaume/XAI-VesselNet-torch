import argparse
import logging

import os
import json

import numpy as np
import torch
from torch.nn import Module
from torch import device

from monai.data.meta_tensor import MetaTensor
from monai.transforms import (
    LoadImage,
    SaveImage,
    Compose,
    SpatialCrop,
    SpatialPad,
    Activations,
    AsDiscrete,
    ToTensor,
    ToNumpy,
)
from monai.networks.utils import one_hot as OneHotEncoding
from monai.data.utils import decollate_batch

from skimage.morphology import remove_small_objects

from utils.load_hyperparameters import load_hyperparameters
from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2ImageGraph
from utils.get_landmark_from_args import get_landmark_obj
from utils.load_patch_position import read_path_position_from_file
from network.model_creator import init_inference_model
from utils.prebuilt_logs import log_hardware
from image.vessel_thickness import compute_vessel_thickness
from image.blobs import detect_blob, compute_blobs_properties
from utils.distances import distance
from metrics.total_variation import image_total_variation
from metrics.descriptive_statistics import univariate_analysis

from infer import infer_patch
from eval import evaluate


def parse_arguments():
    import models.instanciate_model

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "attribution",
        type=str,
        metavar=("ATTRIBUTION_PATH"),
        help="Path to the attribution (*.nii, *.nii.gz)",
    )
    parser.add_argument(
        "graph",
        type=str,
        metavar=("GRAPH_PATH"),
        help="Path to the graph (*.vvg)",
    )

    parser.add_argument(
        "patch",
        type=str,
        metavar=("PATCH_POSITION_PATH"),
        help="Path to the patch position file (*.txt)",
    )

    parser.add_argument(
        "model",
        type=str,
        metavar=("MODEL"),
        choices=models.instanciate_model._all_models,
        default=models.instanciate_model._all_models[0],
        help="Name of the model",
    )

    parser.add_argument(
        "weights",
        type=str,
        metavar=("WEIGHTS_PATH"),
        help="Path to the model's weights",
    )

    parser.add_argument(
        "image",
        type=str,
        metavar=("IMAGE_PATH"),
        help="Path to the image",
    )

    parser.add_argument(
        "reference",
        type=str,
        metavar=("REFERENCE_PATH"),
        help="Path to the ground-truth",
    )

    parser.add_argument(
        "--hyperparameters",
        type=str,
        metavar=("HYPERPARAM_PATH"),
        default="./resources/default_hyperparameters.json",
        help="Path to the hyperparameters file (*.json)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--node",
        "-n",
        type=int,
        metavar=("ID_NODE"),
        help="id of the node to inspect",
        default=None,
    )
    group.add_argument(
        "--centerline",
        "-c",
        type=int,
        metavar=("ID_CENTERLINE"),
        help="id of the centerline to inspect",
        default=None,
    )
    group.add_argument(
        "--position",
        "-p",
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        help="image coordinates (x,y,z) of the voxel to inspect",
        default=None,
    )

    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args


def eval_patch_prediction(
    model: Module,
    x: MetaTensor,
    y: MetaTensor,
    patch_pos: tuple,
    input_size: tuple,
    device: device,
) -> tuple[dict, MetaTensor, MetaTensor]:
    """
    Extract a patch from the whole data, predict the patch and evaluate the resulting segmentation.

    Args:
        model : The trained model
        x : The data MetaTensors (N,C,H,W,[D])
        y : The ground-truth MetaTensors (N,C,H,W,[D])
        patch_pos : The ROI of the patch [ (start, ), (end, ) ]
        input_size : The regular patch size. This is used for padding if cropping results in a smaller size.
        device : The device to store the model and data

    Returns:
        The tuple (metrics, y_pred, y_true).
    """
    # Pre-processing
    start_roi, end_roi = patch_pos

    y_pred = infer_patch(
        model=model,
        data=x,
        device=device,
        patch_pos=patch_pos,
        input_size=input_size,
        postprocessing=None,
    )
    y_pred = y_pred[0]  # We already know there is only one image

    output_channels = y_pred.shape[1]  # No more batch dimension

    pred_postprocess_T = Compose(
        [
            # include saver here
            (
                Activations(sigmoid=True)
                if output_channels == 1
                else Activations(softmax=True)
            ),
            AsDiscrete(threshold=0.5),
        ]
    )
    y_pred_postprocess = torch.stack(
        [pred_postprocess_T(i) for i in decollate_batch(y_pred)]
    )

    # We process the ground-truth in the same way than the data, i.e. in patch
    reference_process_T = Compose(
        [
            # Crop the volume to patch at roi position, and pad the patch if patch size is inferior to input size.
            SpatialCrop(roi_start=start_roi, roi_end=end_roi),
            SpatialPad(spatial_size=input_size),
        ]
    )

    # 1. Crop and pad to input_size if necessary
    # 2. Transform to One Hot
    # 3. Add the batch dim
    y = OneHotEncoding(torch.stack([reference_process_T(y)]), num_classes=2, dim=1)

    metrics = evaluate(ys_pred=[y_pred_postprocess], ys_true=[y])

    return metrics, y_pred, y


def analyse_prediction(
    model: Module,
    x: MetaTensor,
    y: MetaTensor,
    patch_pos: tuple,
    input_size: tuple,
    device: device,
    relative_landmark_pos: tuple,
) -> tuple[dict, MetaTensor, MetaTensor]:
    """
    Perform a prediction on a patch and evaluate the resulting segmentation. Analyse the output values on a specific landmark position and define its status.

    Args:
        model           : The trained model
        x               : The data MetaTensors (N,C,H,W,[D])
        y               : The ground-truth MetaTensors (N,C,H,W,[D])
        patch_pos       : The ROI of the patch [ (start, ), (end, ) ]
        input_size      : The regular patch size. This is used for padding if cropping results in a smaller size.
        device          : The device to store the model and data
        landmark_pos    : The position of the landmark to analyse

    Returns:
        The tuple (metrics, y_pred, y_true).
    """
    metrics, y_pred, y_true = eval_patch_prediction(
        model, x, y, patch_pos, input_size, device
    )

    # Compute the relative position of the node in the provided patch
    relative_landmark_x = relative_landmark_pos[0]
    relative_landmark_y = relative_landmark_pos[1]
    relative_landmark_z = relative_landmark_pos[2]

    output_channels = y_pred.shape[1]
    act = (
        Activations(sigmoid=True, dim=1)
        if output_channels == 1
        else Activations(softmax=True, dim=1)
    )

    output_values = y_pred[
        0, :, relative_landmark_x, relative_landmark_y, relative_landmark_z
    ]  # Get the raw output values
    working_y_pred = act(y_pred)
    activation_values = working_y_pred[
        0, :, relative_landmark_x, relative_landmark_y, relative_landmark_z
    ]  # Get the activated output values

    logger.info(
        f"Compare prediction ({working_y_pred.shape}), with ground-truth ({y_true.shape})"
    )

    working_y_pred = AsDiscrete(threshold=0.5)(working_y_pred)

    #  Get the status of the point
    if output_channels == 1:
        if (
            y_true[:, :, relative_landmark_x, relative_landmark_y, relative_landmark_z]
            == True
        ):
            if (
                working_y_pred[
                    :, :, relative_landmark_x, relative_landmark_y, relative_landmark_z
                ]
                == True
            ):
                point_status = "TP"
            else:
                point_status = "FN"
        else:
            if (
                working_y_pred[
                    :, :, relative_landmark_x, relative_landmark_y, relative_landmark_z
                ]
                == True
            ):
                point_status = "FP"
            else:
                point_status = "TN"

    elif output_channels == 2:
        if (
            y_true[:, 1, relative_landmark_x, relative_landmark_y, relative_landmark_z]
            == True
        ):
            if (
                working_y_pred[
                    :, 1, relative_landmark_x, relative_landmark_y, relative_landmark_z
                ]
                == True
            ):
                point_status = "TP"
            else:
                point_status = "FN"
        else:
            if (
                working_y_pred[
                    :, 1, relative_landmark_x, relative_landmark_y, relative_landmark_z
                ]
                == True
            ):
                point_status = "FP"
            else:
                point_status = "TN"

    else:
        raise RuntimeError("Only binary segmentation allowed.")

    pred_nfo = {
        "point_status": point_status,
        "output_value": output_values,
        "activation_value": activation_values,
        "metrics": metrics,
    }

    return pred_nfo, y_pred, y_true


def convert_typing_to_native(data) -> dict:
    """
    Convert a dictionary containing high-level types into an identical dictionary containing only native types
    
    Args:
        data : The dictionnary to process

    Returns:
        The native types dictionary
    """
    if isinstance(data, dict):
        return {k: convert_typing_to_native(v) for k, v in data.items()}

    elif isinstance(data, tuple):
        return convert_typing_to_native(list(data))

    elif isinstance(data, torch.Tensor):
        return convert_typing_to_native(data.item()) if len(data)==1 else convert_typing_to_native(data.cpu().numpy().tolist())

    elif isinstance(data, np.ndarray):
        return convert_typing_to_native(data.item()) if len(data)==1 else convert_typing_to_native(data.tolist())

    elif isinstance(data, list):
        return [convert_typing_to_native(elem) for elem in data]

    elif isinstance(data, (np.int64, np.int32)):
        return int(data)

    elif isinstance(data, (np.float64, np.float32)):
        return float(data)

    else:
        return data


def main():
    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    # Variables
    attribution_path = args.attribution
    x_path = args.image
    y_true_path = args.reference
    graph_path = args.graph
    patch_pos_path = args.patch
    model_name = args.model
    weights_path = args.weights
    hyperparameters_path = args.hyperparameters

    if args.node is not None:
        landmark_type = "node"
        landmark_id = args.node
    elif args.centerline is not None:
        landmark_type = "centerline"
        landmark_id = args.centerline
    elif args.position is not None:
        landmark_type = "position"
        landmark_id = args.position
    else:
        raise NotImplementedError(
            "You should provide a node, acenterline or a position"
        )

    # Load hyperparameters for training
    hyperparameters = load_hyperparameters(hyperparameters_path)

    in_channels = hyperparameters["in_channels"]
    out_channels = hyperparameters["out_channels"]
    sw_shape = hyperparameters["patch_size"]

    # Load the image and its associated ground-truth
    I_x, meta = LoadImage(image_only=False, ensure_channel_first=True)(
        x_path
    )  # Assume x's metadata as reference
    I_y_true = LoadImage(image_only=True, ensure_channel_first=True)(y_true_path)

    # ------------------------ #
    #   PREDICTIONS ANALYSIS   #
    # ------------------------ #
    logger.info("Prediction analysis ...")
    logger.info("_______________________")

    I_x = I_x.to(device)
    I_y_true = I_y_true.to(device)

    patch_pos = read_path_position_from_file(patch_pos_path)

    # Load the graph and convert to image coordinate system
    vessel_graph = LoadVesselGraph(graph_path)
    vessel_graph = Anatomic2ImageGraph(vessel_graph, meta["original_affine"])

    landmark = get_landmark_obj(
        graph=vessel_graph, landmark_type=landmark_type, landmark_id=landmark_id
    )

    # Compute the relative position of the node in the provided patch
    relative_landmark_pos = (
        landmark.pos[0] - patch_pos[0][0],
        landmark.pos[1] - patch_pos[0][1],
        landmark.pos[2] - patch_pos[0][2],
    )

    model = init_inference_model(
        model_name=model_name,
        weights_path=weights_path,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
    )
    pred_res, y_pred_patch, y_true_patch = analyse_prediction(
        model=model,
        x=I_x,
        y=I_y_true,
        patch_pos=patch_pos,
        input_size=sw_shape,
        device=device,
        relative_landmark_pos=relative_landmark_pos,
    )

    # ------------------------ #
    #      OBJECT ANALYSIS     #
    # ------------------------ #
    logger.info("Vessel object analysis ...")
    logger.info("__________________________")

    transform_obj_analysis = Compose(
        [
            AsDiscrete(threshold=0.5),  # Ensure the ground-truth is binary
            ToTensor(),
        ]
    )

    if out_channels == 2:
        y_true_patch = y_true_patch[:, 1, :]
    elif out_channels > 2:
        raise ValueError("Non-binary models are not allowed")

    logger.info("Compute global vessel size...")
    vessel_thickness_global = compute_vessel_thickness(
        torch.squeeze(transform_obj_analysis(I_y_true)), landmark.pos
    )
    logger.info("Compute patch vessel size...")
    vessel_thickness_patch = compute_vessel_thickness(
        torch.squeeze(transform_obj_analysis(y_true_patch)), relative_landmark_pos
    )

    relative_patch_center_pos = np.array(sw_shape) / 2
    dist_landmark_from_patch_center = distance(
        relative_landmark_pos, relative_patch_center_pos, norm="L2"
    )

    # ------------------------- #
    #   ATTRIBUTIONS ANALYSIS   #
    # ------------------------- #
    logger.info("Attributions analysis ...")
    logger.info("_________________________")

    # Open the attribution map, without channel-dim, as a numpy array
    AttributionLoader = Compose(
        [LoadImage(image_only=False, ensure_channel_first=False), ToNumpy()]
    )
    I_attribution, meta_attribution = AttributionLoader(attribution_path)

    # Compute various data on attribution map
    logger.info("Compute descriptive statistics...")
    attribution_stats = univariate_analysis(I_attribution.flatten())
    logger.info("Compute image's total variation...")
    attribution_tv = image_total_variation(I_attribution, neighborhood="N26", norm="L1")

    # Detect the blobs in the attribution map and compute blobs' region properties
    logger.info("Blob search...")
    blobs_mask = remove_small_objects(
        detect_blob(I_attribution).astype(bool), min_size=5
    )  # For regionprops that requiere convex hull, we remove objects smaller that 4px

    selected_props = [
        "label",
        "area",
        "centroid",
        "equivalent_diameter_area",
        "feret_diameter_max",
    ]

    blobs_props, labeled_blobs, nblobs = compute_blobs_properties(
        blobs_mask, selected_props
    )
    
    # Analysis on blobs
    for blob_idx in range(1, nblobs+1):
        stats_blob = univariate_analysis(I_attribution[labeled_blobs==blob_idx])
        stats_blob["area_check"] = np.sum(labeled_blobs==blob_idx)

        blobs_props[blob_idx-1]["stats"] = stats_blob

    # ------------------------ #
    #     CREATE JSON FILE     #
    # ------------------------ #
    logger.info("Create output file(s) ...")
    logger.info("___________________________")

    _SAVE_INTERMEDIATE_RES = True

    if _SAVE_INTERMEDIATE_RES == True:
        # Completely awful but very useful
        SaveImage(
            output_dir=os.path.join(cfg.result_dir, "blobs"),
            output_ext=".nii.gz",
            output_postfix=f"blobs",
            resample=False,
            separate_folder=False
        )(blobs_mask, meta_attribution)

        model_id, _ = os.path.splitext(os.path.basename(weights_path))
        idx_involved_patch = os.path.splitext(os.path.basename(attribution_path))[0].split("_")[8]
        output_prefix = f"seg_{model_id}_{landmark_type}_{landmark_id}_{idx_involved_patch}"

        SaveImage(
            output_dir=os.path.join(cfg.result_dir, "inferences"),
            output_ext=".nii.gz",
            output_postfix=output_prefix,
            resample=False,
            separate_folder=False
        )(y_pred_patch, meta_attribution)

    # Define the basic structure of the file
    res_output = {
        "point"         : {},
        "inference"     : {},
        "attribution"   : {},
    }

    # Information about the landmark
    res_output["point"] = {
        "type": landmark_type,
        "absolute_position": landmark.pos,
        "relative_position": relative_landmark_pos,
        "degree": landmark.degree,
        "absolute_thickness": vessel_thickness_global,
        "relative_thickness": vessel_thickness_patch,
        "distance_from_center": dist_landmark_from_patch_center
    }

    # About the prediction
    res_output["inference"] = { "patch_position": patch_pos } | pred_res
    
    # About the attribution map
    res_output["attribution"] = attribution_stats
    res_output["attribution"]["total_variation"] = attribution_tv
    res_output["attribution"]["blobs"] = blobs_props

    json_data = json.dumps(convert_typing_to_native(res_output), indent=3)
    save_path = os.path.join(cfg.result_dir, "json", "res_" + os.path.basename(attribution_path).split(".")[0] + ".json")
    
    with open(save_path, "w") as json_file:
        json_file.write(json_data)

    logger.info(f"Finished. Result file saved at {save_path}")


if __name__ == "__main__":
    import utils.configuration as appcfg

    cfg = appcfg.Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": "{}/last_analyse.log".format(cfg.log_dir),
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    main()
