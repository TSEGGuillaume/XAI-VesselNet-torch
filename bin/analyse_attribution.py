import argparse
import logging

import os

import torch
from torch.nn import Module
from torch import device 

import numpy as np
from monai.data.meta_tensor import MetaTensor
from monai.transforms import LoadImage, Compose, SpatialCrop, SpatialPad, Activations, AsDiscrete
from monai.networks.utils import one_hot as OneHotEncoding
from monai.data.utils import decollate_batch

from utils.load_hyperparameters import load_hyperparameters
from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2ImageGraph
from utils.get_landmark_from_args import get_landmark_position
from utils.load_patch_position import read_path_position_from_file
from network.model_creator import init_inference_model
from utils.prebuilt_logs import log_hardware

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
    group.add_argument('--node', '-n', type=int, metavar=("ID_NODE"), help="id of the node to inspect", default=None)
    group.add_argument('--centerline', '-c', type=int, metavar=("ID_CENTERLINE"), help="id of the centerline to inspect", default=None)
    group.add_argument('--position', '-p', nargs=3, type=int, metavar=("X", "Y", "Z"), help="image coordinates (x,y,z) of the voxel to inspect", default=None)

    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args


def eval_patch_prediction(model: Module, x: MetaTensor, y: MetaTensor, patch_pos: tuple, input_size: tuple, device: device) -> tuple[dict, MetaTensor, MetaTensor]:
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
        The tuple ( metrics, y_pred, y_true).
    """
    # Pre-processing
    start_roi, end_roi = patch_pos

    y_pred = infer_patch(model=model, data=x, device=device, patch_pos=patch_pos, input_size=input_size, postprocessing=None)
    y_pred = y_pred[0] # We already know there is only one image

    output_channels = y_pred.shape[1] # No more batch dimension   

    pred_postprocess_T = Compose(
        [
            # include saver here
            Activations(sigmoid=True) if output_channels == 1 else Activations(softmax=True),
            AsDiscrete(threshold=0.5),
        ]
    )
    y_pred_postprocess = torch.stack([pred_postprocess_T(i) for i in decollate_batch(y_pred)])

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


def analyse_prediction(model: Module, x: MetaTensor, y: MetaTensor, patch_pos: tuple, input_size: tuple, device: device, landmark_pos:tuple):
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
        The tuple ( metrics, y_pred, y_true).
    """
    metrics, y_pred, y_true = eval_patch_prediction(model, x, y, patch_pos, input_size, device)

    # Compute the relative position of the node in the provided patch
    relative_landmark_x = landmark_pos[0] - patch_pos[0][0]
    relative_landmark_y = landmark_pos[1] - patch_pos[0][1]
    relative_landmark_z = landmark_pos[2] - patch_pos[0][2]

    output_channels = y_pred.shape[1]
    act = Activations(sigmoid=True, dim=1) if output_channels == 1 else Activations(softmax=True, dim=1)

    output_values       = y_pred[0, :, relative_landmark_x, relative_landmark_y, relative_landmark_z] # Get the raw output values
    working_y_pred      = act(y_pred)
    activation_values   = working_y_pred[0, :, relative_landmark_x, relative_landmark_y, relative_landmark_z] # Get the activated output values

    logger.info(f"Compare prediction ({working_y_pred.shape}), with ground-truth ({y_true.shape})")

    working_y_pred = AsDiscrete(threshold=0.5)(working_y_pred)

    #  Get the status of the point
    if output_channels == 1:
        if y_true[:, :, relative_landmark_x, relative_landmark_y, relative_landmark_z] == True:
            if working_y_pred[:, :, relative_landmark_x, relative_landmark_y, relative_landmark_z] == True:
                point_status = "TP"
            else:
                point_status = "FN"
        else:
            if working_y_pred[:, :, relative_landmark_x, relative_landmark_y, relative_landmark_z] == True:
                point_status = "FP"
            else:
                point_status = "TN"

    elif output_channels == 2:
        if y_true[:, 1, relative_landmark_x, relative_landmark_y, relative_landmark_z] == True:
            if working_y_pred[:, 1, relative_landmark_x, relative_landmark_y, relative_landmark_z] == True:
                point_status = "TP"
            else:
                point_status = "FN"
        else:
            if working_y_pred[:, 1, relative_landmark_x, relative_landmark_y, relative_landmark_z] == True:
                point_status = "FP"
            else:
                point_status = "TN"
 
    else:
        raise RuntimeError("Only binary segmentation allowed.")

    pred_nfo = {
        "point_status": point_status,
        "output_value": output_values,
        "activation_value": activation_values,
        "metrics": metrics
    }

    return pred_nfo


def main():
    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    # Variables
    attribution_path        = args.attribution
    x_path                  = args.image 
    y_true_path             = args.reference
    graph_path              = args.graph
    patch_pos_path          = args.patch
    model_name              = args.model
    weights_path            = args.weights
    hyperparameters_path    = args.hyperparameters

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
        raise NotImplementedError("You should provide a node, acenterline or a position")

    # Load hyperparameters for training
    hyperparameters = load_hyperparameters(hyperparameters_path)

    in_channels     = hyperparameters["in_channels"]
    out_channels    = hyperparameters["out_channels"]
    sw_shape        = hyperparameters["patch_size"]

    # Load the image and its associated ground-truth
    I_x, meta   = LoadImage(image_only=False, ensure_channel_first=True)(x_path) # Assume x's metadata as reference
    I_y_true    = LoadImage(image_only=True, ensure_channel_first=True)(y_true_path)

    I_x         = I_x.to(device)
    I_y_true    = I_y_true.to(device)

    patch_pos = read_path_position_from_file(patch_pos_path)

    # Load the graph and convert to image coordinate system
    vessel_graph = LoadVesselGraph(graph_path)
    vessel_graph = Anatomic2ImageGraph(vessel_graph, meta["original_affine"])

    landmark_pos = get_landmark_position(graph=vessel_graph, landmark_type=landmark_type, landmark_id=landmark_id)

    model = init_inference_model(model_name=model_name, weights_path=weights_path, in_channels=in_channels, out_channels=out_channels, device=device)
    print(analyse_prediction(model=model, x=I_x, y=I_y_true, patch_pos=patch_pos, input_size=sw_shape, device=device, landmark_pos=landmark_pos))


if __name__=="__main__":  
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