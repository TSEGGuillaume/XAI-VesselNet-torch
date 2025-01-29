import argparse
import logging
import logging.config # Mandatory if MONAI is not imported

import os
import json
from copy import deepcopy

from tqdm import tqdm

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import Compose, LoadImage, SaveImage, Activations, AsDiscrete
from monai.networks.utils import one_hot as OneHotEncoding
from monai.data.utils import iter_patch, decollate_batch

from network.model_creator import init_inference_model as InferenceModel
from infer import infer_single_data as Predict
from eval import evaluate
from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2ImageGraph
from utils.load_hyperparameters import load_hyperparameters as LoadHyperparameters
from utils.load_patch_position import read_path_position_from_file as ReadPositionFile
from utils.get_landmark_from_args import get_landmark_obj as GetLandmark
from utils.create_output_dirs import create_output_dirs
from utils.json_format import convert_typing_to_native
from utils.prebuilt_logs import log_hardware

from models.instanciate_model import _all_models as AvailableModels


logger = logging.getLogger("app")


def parse_arguments():
    parser = argparse.ArgumentParser()
        
    parser.add_argument(
        "attribution_dir",
        type=str,
        metavar=("ATTRIBUTION_MAPS_DIRECTORY"),
        help="Path to the directory containing attribution maps (*.nii, *.nii.gz)",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        metavar=("DATA_DIRECTORY"),
        help="Path to the directory containing the data to infer (*.nii, *.nii.gz)",
    )
    parser.add_argument(
        "graph_dir",
        type=str,
        metavar=("GRAPHS_DIRECTORY"),
        help="Path to the vessel graphs (*.vvg)",
    )
    parser.add_argument(
        "model_name",
        type=str,
        metavar=("MODEL_NAME"),
        choices=AvailableModels,
        default=AvailableModels[0],
        help="Name of the model",
    )
    parser.add_argument(
        "weights_dir",
        type=str,
        metavar=("WEIGHTS_DIR"),
        help="Path to the weights directory",
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

    args = parser.parse_args()
    return args


def get_attribution_id(in_path):
    return os.path.normpath(in_path).split(os.sep)[-1]


def eval(y_pred, y_true, T_postprocessing=None):
    
    # Should be : B, C, H, W, D
    output_channels = y_pred.shape[1]

    if T_postprocessing is None:
        T_postprocessing = Compose([])

    y_pred = T_postprocessing(y_pred)

    if output_channels == 1:
        y_true  = OneHotEncoding(labels=y_true, num_classes=2)
        y_pred  = OneHotEncoding(labels=y_pred, num_classes=2)
    elif output_channels >= 1:
        y_true  = OneHotEncoding(labels=y_true, num_classes=output_channels)

    metrics = evaluate(ys_pred=[y_pred], ys_true=[y_true])

    return metrics


def analyse_prediction(y_pred, y_true, position):
    """
    Analyse the prediction at 2 different scales : global and local.
    The analysis at the global scale returns the common metrics
    The analysis at the local scale return the model's output, activated output and the point status (TP, FP, TN, FN) of a specific location. 

    Args:
        y_pred  : The predicted image (B,C,H,W,D)
        y_true  : The ground-truth image (B,C,H,W,D)
        position: Indicate if I is already labeled. If False, the function will labelize the image.

    Returns:
        dict: The results of the analysis
    """

    ldmrk_slice = (0, slice(None)) + position # [B,C,H,W,D]

    output_channels = y_pred.shape[1]
    activation = Activations(sigmoid=True) if output_channels == 1 else Activations(softmax=True)
    binarization = AsDiscrete(threshold=0.5)

    output_values = y_pred[ldmrk_slice] # Get the raw output values

    y_pred = activation(y_pred)
    activated_values = y_pred[ldmrk_slice] # Get the raw output values

    y_pred = binarization(y_pred)

    #  Get the status of the point
    assert output_channels > 0 and output_channels < 3, f"Only binary segmentation allowed, but got {output_channels} output channels" 

    ldmrk_slice = (0, output_channels-1) + position

    if y_true[ldmrk_slice] == True:
        if y_pred[ldmrk_slice] == True:
            point_status = "TP"
        else:
            point_status = "FN"
    else:
        if y_pred[ldmrk_slice] == True:
            point_status = "FP"
        else:
            point_status = "TN"

    metrics_y_pred = eval(y_pred, y_true, T_postprocessing=None) # postprocessing already applied

    pred_nfo = {
        "point_status": point_status,
        "output_value": output_values,
        "activation_value": activated_values,
        "metrics": metrics_y_pred,
    }

    return pred_nfo


def main(attribution_dir, input_dir, graph_dir, model_name, weights_dir, hyperparameters_path, out_path=None):
    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    # Load hyperparameters for inference
    hyperparameters = LoadHyperparameters(hyperparameters_path)

    in_channels = hyperparameters["in_channels"]
    out_channels = hyperparameters["out_channels"]
    input_shape = hyperparameters["input_shape"]

    sw_shape = [in_channels] + input_shape
    sw_overlap = [0] + [hyperparameters["patch_overlap"]] * len(input_shape)
    padding_mode = "constant"

    post_T = Compose([
        Activations(sigmoid=True) if out_channels == 1 else Activations(softmax=True),
        AsDiscrete(threshold=0.5),
    ])

    # Output
    if out_path is None:
        out_path = os.path.join(cfg.result_dir, "patch")

    attribution_id = get_attribution_id(attribution_dir)
    out_xpatch, out_ypatch, out_json = create_output_dirs(
        [
            os.path.join(out_path, "patch", "x"),
            os.path.join(out_path, "patch", "y"),
            os.path.join(out_path, "json")
        ],
        attribution_id
    )

    saver_xpatch = SaveImage(
        output_dir=out_xpatch,
        output_ext=".nii.gz",
        output_postfix="",
        resample=False,
        separate_folder=False,
    )
    saver_ypatch = SaveImage(
        output_dir=out_ypatch,
        output_ext=".nii.gz",
        output_postfix="", # Defined dynamically
        resample=False,
        separate_folder=False,
    )

    samples = {}
    image_loader = LoadImage(ensure_channel_first=True, image_only=False)

    files = [f for f in os.listdir(attribution_dir) if f.endswith(".txt")]

    # TODO: This is suboptimized as fuck.
    # 1/ First, a same patch will be infered multiple times, once per node included in the patch.
    for file in tqdm(files):
        # Example of filename:
        #  3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_pos.txt
        f_basename = os.path.basename(file).split(".")[0]
        decompose_fname = f_basename.split("_")

        fname_prefix = "_".join(decompose_fname[:-1])

        sample_name = "_".join(decompose_fname[:2])
        training_strategy = decompose_fname[2]
        weights_path = os.path.join(weights_dir, "{}.pth".format("_".join(decompose_fname[3:5])))
        
        if sample_name not in samples:
            x, meta = image_loader(os.path.join(input_dir, f"{sample_name}_{training_strategy}.nii.gz"))
            y, _ = image_loader(os.path.join(input_dir, f"{sample_name}.nii.gz"))

            model = InferenceModel(
                model_name=model_name,
                weights_path=weights_path,
                in_channels=in_channels,
                out_channels=out_channels,
                device=device,
            )

            graph = LoadVesselGraph(os.path.join(graph_dir, f"{sample_name}_graph.vvg"))
            graph = Anatomic2ImageGraph(graph, meta["original_affine"])

            samples[sample_name] = {
                "x": x,
                "y": y,
                "meta": meta,
                "model": model,
                "graph": graph,
            }
        
        else:
            x = samples[sample_name]["x"]
            y = samples[sample_name]["y"]
            meta = samples[sample_name]["meta"]
            model = samples[sample_name]["model"]
            graph = samples[sample_name]["graph"]
      
        patch_pos = ReadPositionFile(os.path.join(attribution_dir, file))

        # Change the position of the origin to align with the patch
        meta_cpy = deepcopy(meta)
        image_space_new_origin = np.array(patch_pos[0])
        world_space_new_origin = np.matmul(meta["affine"][:-1, :-1], np.atleast_2d(image_space_new_origin).T)
        meta_cpy["affine"][:-1,-1:] = meta["affine"][:-1,-1:] + world_space_new_origin # [., -1:] to keep 2D dimension of the returned array

        # Translate from [(x_start, y_start, z_start), (x_end, y_end, z_end)] to [[channel_start, channel_end], [x_start, x_end], [y_start, y_end], [z_start, z_end]]
        patch_pos = np.concatenate([[[0, in_channels]], [[dim_start, dim_end] for dim_start, dim_end in zip(patch_pos[0], patch_pos[1])]], axis=0)

        # Creates the patches following the strategy implemented in compute_attribution.py. start_pos is set to jump directly in the interesting position given by the attribution file "_pos.txt"
        patches = iter_patch(
            x.get_array(), patch_size=sw_shape, start_pos=patch_pos[:, 0], overlap=sw_overlap, mode=padding_mode
        )

        for x_patch, pos in patches:
            # Check the positions given by iter_patch and "_pos.txt" file are equal
            if (patch_pos == pos).all() == True:
                # Save the input patch
                meta_cpy["filename_or_obj"] = fname_prefix
                x_patch = MetaTensor(x_patch, meta=meta_cpy)
                saver_xpatch(x_patch)

                # Retrieves the y_true patch at the same location. The use of iter_patch is very ugly but it ensure the same condition of the patch extraction between attribution and x, particularly for padding. 
                y_true_patches = iter_patch(
                    y.get_array(), patch_size=sw_shape, start_pos=patch_pos[:, 0], overlap=sw_overlap, mode=padding_mode
                )
                for y_true_patch, y_true_patch_pos in y_true_patches: # Actually, only the first iteration is supposed to be perform
                    # Check the positions given by iter_patch and "_pos.txt" file are equal. Raise exception otherwise
                    if (patch_pos == y_true_patch_pos).all() == False:
                        raise RuntimeError("Patchs position are not aligned")
                    break # We have the right y_true patch, we can avoid to continue the loop

                # Save the ground-truth patch
                meta_cpy["filename_or_obj"] = f"{fname_prefix}_ytrue"
                y_true_patch = MetaTensor(y_true_patch, meta=meta_cpy)
                saver_ypatch(y_true_patch)

                # Predict and save the predicted patch
                y_pred_patch = Predict(model=model, data=x_patch, device=device)[0] # Return a list of MetaTensor of shape (B,C,X,Y,Z)

                landmark_type = decompose_fname[6]
                landmark_id = decompose_fname[7]

                landmark = GetLandmark(graph, landmark_type, landmark_id)
                relative_landmark_pos = tuple(lpos - ppos for lpos, ppos in zip(landmark.pos, patch_pos[1:, 0]))

                metrics = analyse_prediction(
                    y_pred=y_pred_patch, 
                    y_true=torch.unsqueeze(y_true_patch, dim=0).to(device), # Add the batch dimension to match with y_pred dim.
                    position=relative_landmark_pos
                )

                with open(os.path.join(out_json, f"{fname_prefix}.json"), "w") as f:
                # About the prediction
                    json_data = json.dumps(
                        convert_typing_to_native(
                            { "patch_position": patch_pos } | metrics
                        ), 
                        indent=4
                    )
                    f.write(json_data)        

                meta_cpy["filename_or_obj"] = f"{fname_prefix}_ypred"
                final_y_pred_patch = [post_T(i) for i in decollate_batch(y_pred_patch)]
                y_pred_patch = MetaTensor(final_y_pred_patch[0], meta=meta_cpy)
                saver_ypatch(y_pred_patch)  

                break # We have the right y_true patch, we can avoid to continue the loop


if __name__ == "__main__":
    import utils.configuration as appcfg

    cfg = appcfg.Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": f"{cfg.log_dir}/patch_prediction.log",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    for handler in logger.handlers:
        if type(handler) == logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    main(
        args.attribution_dir,
        args.data_dir,
        args.graph_dir,
        args.model_name,
        args.weights_dir,
        args.hyperparameters,
        args.output
    )