import argparse
import logging
import logging.config # Mandatory if MONAI is not imported

import os
import json
import copy
from tqdm import tqdm

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import Compose, LoadImage, SaveImage, Activations, AsDiscrete
from monai.networks.utils import one_hot as OneHotEncoding
from monai.data.utils import iter_patch, decollate_batch

from network.model_creator import init_inference_model as InferenceModel
from utils.load_hyperparameters import load_hyperparameters as LoadHyperparameters
from infer import infer_single_data as Predict
from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2ImageGraph, image_to_anatomic as UpdateOrigin
from utils.get_landmark_from_args import get_landmark_obj as GetLandmark
from eval import evaluate
from utils.load_patch_position import read_path_position_from_file as ReadPositionFile
from utils.template_filename import CXAIVesselNetFilename
from utils.create_output_dirs import create_output_dirs
from utils.json_format import convert_typing_to_native
from utils.prebuilt_logs import log_hardware

from models.instanciate_model import _all_models as AvailableModels


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


def get_attribution_id(in_path) -> str:
    """
    Return the identifier of the dataset (the last directory of the input path), i.e. /<path>/<dataset_id>/<filename>.nii.gz

    Args:
        in_path : The input path

    Returns:
        The dataset ID
    """
    return os.path.normpath(in_path).split(os.sep)[-1]


def eval(y_pred: MetaTensor, y_true: MetaTensor, T_postprocessing:Compose=None) -> dict:
    """
    Evaluate the prediction w.r.t the ground-truth

    Args:
        y_pred : The prediction Tensor of shape (B,C,H,W,[D])
        y_true : The ground-truth Tensor of shape (B,C,H,W,[D])
        T_postprocessing : The transforms to apply as post-processing

    Returns:
        The dictionnary of the metrics results
    """
    output_channels = y_pred.shape[1]

    if T_postprocessing is not None:
        y_pred = torch.stack([T_postprocessing(i) for i in decollate_batch(y_pred)])

    # Not optimized code lines but more explicit : whatever the number of channels, we use OHE (even binary)
    # The reason is that with the current version of monai (1.1.0), metrics(binary_tensor) and metrics(one_hot_encoded_binary_tensor)) seems not to give the same results.
    # I.o.t not change the behavior between monoclass and multi-classes, we encode everthing to one-hot 
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
        y_pred  : The predicted image (B,C,H,W,[D])
        y_true  : The ground-truth image (B,C,H,W,[D])
        position: The position of the output logit to analyze.

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


# TODO : Working directly on <MetaTensor>.affine would be cleaner imo. In this case, what happens during saving: which affine between meta["affine"] or obj.affine is taken into account ?
def main(in_dir_attribution, in_dir_x, in_graph_dir, model_name, weights_dir, hyperparameters_path, output_dir=None):

    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    # Load hyperparameters
    hyperparameters = LoadHyperparameters(hyperparameters_path)

    in_channels = hyperparameters["in_channels"]
    out_channels = hyperparameters["out_channels"]
    input_shape = hyperparameters["input_shape"]

    sw_shape_x = [in_channels] + input_shape
    sw_overlap = [0] + [hyperparameters["patch_overlap"] for _ in range(len(input_shape))]
    sw_padding_mode = "constant"

    # List files we need to process
    pos_files = [f for f in os.listdir(in_dir_attribution) if f.endswith("_pos.txt")]

    # A position filename is formatted as {DATASET_ID}_{SAMPLE_ID}_{TRAINING_STRATEGY}_model_{MODEL_ID}_[...]_pos.txt
    # The first three elements indicate which data to predict, while the fifth element indicate model weights to use.
    # 
    # The data structure of unique_file_model_combinations is :
    # { data_and_model_id_str: { patch_position_id: [], },}
    # Where the list contains all the attribution files that are related to the same patch. This avoid the multiple predictions of the same patch, and just save multiple times the sameprediction with different names. 
    unique_file_model_combinations = dict()

    logger.info("Read position files...")

    for pos_file in tqdm(pos_files):
        
        unique_file_model_combination_key = "_".join(pos_file.split('_')[:5])

        pos = ReadPositionFile(os.path.join(in_dir_attribution, pos_file))
        file_pos_identifier = "_".join([f"{dim_start}_{dim_end}" for dim_start, dim_end in zip(pos[0], pos[1])])

        unique_file_model_combinations.setdefault(unique_file_model_combination_key, {}).setdefault(file_pos_identifier, []).append(pos_file)

    image_loader = LoadImage(image_only=False, ensure_channel_first=True)

    post_T = Compose([
        Activations(sigmoid=True) if out_channels == 1 else Activations(softmax=True),
        AsDiscrete(threshold=0.5),
    ])

    if output_dir is None:
        output_dir = os.path.join(cfg.result_dir, "patch")

    attribution_id = get_attribution_id(in_dir_attribution)

    out_xpatch, out_json = create_output_dirs(
        [
            os.path.join(output_dir, "patch"),
            os.path.join(output_dir, "json")
        ],
        attribution_id
    )

    logger.info(f"Output directory : {out_json}")

    image_x_saver = SaveImage(
        output_dir=out_xpatch,
        output_ext=".nii.gz",
        output_postfix="", # It will differs dynamically according to the saved image : x, ytrue, ypred 
        resample=False,
        separate_folder=False,
    )
    image_y_saver = SaveImage(
        output_dir=out_xpatch,
        output_ext=".nii.gz",
        output_postfix="", # It will differs dynamically according to the saved image : x, ytrue, ypred 
        resample=False,
        separate_folder=False,
        output_dtype=np.uint8,
    )
    
    for combination_name, file_pos_dict in unique_file_model_combinations.items():

        # Retrieve image and model data
        split_elems = combination_name.split('_')
        sample_name = "{}".format("_".join(split_elems[:2]))
        weights_name = "{}.pth".format("_".join(split_elems[3:]))

        logger.info("Infer {} with the model {}".format("{}_{}.nii.gz".format(sample_name, split_elems[2]), weights_name))

        # Create the model and load weights
        model = InferenceModel(
            model_name=model_name,
            weights_path=os.path.join(weights_dir, weights_name),
            in_channels=in_channels,
            out_channels=out_channels,
            device=device,
        )

        x, meta = image_loader(os.path.join(in_dir_x, "{}_{}.nii.gz".format(sample_name, split_elems[2])))
        y, _ = image_loader(os.path.join(in_dir_x, f"{sample_name}.nii.gz"))

        graph = LoadVesselGraph(os.path.join(in_graph_dir, "{}_graph.vvg".format("_".join(split_elems[:2]))))
        graph = Anatomic2ImageGraph(graph, meta["original_affine"])

        patches = iter_patch(
            x.get_array(), patch_size=sw_shape_x, overlap=sw_overlap, mode=sw_padding_mode
        )

        logger.info(f"Iterate over patch iterator. This may take a while...")
        # Useless, mainly for printing...
        stats_dict = {
            "Total saved files": 0,
            "Total patch infered": 0,
        }

        for patch_x, pos in tqdm(patches):

            stats_dict["Total patch infered"] += 1

            # Update the origin of the patch given the world's coordinate system (i.e. image's origin)  
            image_origin = np.array([dim[0] for dim in pos[1:]])

            patch_meta = copy.deepcopy(meta)
            patch_meta["affine"][:-1,-1:] = UpdateOrigin(image_origin, patch_meta["affine"], use_origin=True)

            patch_x = MetaTensor(
                patch_x,
                meta=patch_meta
            ) # The batch dimension is added in Predict (DataLoader)

            # Predict
            patch_ypred = Predict(model=model, data=patch_x, device=device)[0] # Returns a list of MetaTensor of shape (B,C,X,Y,D)

            patch_pos_identifier = "_".join(["{}_{}".format(dim[0], dim[1]) for dim in pos[1:]])

            try:
                assoc_files = file_pos_dict[patch_pos_identifier]

                # It's awfull but ensures the patch-ization behavior between x and y is equal
                y_true_patches = iter_patch(
                    y.get_array(), patch_size=sw_shape_x, start_pos=pos[:, 0], overlap=sw_overlap, mode=sw_padding_mode
                )
                for patch_ytrue, y_true_patch_pos in y_true_patches: # Actually, only the first iteration is supposed to be perform
                    # Check the positions (spatial only) given by iter_patch and "_pos.txt" file are equal. Raise exception otherwise
                    if (pos[1:] == y_true_patch_pos[1:]).all() == False:
                        raise RuntimeError(
                            "Spatial position of patches does not match. Expected {}, got {}".format(
                                pos[1:], 
                                y_true_patch_pos[1:]
                            )
                        )
                
                    break # We have the right y_true patch, we can avoid to continue the loop
    
                # For all files associated with the current patch position
                for pos_file in assoc_files:
                    stats_dict["Total saved files"] += 1

                    logger.debug(f"Associated file found : {pos_file} \n\tPosition {pos}")

                    fname = CXAIVesselNetFilename(filename=pos_file)

                    fname_prefix = fname.get_prefix()

                    # Get the landmark and evaluate prediction w.r.t it
                    landmark = GetLandmark(graph, fname.landmark_type, fname.landmark_id)
                    relative_landmark_pos = tuple(lpos - ppos for lpos, ppos in zip(landmark.pos, pos[1:, 0]))
                    metrics = analyse_prediction(
                        y_pred=patch_ypred, 
                        y_true=torch.unsqueeze(torch.from_numpy(patch_ytrue), dim=0).to(device), # Add the batch dimension to match with y_pred dim.
                        position=relative_landmark_pos
                    )

                    # Save patches (x, ytrue, ypred)
                    patch_meta["filename_or_obj"] = f"{fname_prefix}_x"
                    image_x_saver(MetaTensor(patch_x.get_array(), meta=patch_meta))

                    patch_meta["filename_or_obj"] = f"{fname_prefix}_ytrue"
                    image_y_saver(MetaTensor(patch_ytrue, meta=patch_meta))

                    pp_patch_ypred = [post_T(i) for i in decollate_batch(patch_ypred)][0] # Only one for sure
                    patch_meta["filename_or_obj"] = f"{fname_prefix}_ypred"
                    image_y_saver(MetaTensor(pp_patch_ypred.get_array(), meta=patch_meta))

                    # Save the metrics JSON
                    with open(os.path.join(out_json, f"{fname_prefix}_patch.json"), "w") as f:
                        # Save the JSON
                        json_data = json.dumps(
                            convert_typing_to_native(
                                { "patch_position": pos } | metrics
                            ), 
                            indent=4
                        )
                        f.write(json_data)    

            except KeyError as e:
                logger.debug(f"No position file associated with {pos}")
                pass

        logger.info(f"Prediction statistics : {stats_dict}")
                

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

    main(
        args.attribution_dir,
        args.data_dir,
        args.graph_dir,
        args.model_name,
        args.weights_dir,
        args.hyperparameters,
        args.output
    )