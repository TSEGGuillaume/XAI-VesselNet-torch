import argparse
import os

import pandas as pd
import numpy as np

from monai.transforms import LoadImage, AsDiscrete, SaveImage
from monai.data.utils import iter_patch

from scipy import ndimage as ndi

from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2Image
from utils.load_hyperparameters import load_hyperparameters 


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "y_pred_path",
        type=str,
        metavar=("PREDICTION_PATH"),
        help="Path to the prediction to analyse for subsampling",
    )

    parser.add_argument(
        "y_true_path",
        type=str,
        metavar=("GROUNDTRUTH_PATH"),
        help="Path to the ground-truth",
    )

    parser.add_argument(
        "graph_path",
        type=str,
        metavar=("GRAPH_PATH"),
        help="Path to the graph to subsample",
    )

    parser.add_argument(
        "hyperparameters_path",
        type=str,
        metavar=("HYPERPARAMETERS_PATH"),
        help="Path to the hyperparameters file",
    )

    parser.add_argument(
        "--savedir",
        "-s",
        type=str,
        metavar=("SAVE_DIRECTORY"),
        default=".",
        help="Path to the directory to save results",
    )

    args = parser.parse_args()
    return args


def distance_map(I: np.ndarray, method: str = "edt") -> np.ndarray:
    """
    Distance map transformation. Compute the distance map from a binary image.
    Implemented distance methods :
        - "edt" : Exact Euclidean Distance Transform

    Parameters
        I       : The image to transform
        method  : The key of the transformation to compute ; see implemented distance methods

    Returns
        The distance map
    """

    # Exact Euclidean Distance Transform
    if method == "edt":
        distance_map = ndi.distance_transform_edt(I)
    else:
        raise ValueError("Selected method not available. Available methods : `edt`")

    return distance_map


def apply_filter_on_df(df: pd.DataFrame, filter: dict) -> pd.DataFrame:
    bool_df = True
    for k, v in filter.items():
        bool_df = bool_df & (df[k]==v)

    return df.loc[bool_df]


def main(y_pred_path: str, y_true_path: str, graph_path: str, hyperparameters_path: str, user_save_dir: str):

    save_dir = user_save_dir

    I_y_pred, meta  = LoadImage(ensure_channel_first=True, image_only=False)(y_pred_path)
    I_y_true        = LoadImage(ensure_channel_first=True, image_only=True)(y_true_path)
    
    print("Prediction :", I_y_pred.shape, "| Ground-truth :", I_y_true.shape)

    affine = meta["affine"]

    # Parameters
    hyperparameters = load_hyperparameters(hyperparameters_path)

    output_channels = hyperparameters["out_channels"]
    sw_shape        = [output_channels] + hyperparameters["patch_size"] # iter_patch needs all the dimensions !
    sw_overlap      = [0] + [hyperparameters["patch_overlap"]] * len(hyperparameters["patch_size"])

    if output_channels == 1:
        idx_vessel_channel = 0
    elif output_channels == 2:
        idx_vessel_channel = 1
    else:
        raise RuntimeError("Only binary segmentation supported (single-channel or OHE.)")

    # Preprocess image
    # Binarize the prediction. /!\ We assume the loaded image has been previously activated
    I_y_pred = AsDiscrete(threshold=0.5)(I_y_pred)
    I_y_true_skel = distance_map(I_y_true[0]) # Ground-truth is not OHE

    print("Binary prediction :", I_y_pred.shape, "| Skeleton :", I_y_true_skel.shape )

    SaveImage(output_dir=f"{save_dir}/skel", output_ext=".nii.gz", output_postfix=f"edt", resample=False, separate_folder=False,)( np.expand_dims(I_y_true_skel, axis=0), meta )

    # Graph
    graph = LoadVesselGraph(graph_path)
    graph = Anatomic2Image(graph, affine)

    df = pd.DataFrame(columns=["type", "id", "pred_status", "patch_pos", "idx_patch", "degree", "thickness"])
    
    # Loop over nodes
    for node_id, node in graph.nodes.items():
        
        pred_status = "TP" if I_y_pred[idx_vessel_channel, node.pos[0], node.pos[1], node.pos[2]] == 1 else "FN"
        thickness = I_y_true_skel[node.pos[0], node.pos[1], node.pos[2]]

        patches = iter_patch(I_y_pred.numpy(), patch_size=sw_shape, overlap=sw_overlap, mode="constant") # Allow the reset at each node

        idx_involved_patch = 0
        for _, patch_pos in patches:
            if (
                node.pos[0] >= patch_pos[1,0] and node.pos[0] < patch_pos[1,1] and 
                node.pos[1] >= patch_pos[2,0] and node.pos[1] < patch_pos[2,1] and 
                node.pos[2] >= patch_pos[3,0] and node.pos[2] < patch_pos[3,1]
            ):
                # "type", "id", "degree", "thickness", "pred_status"
                new_row = ["node", node_id, pred_status, patch_pos.tolist(), idx_involved_patch, node.degree, thickness]
                df = pd.concat([df, pd.DataFrame([new_row], columns=df.columns)], ignore_index=True)

                idx_involved_patch += 1

    # Loop over centerlines
    for centerline_id, centerline in graph.connections.items():
        centerline_midpoint = centerline.getMidPoint()

        pred_status = "TP" if I_y_pred[idx_vessel_channel, centerline_midpoint.pos[0], centerline_midpoint.pos[1], centerline_midpoint.pos[2]] == 1 else "FN"
        thickness = I_y_true_skel[centerline_midpoint.pos[0], centerline_midpoint.pos[1], centerline_midpoint.pos[2]]

        patches = iter_patch(I_y_pred.numpy(), patch_size=sw_shape, overlap=sw_overlap, mode="constant")

        idx_involved_patch = 0
        for _, patch_pos in patches:
            if (
                centerline_midpoint.pos[0] >= patch_pos[1,0] and centerline_midpoint.pos[0] < patch_pos[1,1] and 
                centerline_midpoint.pos[1] >= patch_pos[2,0] and centerline_midpoint.pos[1] < patch_pos[2,1] and 
                centerline_midpoint.pos[2] >= patch_pos[3,0] and centerline_midpoint.pos[2] < patch_pos[3,1]
            ):
                new_row = ["skvx", centerline_id, pred_status, patch_pos.tolist(), idx_involved_patch, 2, thickness]
                df = pd.concat([df, pd.DataFrame([new_row], columns=df.columns)], ignore_index=True)

                idx_involved_patch += 1

    print("Dataframe", df.shape)

    # Filter on columns value
    filters = [
        {
            "pred_status": "TP",
            "type": "node",
            "degree": 1,
        },
        {
            "pred_status": "TP",
            "type": "node",
            "degree": 3,
        },
        {
            "pred_status": "FN",
            "type": "node",
            "degree": 1,
        },
        {
            "pred_status": "FN",
            "type": "node",
            "degree": 3,
        },
        {
            "pred_status": "TP",
            "type": "skvx",
            "degree": 2,
        },
        {
            "pred_status": "FN",
            "type": "skvx",
            "degree": 2,
        }
    ]

    for filter in filters:
        filtered_df = apply_filter_on_df(df, filter)
        print("Filter on ", filter.values(), "->", filtered_df.shape)

        # Save
        save_dir = os.path.join(
            save_dir,
            "_".join(f"{_filter_val}" for _filter_val in filter.values())
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = "{}/{}_subsampling_landmarks.csv".format(
            save_dir,
            os.path.basename(meta["filename_or_obj"]).split('.')[0]
        )

        subsample_df = filtered_df.sample(n=min(100, len(filtered_df.index)), random_state=5) # seed for reproductibility

        subsample_df.to_csv(save_path, sep=";")
        print("Saved at ", save_path)


if __name__ == "__main__":
    args = parse_arguments()

    print(args)

    y_pred_path = args.y_pred_path
    y_true_path = args.y_true_path
    graph_path = args.graph_path
    hyperparameters_path = args.hyperparameters_path

    user_save_dir = args.savedir

    main(y_pred_path, y_true_path, graph_path, hyperparameters_path, user_save_dir)