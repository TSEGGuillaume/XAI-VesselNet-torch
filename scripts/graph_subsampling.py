import argparse
import os

from monai.transforms import LoadImage, AsDiscrete, SaveImage
import pandas as pd
import numpy as np

from scipy import ndimage as ndi

from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2Image
from utils.load_hyperparameters import load_hyperparameters 


#TODO: Write the 'patch' version of this function:
#   1/ Duplicate the lines according to the number of patches involved: add to the Dataframe a column patch_pos and patch_id.
#   2/ Filter the lines
#   3/ Choose X random points (minimum 100 except in the case where the number of lines after filtering is lower, in this case X=count_min)
#   4/ Save in a file the subsampled list of the graph: [type;id;patch_pos]
#   5/ Temporarily modify compute_attribution.py so that it no longer iterates over the complete graph but the subsampled file"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image_path",
        type=str,
        metavar=("IMAGE_PATH"),
        help="Path to the prediction to analyse for subsampling",
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


def apply_filter_on_df(df: pd.DataFrame, filter: dict):
    bool_df = True
    for k, v in filter.items():
        bool_df = bool_df & (df[k]==v)

    return df.loc[bool_df]

def main(image_path, graph_path, hyperparameters_path):
    # Load image as binary segmentation. /!\ We assume the loaded image has been previously activated
    I, meta = LoadImage(ensure_channel_first=True, image_only=False)(image_path)

    affine = meta["affine"]
    output_channels = load_hyperparameters(hyperparameters_path)["out_channels"]

    if output_channels == 1:
        idx_vessel_channel = 0
    elif output_channels == 2:
        idx_vessel_channel = 1
    else:
        raise RuntimeError("Only binary segmentation supported (single-channel or OHE.)")

    I = AsDiscrete(threshold=0.5)(I)
    I_skel = distance_map(I[idx_vessel_channel])

    SaveImage(output_dir=".", output_ext=".nii.gz", output_postfix=f"edt", resample=False, separate_folder=False,)( np.expand_dims(I_skel, axis=0), meta) # TEMP

    graph = LoadVesselGraph(graph_path)
    graph = Anatomic2Image(graph, affine)

    df = pd.DataFrame(columns=["type", "id", "degree", "thickness", "pred_status"])
    
    for node_id, node in graph.nodes.items():
        if output_channels == 1:
            idx_vessel_channel = 0
        elif output_channels == 2:
            idx_vessel_channel = 1
        else:
            raise RuntimeError("Only binary segmentation supported (single-channel or OHE.)")

        pred_status = "TP" if I[idx_vessel_channel, node.pos[0], node.pos[1], node.pos[2]] == 1 else "FN"

        # "type", "id", "degree", "thickness", "pred_status"
        new_row = ["node", node_id, node.degree, float('NaN'), pred_status]
        df = pd.concat([df, pd.DataFrame([new_row], columns=df.columns)], ignore_index=True)

    for centerline_id, centerline in graph.connections.items():
        centerline_midpoint = centerline.getMidPoint()

        pred_status = "TP" if I[idx_vessel_channel, centerline_midpoint.pos[0], centerline_midpoint.pos[1], centerline_midpoint.pos[2]] == 1 else "FN"
        thickness = I_skel[centerline_midpoint.pos[0], centerline_midpoint.pos[1], centerline_midpoint.pos[2]]
        # "type", "id", "degree", "thickness", "pred_status"
        new_row = ["skvx", centerline_id, 2, thickness, pred_status]
        df = pd.concat([df, pd.DataFrame([new_row], columns=df.columns)], ignore_index=True)

    # Filter on columns value
    filter = {
        "pred_status": "FN",
        "type": "skvx",
        "degree": 2,
    }
    print("Filter on ", filter.values())
    filtered_df = apply_filter_on_df(df, filter)

    save_dir = "_".join(f"{_filter_val}" for _filter_val in filter.values())

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = "./{}/{}_subsampling_landmarks.csv".format(
        save_dir,
        os.path.basename(meta["filename_or_obj"]).split('.')[0]
    )
    filtered_df.to_csv(save_path, sep=";")
    print("Saved at ", save_path)


if __name__ == "__main__":
    args = parse_arguments()

    image_path = args.image_path
    graph_path = args.graph_path
    hyperparameters_path = args.hyperparameters_path

    main(image_path, graph_path, hyperparameters_path)