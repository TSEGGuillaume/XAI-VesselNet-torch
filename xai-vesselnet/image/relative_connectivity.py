import argparse
import logging

import os

import numpy as np
from numpy import ndarray

from monai.transforms import LoadImage, SaveImage
from monai.data.meta_tensor import MetaTensor

from skimage.morphology import ball
from skimage.measure import label

from graph.voreen_parser import (
    voreen_VesselGraphSave_file_to_graph as LoadVesselGraph,
)
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2ImageGraph
from utils.load_patch_position import read_path_position_from_file
from graph.graph import CGraph, CNode


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "id_landmark",
        type=int,
        metavar=("LANDMARK_IDENTIFIER"),
        help="The ID of the landmark",
    )
    parser.add_argument(
        "vessel_mask",
        type=str,
        metavar=("VESSEL_MASK_PATH"),
        help="Path to the vessel mask (*.nii)",
    )
    parser.add_argument(
        "vessel_graph",
        type=str,
        metavar=("VESSEL_GRAPH_PATH"),
        help="Path to the vascular graph (*.vvg)",
    )
    parser.add_argument(
        "patch_position",
        type=str,
        metavar=("PATCH_POSITION_PATH"),
        help="Path to the patch position file (*.txt)",
    )

    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args


def create_mask_bifurcation(
    I: MetaTensor | ndarray, landmark: CNode, threshold: float = 0.75
) -> ndarray:
    """
    Create the mask of a given bifurcation

    Args:
        I           : The vessel mask (H,W,[D]).
        landmark    : The instance of the corresponding bifurcation node.
        threshold   : The dilation stop condition. The dilation process is automatically stopped when the number of new background voxels exceeds a proportion of the new vessel voxels count (threshold).

    Note: if the dilation stop condition is never reached, it stops when the structuring element reaches its asbolted size limit. See SE_radii range

    Returns:
        The bifurcation mask
    """

    SE_radii = range(2, 10, 1)

    SE_range = []
    for SE_radius in SE_radii:
        SE_range.append(ball(SE_radius))

    out_bif_mask = np.zeros_like(I)

    mem_count_vessel = 0
    mem_count_background = 0

    for SE, SE_radius in zip(SE_range, SE_radii):

        logger.debug(f"SE radius: {SE_radius}")

        y_true_pad = np.pad(I, SE_radius)

        mask_current_bif = (
            y_true_pad[
                landmark.pos[0] : landmark.pos[0] + SE.shape[0],
                landmark.pos[1] : landmark.pos[1] + SE.shape[1],
                landmark.pos[2] : landmark.pos[2] + SE.shape[2],
            ]
            * SE
        )

        new_vessel = np.sum(mask_current_bif) - mem_count_vessel
        new_background = (
            np.sum(SE) - mem_count_background - new_vessel - mem_count_vessel
        )

        w_mask_current_bif = np.zeros_like(y_true_pad)
        w_mask_current_bif[
            landmark.pos[0] : landmark.pos[0] + SE.shape[0],
            landmark.pos[1] : landmark.pos[1] + SE.shape[1],
            landmark.pos[2] : landmark.pos[2] + SE.shape[2],
        ] = mask_current_bif

        # Undo the padding
        out_bif_mask = (
            out_bif_mask
            + w_mask_current_bif[
                SE_radius:-SE_radius, SE_radius:-SE_radius, SE_radius:-SE_radius
            ]
        )

        logger.debug(f"New vessel {new_vessel} | New background {new_background}")

        if new_background > threshold * new_vessel:
            break
        else:
            mem_count_vessel = np.sum(mask_current_bif)
            mem_count_background = np.sum(SE) - mem_count_vessel

    return out_bif_mask


def compute_relative_degree(
    id_landmark: int,
    graph: CGraph,
    y_true: MetaTensor | np.ndarray,
    patch_pos: tuple[tuple],
    save: SaveImage | tuple[SaveImage | dict] = None,
) -> int:
    """
    Compute the relative connectivity of a landmark given a specific patch

    Args:
        id_landmark : The id of the landmark.
        graph       : The vascular graph.
        y_true      : The vessel mask (H,W,[D]).
        patch_pos   : The [ (start, ), (end, ) ] coordinate of the patch
        save        : The SaveImage object or a tuple (SaveImage, metadata) to save intermediate results. `None` as default : will not save intermediate results.


    Returns:
        The relative degree of a landmark
    """

    bifurcation = graph.nodes[id_landmark]

    logger.info(f"Compute relative degree for {bifurcation} in patch {patch_pos}")

    # Get the bifurcation mask
    # \ . . . _ . . . /
    # . \ . / . \ . / .
    # . . / . . . \ . .
    # . . | \ . / | . .
    # . . \ . O . / . .
    # . . . \ _ / . . .
    # . . . . | . . . .
    M_bif = create_mask_bifurcation(y_true, bifurcation)
    not_M_bif = M_bif == 0  # Exclude the masked zone.

    # Create the centerlines image -> draw all centerlines connected to the landmark
    # \ . . . . . . . /
    # . \ . . . . . / .
    # . . \ . . . / . .
    # . . . \ . / . . .
    # . . . . | . . . .
    # . . . . | . . . .
    # . . . . | . . . .
    I_skel = np.zeros_like(M_bif)

    for connection in [
        cnx
        for cnx in graph.connections.values()
        if cnx.node1._id == bifurcation._id or cnx.node2._id == bifurcation._id
    ]:
        for skvx in connection.skeleton_points:
            I_skel[skvx["pos"]] = 1

    # Disconnect the centerlines by removing the interconnection, depicted by our bifurcation mask
    # \ . . . . . . . /
    # . \ . . . . . / .
    # . . . . . . . . .
    # . . . . . . . . .
    # . . . . . . . . .
    # . . . . | . . . .
    # . . . . | . . . .
    disconnected_skel = I_skel * not_M_bif

    # Get the considered patch only
    I_skel_patch = disconnected_skel[
        patch_pos[0][0] : patch_pos[1][0],
        patch_pos[0][1] : patch_pos[1][1],
        patch_pos[0][2] : patch_pos[1][2],
    ]

    # Labelize the patch : the number of labels = number of remaining disconnected centerlines in the patch, e.g. the patch includes the entire bifurcation
    _, relative_degree = label(I_skel_patch, connectivity=None, return_num=True)

    logger.info(f"Degree : {bifurcation.degree} -> {relative_degree}")

    if save is not None:

        if isinstance(save, SaveImage):
            saver = save
            meta_data = None
        elif isinstance(save, tuple):
            saver = save[0]
            meta_data = save[1]

        # if os.path.isdir(saver.folder_layout.output_dir) == False:
        #     os.mkdir(saver.folder_layout.output_dir)

        # Image scale
        saver.folder_layout.postfix = f"mask_biff_{id_landmark}"
        saver(np.expand_dims(M_bif, axis=0), meta_data=meta_data)
        saver.folder_layout.postfix = f"skel_biff_{id_landmark}"
        saver(np.expand_dims(I_skel, axis=0), meta_data=meta_data)
        saver.folder_layout.postfix = f"skel_exclude_mask_biff_{id_landmark}"
        saver(np.expand_dims(disconnected_skel, axis=0), meta_data=meta_data)

        # Patch scale
        saver.folder_layout.postfix = f"patch_ytrue_{id_landmark}"
        saver(
            np.expand_dims(
                y_true[
                    patch_pos[0][0] : patch_pos[1][0],
                    patch_pos[0][1] : patch_pos[1][1],
                    patch_pos[0][2] : patch_pos[1][2],
                ],
                axis=0,
            ),
            meta_data=meta_data,
        )

        saver.folder_layout.postfix = f"patch_skel_exclude_mask_biff_{id_landmark}"
        saver(np.expand_dims(I_skel_patch, axis=0), meta_data=meta_data)

    return relative_degree


def main():
    vessel_mask_path = args.vessel_mask
    vessel_graph_path = args.vessel_graph
    patch_pos_path = args.patch_position

    id_node = args.id_landmark

    # Load all inputs
    I, meta = LoadImage(ensure_channel_first=False, image_only=False)(vessel_mask_path)

    vessel_graph = LoadVesselGraph(vessel_graph_path)
    vessel_graph = Anatomic2ImageGraph(vessel_graph, meta["original_affine"])

    patch_pos = read_path_position_from_file(patch_pos_path)

    save = SaveImage(
        output_dir=os.path.join(cfg.result_dir, "OUT_DEBUG"),
        output_ext=".nii.gz",
        output_postfix="",  # Defined later
        resample=False,
        separate_folder=False,
        output_dtype=np.uint8,  # All data are binary
    )

    compute_relative_degree(id_node, vessel_graph, I, patch_pos, save=(save, meta))


if __name__ == "__main__":
    from utils.configuration import Configuration

    cfg = Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    # Logger
    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": f"{cfg.log_dir}/relative_connectivity_node_{args.id_landmark}.log",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    logger.debug(args)

    main()
