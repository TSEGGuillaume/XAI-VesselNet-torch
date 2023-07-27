import argparse
import os
import logging

import torch
import numpy as np

from monai.data.utils import iter_patch
from monai.transforms import LoadImage, SaveImage

from captum.attr import IntegratedGradients

import models.instanciate_model
from models.instanciate_model import instanciate_model
from graph import voreen_parser
from utils import coordinates
from utils.logging_blocks import log_hardware


def parse_arguments():
    parser = argparse.ArgumentParser()

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
        "img_path",
        type=str,
        metavar=("IMG_PATH"),
        help="Path to the image (*.nii)",
    )
    parser.add_argument(
        "graph_path",
        type=str,
        metavar=("GRAPH_PATH"),
        help="Path to the graph (*.vvg)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--node', '-n', type=int, metavar=("ID_NODE"), help="id of the node to inspect", default=None)
    group.add_argument('--centerline', '-c', type=int, metavar=("ID_CENTERLINE"), help="id of the centerline to inspect", default=None)
    group.add_argument('--position', '-p', nargs=3, type=int, metavar=("X", "Y", "Z"), help="image coordinates (x,y,z) of the voxel to inspect", default=None)

    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args

def main():
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    # Image
    I = LoadImage(image_only=False, ensure_channel_first=True)(args.img_path)
    meta = I[1]
    I = I[0]

    in_channels = I.shape[0]
    logger.info(f"Image shape : {I.shape}")

    # Observation point
    vessel_graph = voreen_parser.voreen_VesselGraphSave_file_to_graph(args.graph_path)
    vessel_graph = coordinates.anatomical_graph_to_image_graph(vessel_graph, meta["original_affine"]) 

    if args.node is not None:
        obs_pt = vessel_graph.nodes[args.node]
        output_postfix = f"ig_node_{args.node}" # TODO : ajouter dataname et model_id au postfix

        logger.info(obs_pt)


    elif args.centerline is not None:
        obs_pt = None # TODO : Voir integrated_gradients.py TF
        logger.info("Centerline")

    elif args.position is not None:
        obs_pt = None # TODO : Voir integrated_gradients.py TF
        logger.info("Position")

    else:
        obs_pt = None # TODO : Gérer le cas où aucune position n'est fournit https://captum.ai/tutorials/Segmentation_Interpret
        logger.info("No logit provided. Computation of the gradients on the whole prediction...")

    # Model
    model = instanciate_model(args.model, in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(args.weights)["model_state_dict"])
    model.eval()

    # Integrated Gradients         
    ig = IntegratedGradients(model)
    n_steps = 100
    baseline = None

    sw_shape = (1, 64, 64, 64) # Channel dim must be specified ; spatial dim must be the same as that of the inference 
    sw_overlap = (0.0, 0.25, 0.25, 0.25) # must be the same as that of the inference 
    patches = iter_patch(I.numpy(), sw_shape, overlap=sw_overlap, mode="constant") # I don't know why iter_patch is crashing if Tensor instead of np.ndarray 

    attributions = []
    positions = []

    for patch, pos in patches:
        if (
            obs_pt.pos[0] > pos[1,0] and obs_pt.pos[0] < pos[1,1] and 
            obs_pt.pos[1] > pos[2,0] and obs_pt.pos[1] < pos[2,1] and 
            obs_pt.pos[2] > pos[3,0] and obs_pt.pos[2] < pos[3,1]
        ):
            # TODO : sw_chape passe en (7...)
            # Si bonne position
            # Pour k de 0 à n_channel : attribution = add(attribution, new_attrib)
            target = (pos[0,0], obs_pt.pos[0]-pos[1,0], obs_pt.pos[1]-pos[2,0], obs_pt.pos[2]-pos[3,0]) # Target should change channel
            logger.info(f"Relative target: {target}")

            positions.append(np.copy(pos))

            patch = torch.from_numpy(np.expand_dims(patch, axis=0)).type(torch.FloatTensor).to(device) # TODO: improve cast and numpy<->torch
            attribution = ig.attribute(patch, target=target, baselines=baseline, n_steps=n_steps)
            attributions.append(attribution)

    final_attribution = torch.zeros_like(I).to(device)
    print(final_attribution.shape)

    for attr, pos in zip(attributions, positions):
        stop_x = final_attribution.shape[1] if pos[1,0] + sw_shape[1] > final_attribution.shape[1] else pos[1,1]
        stop_y = final_attribution.shape[2] if pos[2,0] + sw_shape[2] > final_attribution.shape[2] else pos[2,1]
        stop_z = final_attribution.shape[3] if pos[3,0] + sw_shape[3] > final_attribution.shape[3] else pos[3,1]

        final_attribution[pos[0,0], pos[1,0]:pos[1,1], pos[2,0]:pos[2,1], pos[3,0]:pos[3,1]] = torch.add(
            final_attribution[pos[0,0], pos[1,0]:pos[1,1], pos[2,0]:pos[2,1], pos[3,0]:pos[3,1]],
            attr[0, 0, 0:stop_x-pos[1,0], 0:stop_y-pos[2,0], 0:stop_z-pos[3,0]]
        )

    # Save
    saver = SaveImage(
        output_dir=cfg.result_dir,
        output_ext=".nii",
        output_postfix=output_postfix,
        resample=False,
        separate_folder=False
    )
    saver(final_attribution )

if __name__ == "__main__":
    import utils.configuration as appcfg

    cfg = appcfg.Configuration(p_filename="./resources/default.ini")

    args = parse_arguments()

    logging.config.fileConfig(
        os.path.join(cfg.workspace, "resources", "logger.conf"),
        defaults={
            "filename": "{}/last_attribution.log".format(cfg.log_dir),
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    )
    logger = logging.getLogger("app")

    main()
