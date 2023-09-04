import argparse
import os
import logging
import time

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
        output_postfix = f"ig_node_{args.node}" 

        logger.info(obs_pt)

    elif args.centerline is not None:
        centerline = vessel_graph.connections[args.centerline]
        obs_pt = centerline.getMidPoint()

        output_postfix = f"ig_skvx_{args.centerline}"

        logger.info("_{}_ |{}<->{}| - Skeleton voxel : {}".format(centerline._id, centerline.node1._id, centerline.node2._id, obs_pt.pos))
        
    elif args.position is not None:
        obs_pt = None # TODO
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
    baseline = None # None : full-black volume 
    sw_shape = (in_channels, 64, 64, 64) # Channel dim must be specified ; spatial dim must be the same as that of the inference 
    sw_overlap = (0.0, 0.25, 0.25, 0.25) # must be the same as that of the inference 
    patches = iter_patch(I.numpy(), sw_shape, overlap=sw_overlap, mode="constant") # I don't know why iter_patch is crashing if Tensor instead of np.ndarray 

    attributions = []
    positions = []

    stime = time.time()

    for patch, pos in patches:
        if (
            obs_pt.pos[0] >= pos[1,0] and obs_pt.pos[0] < pos[1,1] and 
            obs_pt.pos[1] >= pos[2,0] and obs_pt.pos[1] < pos[2,1] and 
            obs_pt.pos[2] >= pos[3,0] and obs_pt.pos[2] < pos[3,1]
        ):
            # Compute targeted node in the patch coordinate system
            target = (0, obs_pt.pos[0]-pos[1,0], obs_pt.pos[1]-pos[2,0], obs_pt.pos[2]-pos[3,0]) # Don't forget that we're targeting an output voxel! The output has only 1 channel
            logger.info(f"Relative target: {target}")

            patch = torch.from_numpy(np.expand_dims(patch, axis=0)).type(torch.FloatTensor).to(device) # TODO: improve cast and numpy<->torch
            patch.requires_grad = True # Not sure this is mandatory
            attribution = ig.attribute(patch, target=target, baselines=baseline, n_steps=n_steps)

            print("Attribution shape : ", attribution.shape, torch.sum(attribution))

            positions.append(np.copy(pos))
            attributions.append(attribution)
    
    etime = time.time()
    logger.info(
        f"Integrated Gradients map computed. Enalpsed time: {etime - stime} s"
    )

    # Compute the shape of fused attribution maps
    min_x = min([position[1,0] for position in positions]) # Could be moved directly to the above iter. in a future improvement 
    max_x = max([position[1,1] for position in positions]) # Could be moved directly to the above iter. in a future improvement 
    min_y = min([position[2,0] for position in positions]) # Could be moved directly to the above iter. in a future improvement 
    max_y = max([position[2,1] for position in positions]) # Could be moved directly to the above iter. in a future improvement 
    min_z = min([position[3,0] for position in positions]) # Could be moved directly to the above iter. in a future improvement 
    max_z = max([position[3,1] for position in positions]) # Could be moved directly to the above iter. in a future improvement 

    final_attribution = torch.zeros((in_channels, max_x-min_x, max_y-min_y, max_z-min_z)).to(device)
    
    for attr, pos in zip(attributions, positions):
        # The fusing method is addition ; chosen arbitrarily
        final_attribution[:, pos[1,0]-min_x:pos[1,1]-min_x, pos[2,0]-min_y:pos[2,1]-min_y, pos[3,0]-min_z:pos[3,1]-min_z] = torch.add(
            final_attribution[:, pos[1,0]-min_x:pos[1,1]-min_x, pos[2,0]-min_y:pos[2,1]-min_y, pos[3,0]-min_z:pos[3,1]-min_z],
            attr
        )
    output_fname_pos =  f"{os.path.splitext(os.path.basename(args.img_path))[0]}_{os.path.splitext(os.path.basename(args.weights))[0]}_{output_postfix}_pos"

    # Save
    saver = SaveImage(
        output_dir=cfg.result_dir,
        output_ext=".nii",
        output_postfix=f"{os.path.splitext(os.path.basename(args.weights))[0]}_{output_postfix}",
        resample=False,
        separate_folder=False,
        output_dtype=I.dtype
    )
    saver(final_attribution, meta_data=meta)

    with open(os.path.join(cfg.result_dir, output_fname_pos + ".txt"), "w") as f:
        f.write(f"{min_x};{max_x}\n{min_y};{max_y}\n{min_z};{max_z}")
    
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