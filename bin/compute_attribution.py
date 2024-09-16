import argparse
import os
import logging
import time

import torch
import numpy as np

from torch.nn import Sequential, Softmax, Sigmoid
from monai.transforms import LoadImage, SaveImage
from monai.data.utils import iter_patch

from captum.attr import IntegratedGradients, InputXGradient

import models.instanciate_model
from graph.voreen_parser import voreen_VesselGraphSave_file_to_graph as LoadVesselGraph
from utils.coordinates import anatomic_graph_to_image_graph as Anatomic2ImageGraph
from utils.load_hyperparameters import load_hyperparameters
from utils.prebuilt_logs import log_hardware
from utils.get_landmark_from_args import get_landmark_obj
from network.model_creator import init_inference_model


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        metavar=("MODEL_NAME"),
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
        help="Path to the image (*.nii | *.nii.gz)",
    )
    parser.add_argument(
        "graph_path",
        type=str,
        metavar=("GRAPH_PATH"),
        help="Path to the graph (*.vvg)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--node",
        "-n",
        type=int,
        metavar=("ID_NODE"),
        help="Id of the node to inspect",
        default=None,
    )
    group.add_argument(
        "--centerline",
        "-c",
        type=int,
        metavar=("ID_CENTERLINE"),
        help="Id of the centerline to inspect",
        default=None,
    )
    group.add_argument(
        "--position",
        "-pos",
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        help="Image coordinates X Y Z of the voxel to inspect",
        default=None,
    )

    parser.add_argument(
        "--hyperparameters",
        "-p",
        type=str,
        metavar=("HYPERPARAMETERS_JSON_PATH"),
        default=os.path.join(
            cfg.workspace, "resources", "default_hyperparameters.json"
        ),
        help="Path to the hyperparameters JSON",
    )

    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()
    return args


def main():
    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_hardware(device)

    # Variables
    model_name      = args.model
    weights_path    = args.weights
    x_path          = args.img_path
    graph_path      = args.graph_path
    hyperparameters_path  = args.hyperparameters

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

    is_patch = hyperparameters["patch"]

    # Image
    I, meta = LoadImage(image_only=False, ensure_channel_first=True)(x_path)

    # Observation point
    vessel_graph = LoadVesselGraph(graph_path)
    vessel_graph = Anatomic2ImageGraph(vessel_graph, meta["original_affine"])

    landmark = get_landmark_obj(graph=vessel_graph, landmark_type=landmark_type, landmark_id=landmark_id)

    # Load the trained model
    model = init_inference_model(
        model_name=model_name,
        weights_path=weights_path,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
    )

    # I'm not convinced we should activate the logit, but if we don't, are we back to the single-channel problem? 
    if out_channels==1:
        act = Sigmoid()
    elif out_channels==2:
        act = Softmax(dim=1)
    else:
        raise ValueError("Non-binary models are not allowed")
    model = Sequential(model, act)

    # Explanation methods
    mapping = {
        "IntegratedGradients": IntegratedGradients(model),
        "InputXGradient": InputXGradient(model),
    }
    kwargs = {
        # Inputs and targets are directly given in the call of attribute
        "IntegratedGradients": {
            "baselines": None,  # use zero scalar corresponding to each input tensor
            "n_steps": 50,
        },
        "InputXGradient": {
            # No more parameters than input and target. See attribute()
        },
    }

    # Prepare the output
    shared_output_dir = os.path.join(cfg.result_dir, "attributions")

    save = SaveImage(
        output_dir=shared_output_dir,
        output_ext=".nii.gz",
        output_postfix="", # Defined dynamically in the loop
        resample=False,
        separate_folder=False,
        output_dtype=I.dtype,
    )

    if is_patch:
        sw_shape    = [in_channels] + hyperparameters["input_shape"] # All dimensions should be specified
        sw_overlap  = [0] + [hyperparameters["patch_overlap"]] * len(hyperparameters["input_shape"])

        patches = iter_patch(I.numpy(), patch_size=sw_shape, overlap=sw_overlap, mode="constant")
        idx_involved_patch = 0

        for patch, pos in patches:
            if (
                landmark.pos[0] >= pos[1, 0]
                and landmark.pos[0] < pos[1, 1]
                and landmark.pos[1] >= pos[2, 0]
                and landmark.pos[1] < pos[2, 1]
                and landmark.pos[2] >= pos[3, 0]
                and landmark.pos[2] < pos[3, 1]
            ):
                patch = (
                    torch.from_numpy(np.expand_dims(patch, axis=0))
                    .type(torch.FloatTensor)
                    .to(device)
                )  # TODO: improve this conversion
                patch.requires_grad = True  # Not sure this is requiered

                logger.debug(f"Patch shape: {patch.shape} - {pos.tolist()}")

                # Compute targeted landmark in the patch coordinate system
                for idx_output_channel in range(out_channels):
                # For outputs with > 2 dimensions, targets can be either:
                #  - A single tuple, which contains #output_dims - 1 elements. This target index is applied to all examples.
                #  - A list of tuples with length equal to the number of examples in inputs (dim 0), and each tuple containing #output_dims - 1 elements.
                #     Each tuple is applied as the target for the corresponding example.
                    target = (
                        idx_output_channel,
                        landmark.pos[0] - pos[1, 0],
                        landmark.pos[1] - pos[2, 0],
                        landmark.pos[2] - pos[3, 0],
                    )
                    logger.info(f"Relative target: {target}")

                    for xai_key, xai_method in mapping.items():

                        stime = time.time()
                        attribution = xai_method.attribute(
                            inputs=patch, target=target, **kwargs[xai_key]
                        )
                        etime = time.time()
                        logger.info(
                            f"{xai_key}: Finished. \tEnalpsed time: {etime - stime}s"
                        )

                        logger.debug(
                            "Attribution shape : {} | sum: {}".format(
                                attribution.shape, torch.sum(attribution)
                            )
                        )

                        # Saving the attribution of the patch
                        output_fname_pos = f"{os.path.basename(x_path).split('.')[0]}_{os.path.splitext(os.path.basename(weights_path))[0]}_{xai_key}_{landmark_type}_{landmark_id}_{idx_involved_patch}_pos"
                        basic_postfix = f"{os.path.splitext(os.path.basename(weights_path))[0]}_{xai_key}_{landmark_type}_{landmark_id}_{idx_involved_patch}_ochan{idx_output_channel}"

                        # Attribution channels are saved separately
                        for idx_input_channel in range(in_channels):
                            save.folder_layout.postfix = (
                                basic_postfix + f"_ichan{idx_input_channel}"
                            )  # Update the postfix to add index for channel
                            save(attribution[0, idx_input_channel, :, :, :], meta_data=meta)

                        # We could have avoided saving the position file for each attribution method, but it's easier to associate an attribution with a position if they share the same basename.
                        with open(
                            os.path.join(shared_output_dir, output_fname_pos + ".txt"), "w"
                        ) as f:
                            f.write(
                                f"{pos[1,0]};{pos[1,1]}\n{pos[2,0]};{pos[2,1]}\n{pos[3,0]};{pos[3,1]}"
                            )

                idx_involved_patch += 1
    
    else:
        for idx_output_channel in range(out_channels):
            # For outputs with > 2 dimensions, targets can be either:
            #  - A single tuple, which contains #output_dims - 1 elements. This target index is applied to all examples.
            #  - A list of tuples with length equal to the number of examples in inputs (dim 0), and each tuple containing #output_dims - 1 elements.
            #     Each tuple is applied as the target for the corresponding example.

            # TODO : vérifier qu'aucun padding ne soit nécessaire. Sinon il faut calculer la position en tenant compte du padding.
            # Peut-être que le plus simple sera alors de padding non symétrique pour ne rien changer sur la position
            target = (
                idx_output_channel,
                landmark.pos[0],
                landmark.pos[1],
                landmark.pos[2],
            )
            logger.info(f"Relative target: {target}")

            for xai_key, xai_method in mapping.items():

                stime = time.time()
                attribution = xai_method.attribute(
                    inputs=I, target=target, **kwargs[xai_key] # TODO : vérifier si I a besoin de la dimension batch ou non ?
                )
                etime = time.time()
                logger.info(
                    f"{xai_key}: Finished. \tEnalpsed time: {etime - stime}s"
                )

                logger.debug(
                    "Attribution shape : {} | sum: {}".format(
                        attribution.shape, torch.sum(attribution)
                    )
                )

                # Saving the attribution of the patch
                basic_postfix = f"{os.path.splitext(os.path.basename(weights_path))[0]}_{xai_key}_{landmark_type}_{landmark_id}_ochan{idx_output_channel}"

                # Attribution channels are saved separately
                for idx_input_channel in range(in_channels):
                    save.folder_layout.postfix = (
                        basic_postfix + f"_ichan{idx_input_channel}"
                    )  # Update the postfix to add index for channel
                    save(attribution[0, idx_input_channel, :, :, :], meta_data=meta)


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
