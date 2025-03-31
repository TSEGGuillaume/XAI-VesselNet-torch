import logging

from graph.graph import CGraph, CNode

from ast import literal_eval as make_tuple

logger = logging.getLogger("app")


def get_landmark_obj(graph: CGraph, landmark_type: str=None, landmark_id: int|tuple|str=None) -> tuple[int]|None:
    """
    Get the landmark object depending on the provided informations.

    Args:
        graph           : The graph
        landmark_type   : The type of the landmark. `None` by default 
        landmark_id     : The ID of the landmark. `None` by default 

    Returns:
        The position of the landmark. `None` if no landmark_type and landmark_id provided
    """
    log = logger.debug

    if landmark_type == None and landmark_id == None:
        return None

    if landmark_type == "node":
        landmark = graph.nodes[int(landmark_id)]

        log(landmark)

    elif landmark_type == "centerline":
        landmark = graph.connections[int(landmark_id)]
        
        # Save a few information about the centerline for logging
        _centerline_id = landmark._id
        _centerline_node1 = landmark.node1._id
        _centerline_node2 = landmark.node2._id

        landmark = landmark.getMidPoint()

        log(
            "_{}_ |{}<->{}| - Skeleton voxel : {}".format(
                _centerline_id, _centerline_node1, _centerline_node2, landmark.pos
            )
        )

    elif landmark_type == "position":
        # In this case, the landmark id is its position ! 
        if not isinstance(landmark_id, tuple):
            if isinstance(landmark_id, str):
                landmark_id = make_tuple(landmark_id)
            else:
                raise TypeError(
                    f"Landmark ID must be a tuple (or literal tuple), not a {type(landmark_id)}"
                )

        landmark = CNode(-1, landmark_id, -1)

        log(f"Raw position: {landmark.pos}")

    else:
        landmark = None  # TODO : Manage the case where no position provided -> https://captum.ai/tutorials/Segmentation_Interpret
        
        logger.info(
            "No logit provided. Computation of the gradients on the whole prediction..."
        )

    return landmark