import logging

from graph.graph import CGraph, CNode


logger = logging.getLogger("app")


def get_landmark_obj(graph: CGraph, landmark_type: str=None, landmark_id: int|tuple=None) -> tuple[int]|None:
    """
    Get the landmark object depending on the provided informations.

    Args:
        graph           : The graph
        landmark_type   : The type of the landmark. `None` by default 
        landmark_id     : The ID of the landmark. `None` by default 

    Returns:
        The position of the landmark. `None` if no landmark_type and landmark_id provided
    """
    if landmark_type == None and landmark_id == None:
        return None

    if landmark_type == "node":
        landmark = graph.nodes[landmark_id]

        logger.info(landmark)

    elif landmark_type == "centerline":
        landmark = graph.connections[landmark_id]
        
        # Save a few information about the centerline for logging
        _centerline_id = landmark._id
        _centerline_node1 = landmark.node1._id
        _centerline_node2 = landmark.node2._id

        landmark = landmark.getMidPoint()

        logger.info(
            "_{}_ |{}<->{}| - Skeleton voxel : {}".format(
                _centerline_id, _centerline_node1, _centerline_node2, landmark.pos
            )
        )

    elif landmark_type == "position":
        # In this case, the landmark id is its position ! 
        landmark = CNode(-1, landmark_id, -1)

        logger.info(f"Raw position: {landmark.pos}")

    else:
        landmark = None  # TODO : Manage the case where no position provided -> https://captum.ai/tutorials/Segmentation_Interpret
        
        logger.info(
            "No logit provided. Computation of the gradients on the whole prediction..."
        )

    return landmark