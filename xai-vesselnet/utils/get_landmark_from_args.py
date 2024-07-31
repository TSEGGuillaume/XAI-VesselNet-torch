import logging

from graph.graph import CGraph


logger = logging.getLogger("app")


def get_landmark_position(graph: CGraph, landmark_type: str=None, landmark_id: int|tuple=None) -> tuple[int]|None:
    """
    Get the position of the provided landmark depending on its type.

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
        landmark_pos = landmark.pos

        logger.info(landmark)

    elif landmark_type == "centerline":
        landmark = graph.connections[landmark_id]
        landmark_pos = landmark.getMidPoint().pos

        logger.info(
            "_{}_ |{}<->{}| - Skeleton voxel : {}".format(
                landmark._id, landmark.node1._id, landmark.node2._id, landmark_pos
            )
        )

    elif landmark_type == "position":
        landmark_pos = landmark_id

        logger.info(f"Raw position: {landmark_pos}")

    else:
        landmark_pos = None  # TODO : Manage the case where no position provided -> https://captum.ai/tutorials/Segmentation_Interpret
        
        logger.info(
            "No logit provided. Computation of the gradients on the whole prediction..."
        )

    return landmark_pos