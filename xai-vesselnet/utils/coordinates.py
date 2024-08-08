import numpy as np

from graph.graph import CGraph, CNode, CCenterline


def image_to_anatomic(image_pos: tuple, affine: np.array) -> tuple:
    """
    Transforms an image position to anatomic position.
    Notes: coordinate systems in medical imaging, see https://www.slicer.org/wiki/Coordinate_systems

    TODO: deal with the origin to obtain the actual position of the patient.

    Args:
        image_pos   : The image position.
        affine      : The transformation matrix (4x4).


    Returns:
        The anatomic position
    """

    image_pos_homo = image_pos + (1,)  # Make vector homogeneous with affine
    return tuple(np.matmul(affine, image_pos_homo)[:-1])


def anatomic_graph_to_image_graph(graph: CGraph, affine: np.array) -> CGraph:
    """
    Transform a graph that uses anatomic system into a graph that uses image system.
    Notes: coordinate systems in medical imaging, see https://www.slicer.org/wiki/Coordinate_systems

    Args:
        graph   : The graph with anatomical coordinates.
        affine  : The transformation matrix (4x4).

    Returns:
        CGraph with image coordinates
    """

    image_system_nodes          = []
    image_system_connections    = []

    inv_affine = np.linalg.inv(affine)

    for node in graph.nodes.values():
        anatomic_pos = np.array([node.pos[0], node.pos[1], node.pos[2], 1])  # Get anatomic position and make vector homogeneous with affine
        image_pos = np.round(np.matmul(inv_affine, anatomic_pos)).astype(int)

        image_system_nodes.append(
            CNode(node._id, (image_pos[0], image_pos[1], image_pos[2]), node.degree)
        )

    # Skeleton points translation for all centerlines : from patient system to image system.
    # There is certainly something more elegant to do that...
    for connection in graph.connections.values():
        # If the connection is a CCenterline, the object has a skeleton_points member.
        # This member can be NoneType or an empty/populated list)
        if isinstance(connection, CCenterline):
            if connection.skeleton_points is not None:
                image_system_skeleton_points = []
                for sk_point in connection.skeleton_points:
                    anatomic_pos = np.array([sk_point["pos"][0], sk_point["pos"][1], sk_point["pos"][2], 1])
                    image_pos = np.round(np.matmul(inv_affine, anatomic_pos)).astype(int)

                    image_system_skeleton_points.append(
                        {
                            "pos": (image_pos[0], image_pos[1], image_pos[2]),
                            "avgDistToSurface": sk_point["avgDistToSurface"],
                        }
                    )

                image_system_connections.append(
                    CCenterline(connection._id, connection.node1, connection.node2, p_skeleton_points=image_system_skeleton_points)
                )
        else:
            raise RuntimeError("Graph connections are unknown type.")

    return CGraph(image_system_nodes, image_system_connections)
