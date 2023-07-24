import numpy as np

from graph.graph import CGraph, CNode, CCenterline

def image_to_anatomical(image_pos: tuple, affine) -> tuple:
    """
    Transform an image position to anatomical position.

    Parameters
        image_pos (tuple)   : The image position.
        affine              : The transformation matrix (4x4).

    Notes:
        1. To understand coordinate systems in medical imaging, see https://www.slicer.org/wiki/Coordinate_systems

    Returns
        To anatomical position
    """

    image_pos_homo = image_pos + (1,)
    return tuple(np.matmul(affine, image_pos_homo)[:-1])

def anatomical_graph_to_image_graph(graph : CGraph, affine):
    """
    Transform a graph that uses anatomical system into a graph that uses image system.

    Parameters
        graph (CGraph)  : The graph with anatomical coordinates.
        affine          : The transformation matrix (4x4).

    Notes:
        1. To understand coordinate systems in medical imaging, see https://www.slicer.org/wiki/Coordinate_systems

    Returns
        CGraph with image coordinates
    """
 
    image_system_nodes = []
    image_system_connections = []
    
    inv_affine = np.linalg.inv(affine)
    
    for node in graph.nodes.values():
        anatomical_pos = np.array([node.pos[0], node.pos[1], node.pos[2], 1]) # Get anatomical position and make vector homogeneous
        image_pos = np.round(np.matmul(inv_affine, anatomical_pos)).astype(int)
        
        image_system_nodes.append(CNode(node._id, (image_pos[0], image_pos[1], image_pos[2]), node.degree))
        
    # Skeleton points translation (for all centerlines) : from patient system to image system.
    # There is certainly something more elegant to do that... 
    for connection in graph.connections.values():
        # If the connection is a centerline, the object has a skeleton_points member.
        # This member can be NoneType or an empty/populated list)
        if type(connection) == CCenterline:
            if connection.skeleton_points is not None:
                image_system_skeleton_points = []
                for sk_point in connection.skeleton_points:
                    anatomical_pos = np.array([sk_point["pos"][0], sk_point["pos"][1], sk_point["pos"][2], 1]) # Get anatomical position and make vector homogeneous
                    image_pos = np.round(np.matmul(inv_affine, anatomical_pos)).astype(int)
                
                    image_system_skeleton_points.append(
                        {
                         "pos": (image_pos[0], image_pos[1], image_pos[2]), 
                         "avgDistToSurface": sk_point["avgDistToSurface"]
                        }
                    )
                
                # In order to have the same behaviour that when we create the Graph by parsing file. 
                # Remember that node ID is replaced by the CNode itself in the CTOR. We have to go back to the ID representation to avoid crash.
                connection.node1 = connection.node1._id
                connection.node2 = connection.node2._id

                connection.skeleton_points = image_system_skeleton_points
                
                image_system_connections.append(connection)
            
    return CGraph(image_system_nodes, image_system_connections)