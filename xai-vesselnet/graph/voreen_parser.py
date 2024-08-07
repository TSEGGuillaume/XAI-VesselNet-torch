import os

import json

from graph.graph import CGraph, CCenterline, CNode


def voreen_VesselGraphSave_file_to_graph(p_vessel_graph_file: str):
    """
    Create a graph of a vessel structure by parsing a VesselGraphSave output file (*.vvg ; see Voreen).

    Args:
        p_vessel_graph_file : The path to the VesselGraphSave output file.

    Returns:
        The CGraph of a vessel structure.
    """

    _, extension = os.path.splitext(p_vessel_graph_file)
    assert extension == ".vvg", "The graph file is not a VVG file."

    with open(p_vessel_graph_file) as graph_file:
        json_data = json.load(graph_file)
        graph_data = json_data["graph"]

        indexed_nodes = {}
        for node in graph_data["nodes"]:
            node_id     = node["id"]
            node_pos    = tuple(node["pos"])
            node_degree = len(node["edges"])

            indexed_nodes[node_id] = CNode(node_id, node_pos, node_degree)

        centerlines = []
        for centerline in graph_data["edges"]:
            centerline_id = centerline["id"]

            node1_id    = centerline["node1"]
            node2_id    = centerline["node2"]
            node1       = indexed_nodes[node1_id]
            node2       = indexed_nodes[node2_id]

            skeleton_points = []
            for point in centerline["skeletonVoxels"]:
                skeleton_points.append(
                    {
                        "pos": tuple(point["pos"]),
                        "avgDistToSurface": point["avgDistToSurface"],
                    }
                )

            centerlines.append(
                CCenterline(
                    centerline_id, node1, node2, p_skeleton_points=skeleton_points
                )
            )

    return CGraph(list(indexed_nodes.values()), centerlines)


#@deprecated ; keep at the moment
def voreen_VesselGraphGlobalStats_files_to_graph(
    p_nodes_file_path: str, p_centerlines_file_path: str
):
    """
    Create a graph of a vessel structure by parsing a VesselGraphGlobalStats output files (*.csv ; see Voreen).

    Parameters
        p_nodes_file_path (str) : The path to the VesselGraphGlobalStats nodes export file.
        p_centerlines_file_path (str) : The path to the VesselGraphGlobalStats edges export file.

    Returns
        CGraph of a vessel structure.
    """
    import csv

    filename_nodes_f, extension_nodes_f = os.path.splitext(p_nodes_file_path)
    filename_edges_f, extension_edges_f = os.path.splitext(p_centerlines_file_path)
    assert extension_nodes_f == ".csv" and extension_edges_f == ".csv"

    nodes = []
    with open(p_nodes_file_path, newline="") as nodes_csvfile:
        nodereader = csv.reader(nodes_csvfile, delimiter=";")

        for row in list(nodereader)[1:]:
            node_id = int(row[0])
            node_pos = (float(row[1]), float(row[2]), float(row[3]))
            node_degre = int(row[4])

            nodes.append(CNode(node_id, node_pos, node_degre))

    centerlines = []
    with open(p_centerlines_file_path, newline="") as centerlines_csvfile:
        centerlinereader = csv.reader(centerlines_csvfile, delimiter=";")

        for row in list(centerlinereader)[1:]:
            centerline_id = int(row[0])
            node1_id = int(row[1])
            node2_id = int(row[2])
            curveness = float(row[5])

            centerlines.append(
                CCenterline(centerline_id, node1_id, node2_id, p_curveness=curveness)
            )

    return CGraph(nodes, centerlines)
