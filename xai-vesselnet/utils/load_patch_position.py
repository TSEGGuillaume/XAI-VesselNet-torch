def read_path_position_from_file(patch_pos_path:str) -> list[tuple]:
    """
    Read the file that contains a patch position and return this position

    Args:
        patch_pos_path : Path to the position file

    Returns:
        patch_pos : The list of start and end positions (tuple) of the patch [ (start, ), (end, ) ]
    """
    start_roi   = ()
    end_roi     = ()

    with open(patch_pos_path, 'r') as f:
        lines = f.read().splitlines()

        for line in lines:
            spline = line.split(";")
            start_roi   += (int(spline[0]),)
            end_roi     += (int(spline[1]),)

    patch_pos = [start_roi, end_roi]

    return patch_pos