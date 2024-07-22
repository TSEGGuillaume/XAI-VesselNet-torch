# This script create multichannels images from multiple images modalities
# Input : Path to the dataset. Beware, the filenames of images should respect the nnUNet format {CASE_IDENTIFIER}_{XXXX}.{FILE_ENDING}, Hereby, XXXX is the 4-digit modality/channel identifier
# I.e : 3Dircadb1_001_0000.nii.gz

import argparse
import os

from collections import defaultdict
import numpy as np
from monai.transforms import LoadImage, SaveImage

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_path",
        type=str,
        metavar=("DATASET_PATH"),
        help="Path to the dataset directory ({CASE_IDENTIFIER}_{XXXX}.nii.gz)",
    )

    args = parser.parse_args()
    return args


def assert_even_modalities_count(multichannel_dict: dict):
    # Assert every case have the same count of modalities
    even_modalities_count = True
    _identifiers = list(multichannel_dict.keys())
    for idx_key in range(len(_identifiers[:-1])):
        even_modalities_count = even_modalities_count and (multichannel_dict[_identifiers[idx_key]]==multichannel_dict[_identifiers[idx_key+1]])
        
    assert even_modalities_count == True


def main(dataset_dir_path):
    fextension = ".nii.gz"

    files = [f.split('.')[0] for f in os.listdir(dataset_dir_path) if f.endswith(fextension)]
    
    multichannel_dict = defaultdict(lambda: [])

    for file in files:
        _ = file.split('_')
        case_identifier = '_'.join(_[:-1])
        modality = _[-1]

        multichannel_dict[case_identifier].append(modality)
    
    assert_even_modalities_count(multichannel_dict)

    # Stack
    loader = LoadImage(ensure_channel_first=True, image_only=False)
    saver = SaveImage(
        output_dir=os.path.join(dataset_dir_path, "multichannels"),
        output_postfix="multichannels",
        output_ext=".nii.gz",
        resample=False,
        separate_folder=False,
    )

    for k, v in multichannel_dict.items():
        full_paths = [os.path.join(dataset_dir_path, f"{k}_{channel}{fextension}") for channel in v]
        
        I_s, meta = loader(full_paths[0]) 

        meta["filename_or_obj"] = meta["filename_or_obj"].replace("_0000", "")

        for path in full_paths[1:]:
            I, cmeta = loader(path)

            # Minimal check
            assert np.all((cmeta["spatial_shape"] == meta["spatial_shape"])), f"Different spatial shape across modalities. {path}"
            assert np.all((cmeta["affine"] == meta["affine"]).numpy()), f"Different affines across modalities. {path}"
            assert np.all((cmeta["space"] == meta["space"])), f"Different space across modalities. {path}"

            I_s = np.concatenate([I_s, I], axis=0)
            print(I_s.shape)
        
        saver(img=I_s, meta_data=meta)


if __name__ == "__main__":
    args = parse_arguments()

    dataset_dir_path = args.dataset_path

    main(dataset_dir_path)