import os

import numpy as np

from monai.data import ImageDataset, IterableDataset
from monai.transforms import apply_transform


class CImageDatasetd(ImageDataset):
    """
    Dictionary-based monai.data.ImageDataset (ref. https://docs.monai.io/en/stable/data.html#imagedataset)
    
    Each item is a dictionnary with keys ["image", "mask", "fname"]

    Loads image/segmentation pairs of files from the given filename lists. Transformations can be specified
    for the image and segmentation arrays separately.

    """

    def __init__(
        self,
        image_files,
        seg_files=None,
        labels=None,
        transform=None,
        seg_transform=None,
        label_transform=None,
        image_only=True,
        transform_with_metadata=False,
        dtype=np.float32,
        reader=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            image_files=image_files,
            seg_files=seg_files,
            labels=labels,
            transform=transform,
            seg_transform=seg_transform,
            label_transform=label_transform,
            image_only=image_only,
            transform_with_metadata=transform_with_metadata,
            dtype=dtype,
            reader=reader,
            *args,
            **kwargs
        )

    def __len__(self):
        return super().__len__()

    def randomize(self, data=None):
        super().randomize()

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        head, tail = os.path.split(self.image_files[idx])

        # Dictionnarize the ImageDataset items
        if self.seg_files is not None:
            return {"image": item[0], "mask": item[1], "fname": tail}
        else:
            return {"image": item[0], "mask": None, "fname": tail}


class CCustomGridPatchDataset_(IterableDataset):
    """
    Yields patches from data read from an image dataset.

    The patches used by the generator manage class imbalance by over-representing the foreground through a balance factor.

    Overwritting of monai.data.GridPatchDataset (ref. https://docs.monai.io/en/stable/data.html#gridpatchdataset)

    Args:
        data: the data source to read image data from.
        patch_iter: converts an input image (item from dataset) into a iterable of image patches.
            `patch_iter(dataset[idx])` must yield a tuple: (patches, coordinates).
            see also: :py:class:`monai.data.PatchIter` or :py:class:`monai.data.PatchIterd`.
        transform: a callable data transform operates on the patches.
        foreground_ratio: the requierement foreground ratio.
        with_coordinates: whether to yield the coordinates of each patch, default to `True`.

    """

    def __init__(
        self,
        data,
        patch_iter,
        transform,
        foreground_ratio=0.5,
        with_coordinates: bool = True,
    ) -> None:
        super().__init__(data=data, transform=None)
        self.patch_iter = patch_iter
        self.patch_transform = transform
        self.with_coordinates = with_coordinates

        self.rt_foreground_ratio = -1.0
        self.required_foreground_ratio = foreground_ratio
        self.n_foreground_patches = 0  # Shared within the entire ImageDataset
        self.n_patches = 0  # Shared within the entire ImageDataset

    def __iter__(self):
        for image in super().__iter__():
            for patch, *others in self.patch_iter(image):
                out_patch = patch

                use_patch = False  # State variable that controls the integration of the current patch in the generator

                # Handles the class imbalance problem.
                # The current patch is pass to the generator with conditions -> 1 AND (2 OR 3)
                #
                # Conditions:
                # 1. There is a signal in the patch (e.g. not a patch from a masked area of the image)
                # 2. The patch integrates a positive target (i.e. positive voxels in the ground-truth)
                # 3. There is less background patches than expected (i.e. the required proportion of background patches is not respected)
                if np.any(out_patch["image"]):
                    if np.any(out_patch["mask"]):
                        use_patch = True
                        self.n_foreground_patches += 1
                        self.n_patches += 1
                        self.rt_foreground_ratio = (
                            self.n_foreground_patches / self.n_patches
                        )  # Not factorized, even if used in both cases, to avoid division by 0
                    elif self.rt_foreground_ratio > self.required_foreground_ratio:
                        use_patch = True
                        self.n_patches += 1
                        self.rt_foreground_ratio = (
                            self.n_foreground_patches / self.n_patches
                        )

                    if use_patch == True:
                        use_patch = False

                        if self.patch_transform is not None:
                            out_patch = apply_transform(
                                self.patch_transform, patch, map_items=False
                            )
                        if (
                            self.with_coordinates and len(others) > 0
                        ):  # patch_iter to yield at least 2 items: patch, coords
                            yield out_patch, others[0]
                        else:
                            yield out_patch
