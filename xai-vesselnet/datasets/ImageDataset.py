import numpy as np

from monai.data import ImageDataset, GridPatchDataset, IterableDataset
from monai.transforms import apply_transform


class ImageDatasetd(ImageDataset):
    """
    Dictionary-based monai.data.ImageDataset (https://docs.monai.io/en/stable/data.html#imagedataset).

    Args:
        image_files: list of image filenames.
        seg_files: if in segmentation task, list of segmentation filenames.
        labels: if in classification task, list of classification labels.
        transform: transform to apply to image arrays.
        seg_transform: transform to apply to segmentation arrays.
        label_transform: transform to apply to the label data.
        image_only: if True return only the image volume, otherwise, return image volume and the metadata.
        transform_with_metadata: if True, the metadata will be passed to the transforms whenever possible.
        dtype: if not None convert the loaded image to this data type.
        reader: register reader to load image file and metadata, if None, will use the default readers.
            If a string of reader name provided, will construct a reader object with the `*args` and `**kwargs`
            parameters, supported reader name: "NibabelReader", "PILReader", "ITKReader", "NumpyReader"
        args: additional parameters for reader if providing a reader name.
        kwargs: additional parameters for reader if providing a reader name.

    Notes:
    This class simply overwrites __getitem__ function to return a dict with the keys {"img"(, "seg", "meta_img", "meta_seg")} instead of an array.
    Allows to use monai.data.PatchIterd more easily (https://docs.monai.io/en/stable/data.html#patchiterd).

    More details: monai.data.ImageDataset (https://docs.monai.io/en/stable/data.html#imagedataset)

    """

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        if self.seg_files is None and self.image_only == True:
            return {
                "img": item 
            }
        
        out = {
            "img": item[0]
        }
        if self.seg_files is not None:
            out["seg"] = item[1]
            if self.image_only == False:
                out["img_meta"] = item[2]
                out["seg_meta"] = item[3]
        else:
            if self.image_only == False:
                out["img_meta"] = item[1]   

        return out
        

class BalancedGridPatchDataset(GridPatchDataset):
    """
    Yields patches from the data of an ImageDatasetd object.

    This class overwrites monai.data.GridPatchDataset (https://docs.monai.io/en/stable/data.html#gridpatchdataset) to handle class imbalance problem by over-representing the foreground in the generator.
    In addition, it does not yield full-black patches.

    Args:
        data: the data source to read image data from.
        patch_iter: converts an input image (item from dataset) into a iterable of image patches.
            `patch_iter(dataset[idx])` must yield a tuple: (patches, coordinates).
            see also: :py:class:`monai.data.PatchIter` or :py:class:`monai.data.PatchIterd`.
        foreground_ratio: the required foreground ratio, default to 0.5
        transform: a callable data transform operates on the patches.
        with_coordinates: whether to yield the coordinates of each patch, default to `False`.

    Note: IMO, a better way would be to create a custom patch iterator instead. 
    
    More details: monai.data.ImageDataset (https://docs.monai.io/en/stable/data.html#gridpatchdataset).

    """

    def __init__(
        self,
        data,
        patch_iter,
        foreground_ratio=0.5,
        transform = None,
        with_coordinates: bool = False,
    ) -> None:
        super().__init__(data, patch_iter, transform=transform, with_coordinates=with_coordinates)
        
        self.minimal_foreground_ratio   = foreground_ratio
        self.rt_foreground_ratio        = -1.0
        self.n_foreground_patches   = 0  # Shared within the entire ImageDataset ; what is better (dataset of individual volumes ?)
        self.n_patches              = 0  # Shared within the entire ImageDataset ; what is better (dataset of individual volumes ?)

    #@deprecated ; keep at the moment
    def __iter__(self):
        print("Deprecated __iter__ function")
        for image in IterableDataset.__iter__(self):
            for patch, *others in self.patch_iter(image):
                use_patch = False # State variable that controls the integration of the current patch into the generator

                if np.any(patch["img"]):
                    if self.data.seg_files is not None:
                        if np.any(patch["seg"]):
                            use_patch = True
                            self.n_foreground_patches += 1
                            self.n_patches += 1
                        elif self.rt_foreground_ratio > self.minimal_foreground_ratio:
                            use_patch = True
                            self.n_patches += 1

                        self.rt_foreground_ratio = (self.n_foreground_patches / self.n_patches) if self.n_patches != 0 else self.rt_foreground_ratio

                    else:
                        use_patch = True # Always use patch if no segmentations provided

                if use_patch == True:
                    out_patch = patch
                    if self.patch_transform is not None:
                        out_patch = apply_transform(self.patch_transform, patch, map_items=False)
                    if self.with_coordinates and len(others) > 0:  # patch_iter to yield at least 2 items: patch, coords
                        yield out_patch, others[0]
                    else:
                        yield out_patch


    def __iter__(self):
        for in_patch in super().__iter__():
            if self.with_coordinates:
                patch = in_patch[0] # If with_coordinates, an array is yielded instead of a dict, with [0]:dict, [1]:tensor of coordinates
            else:
                patch = in_patch

            use_patch = False # State variable that controls the integration of the current patch into the generator

            if np.any(patch["img"]):
                #if patch["seg"] is not None:
                if self.data.seg_files is not None:
                    if np.any(patch["seg"]):
                        use_patch = True
                        self.n_foreground_patches += 1
                        self.n_patches += 1
                    elif self.rt_foreground_ratio > self.minimal_foreground_ratio:
                        use_patch = True
                        self.n_patches += 1
                    
                    self.rt_foreground_ratio = (self.n_foreground_patches / self.n_patches) if self.n_patches != 0 else self.rt_foreground_ratio
                
                else:
                    use_patch = True # Always use patch if no segmentations provided

            if use_patch == True:
                use_patch = False
                out_patch = in_patch

                yield out_patch
