import logging
import logging.config # Mandatory if MONAI is not imported

import os

logger = logging.getLogger("app")

class CXAIVesselNetFilename:
    def __init__(
        self,
        filename: str = None,
        dataset_id: str = None,
        sample_id: int|str = None,
        training_strategy: int|str = None,
        model_id: str = None,
        attribution_method: str = None,
        landmark_type: str = None,
        landmark_id: int|str = None,
        patch_id: int|str = None,
        output_channel_id: int|str = None,
        input_channel_id: int|str = None,
    ):
        """
        Class to contain input/output XAI-VesselNet files.
        This class allows to:
            1/ retrieves data information from a filename (default). If filename is provided, other arguments are ignored.
            2/ construct a filename from data information passed as argument data.

        Note:
            1/ This class is able to manage both attribution map filename, or the associated position file, such as
                - `3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_ochan0_ichan0.nii.gz`
                - `3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_pos.txt`
            2/ We've made the choice to inspire from the Medical Segmentation Decathlon for dataset naming convention (sample id, image modality, ect.). As so, the sample id is formatted with 3 digit (e.g. 009) and training strategy with 4 digit (if `int`) 

        Args:
            filename : The path (filename or full path) of the file
            -------------------------
            dataset_id          : The ID of the dataset (e.g. 3Dircadb1)
            sample_id           : The ID of the sample (e.g. 009)
            training_strategy   : The ID of the training strategy (e.g. 0000)
            model_id            : The ID of the model (e.g. 20230713-144625)
            attribution_method  : The name the attribution method (e.g. Saliency))
            landmark_type       : The type of the landmark (e.g. centerline)
            landmark_id         : The ID of the landmark (e.g. 0)
            patch_id            : The ID of the patch (e.g. 0)
            output_channel_id   : The ID of the output channel (e.g. 0). `None` by default
            input_channel_id    : The ID of the input channel (e.g. 0). `None` by default
        """
        components_provided = all(arg_v is not None for arg_v in [
                dataset_id,
                sample_id,
                training_strategy,
                model_id,
                attribution_method,
                landmark_type,
                landmark_id,
                patch_id,
            ]
        ) # We do not check for output_channel_id and input_channel_id, as one can be interessed on the prefix only

        if filename is not None:
            if components_provided:
                logger.warning(
                    "Both filename and components have been provided. Components are ignored."
                )

            self.basename = os.path.normpath(filename).split(os.sep)[-1]

            self._ensure_no_file_extension()

            self._retrieves_filename_elements()

        else:
            if components_provided:
                # If prefix model_, ochan and ichan are passed, delete them to preserve only IDs
                model_id = model_id.replace("model_", "")

                if output_channel_id is not None and isinstance(output_channel_id, str):
                    output_channel_id = output_channel_id.replace("ochan", "")
                if input_channel_id is not None and isinstance(input_channel_id, str):
                    input_channel_id = input_channel_id.replace("ichan", "")

                self.dataset_id = dataset_id
                self.sample_id = "{:03}".format(sample_id)
                self.training_strategy = "{:04}".format(training_strategy)
                self.model_id = model_id
                self.attribution_method = attribution_method
                self.landmark_type = landmark_type
                self.landmark_id = landmark_id
                self.patch_id = patch_id
                self.output_channel_id = output_channel_id
                self.input_channel_id = input_channel_id

                self.basename = self._create_filename_from_elements()
            
                self._ensure_no_file_extension()

            else:
                raise ValueError("All filename components must be provided.")


    def _ensure_no_file_extension(self, idx_to_keep: int|slice = 0):
        """
        Ensure the basename of the file does not contains file extension.
        Delete the file extension if present. Define member variable `ext` to store the extension.

        Args:
            idx_to_keep: Indicate which index/indices to consider as the filename (e.g. without file extension). By default, the left part before the first dot will be considered as the basename.
        """ 
        self.ext  = None

        if "." in self.basename:

            fname_split = self.basename.split(".")

            working_filename = fname_split[idx_to_keep]

            if isinstance(idx_to_keep, slice):
                # In case of slice, a list is returned -> we have to rejoin into a single str
                assert isinstance(working_filename, list), "We are doomed"
                working_filename = ".".join(working_filename)

                idx_to_keep = idx_to_keep.stop
            else:
                # int
                idx_to_keep += 1

            self.ext = self.basename[idx_to_keep:] # Get the file extension
            self.basename = working_filename


    def _retrieves_filename_elements(self):
        """
        Define the individual components of the filename from the basename.
        Note: in a normal case, a file name is either :
        - `3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_ochan0_ichan0.nii.gz`, or associated position file
        - `3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_pos.txt`
        """
        # Example of a filename 3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_ochan0_ichan0.nii.gz
        splitted_fname = self.basename.split("_")

        self.dataset_id = splitted_fname[0]
        self.sample_id = splitted_fname[1]
        self.training_strategy = splitted_fname[2]
        self.model_id = splitted_fname[4]
        self.attribution_method = splitted_fname[5]
        self.landmark_type = splitted_fname[6]
        self.landmark_id = splitted_fname[7]
        self.patch_id = splitted_fname[8]

        if self.ext == "nii" or self.ext == "nii.gz":
            self.output_channel_id = splitted_fname[9].replace('ochan', '')
            self.input_channel_id = splitted_fname[10].replace('ichan', '')
        else:
            # It's a .txt position file, with suffix _pos that we don't care
            self.output_channel_id = None
            self.input_channel_id = None
    

    def _create_filename_from_elements(self):
        """
        Create the filename from the individual components.
        Note: in a normal case, the filename should be composed as follows:
        `{PREFIX}_ochan{OUTPUT_CHANNEL_ID}_ichan{INPUT_CHANNEL_ID}`

        See. `prefix()` for details on prefix composition
        """
        prefix = self.get_prefix()

        if self.output_channel_id is not None and self.input_channel_id is not None:
            return "_".join([
                        prefix,
                        f"ochan{self.output_channel_id}",
                        f"ichan{self.input_channel_id}"
                    ])
        else:
            logger.warning(
                f"Output and/or input channels are {self.output_channel_id}|{self.input_channel_id}. Returns prefix."
            )
            return prefix


    def get_prefix(self) -> str:
        """
        Create the file name prefix from the individual components.
        Note: in a normal case, the prefix should be composed as follows:
        `{DATASET_ID}_{SAMPLE_ID}_{TRAINING_STRATEGY}_model_{MODEL_ID}_{ATTRIBUTION_METHOD}_{LANDMARK_TYPE}_{LANDMARK_ID}_{PATCH_ID}`

        Returns:
            self.prefix (str)
        """
        return "{}_{}_{}_model_{}_{}_{}_{}_{}".format(
            self.dataset_id,
            self.sample_id,
            self.training_strategy,
            self.model_id,
            self.attribution_method,
            self.landmark_type,
            self.landmark_id,
            self.patch_id
        )


    def get_filename(self) -> str:
        """
        Return the basename of the file

        Returns:
            self.basename (str) 
        """
        return self.basename
    

def main():
    logging.debug("Hello world !")

    path = "/home/gugarret/workspace/XAI-VesselNet-torch/results/attributions/3Dircadb1_0000_attr_Saliency/"

    obj = CXAIVesselNetFilename(os.path.join(path, "3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_ochan0_ichan0.nii.gz"))
    print(obj.get_filename())
    obj = CXAIVesselNetFilename(os.path.join(path, "3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_ochan0_ichan0.nii"))
    print(obj.get_filename())
    obj = CXAIVesselNetFilename(os.path.join(path, "3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_ochan0_ichan0"))
    print(obj.get_filename())
    obj = CXAIVesselNetFilename(os.path.join(path, "3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_pos.txt"))
    print(obj.get_filename(), "vs.", obj.get_prefix())
    
    obj = CXAIVesselNetFilename(None, "3Dircadb1", "009", "0000", "model_20230713-144625", "Saliency", "centerline", "0", "0", "ochan0", "ichan0")
    print(obj.get_filename())                      
    obj = CXAIVesselNetFilename(None, "3Dircadb1", "009", "0000", "20230713-144625", "Saliency", "centerline", "0", "0", "0", "0")
    print(obj.get_filename())
    obj = CXAIVesselNetFilename(None, "3Dircadb1", 9, 0, "20230713-144625", "Saliency", "centerline", 0, 0, 0, 0)
    print(obj.get_filename())
    obj = CXAIVesselNetFilename(None, "3Dircadb1", 9, 0, "20230713-144625", "Saliency", "centerline", 0, 0)
    print(obj.get_filename())
    obj = CXAIVesselNetFilename(None, "3Dircadb1", 9, "multichannels", "20230717-162426", "Saliency", "centerline", 0, 0, 0, 6)
    print(obj.get_filename())

    obj = CXAIVesselNetFilename(os.path.join(path, "3Dircadb1_009_0000_model_20230713-144625_Saliency_centerline_0_0_ochan0_ichan0.nii.gz"))
    print(obj.get_filename())

    obj = CXAIVesselNetFilename(None, "3Dircadb1", "009", "0000", "model_20230713-144625", "Saliency", "centerline", "0", "0", "ochan0", "ichan0.nii.gz")
    print(obj.get_filename())


if  __name__ == "__main__":
    main()