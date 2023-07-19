XAI-VesselNet-torch : Explainable AI for deep vessel segmentation
===============

This work is a part of a PhD thesis, "[Détection faiblement supervisée de pathologies vasculaires](https://www.theses.fr/en/s307470#)" (Weakly supervised detection of vascular pathologies).


# Warnings
* This is a work in progress.
* This work was tested on Windows using a Quadro RTX 8000 GPU.

# Installation
1. Clone this repository in your workspace and move into it.
2. Set __PYTHONPATH__ environment variable :<br>
    Windows (this will only affect the current user's environment):<br>
    > setx PYTHONPATH "$Env:PYTHONPATH;`<workspace_path>`\XAI-VesselNet-torch\xai-vesselnet".

    Linux (preferably in the .bashrc):<br>
    > export PYTHONPATH="`<workspace_path>`/XAI-VesselNet-torch/xai-vesselnet:$PYTHONPATH"

3. Create the requiered conda environment to execute __XAI-VesselNet-torch__
> conda env create -f environment.yml

NB : Replace the `<workspace_path>` aliases by your own path.

# Models
We integrated only a few classical models from the [MONAI](https://github.com/Project-MONAI/MONAI) framework.

Model | Arg | Description
--- | --- | ---
U-Net | `unet` | [Basic U-Net](https://docs.monai.io/en/stable/networks.html#unet). Downsampling are perform by stride, not pooling. 
Residual U-Net | `res-unet` | [Residual U-Net](https://docs.monai.io/en/stable/networks.html#unet)
Attention U-Net | `attention-unet` | [Attention U-Net](https://docs.monai.io/en/stable/networks.html#attentionunet)

# Usage
First, activate the freshly created environment.
> conda activate -n xai_vesselnet
## Train a model
> python ./bin/train.py [--hyperparameters HYPERPARAMETERS_JSON_PATH] MODEL TRAIN_CSV_PATH VAL_CSV_PATH
## Evaluate a model
> python ./bin/eval.py [--mask MASK_CSV_PATH] MODEL WEIGHTS_PATH EVAL_CSV_PATH
## Infer a data
> python ./bin/infer.py MODEL WEIGHTS_PATH INFER_CSV_PATH
## Build a CSV dataset
We provides an easy way to define your own datasets. XAI-VesselNet-torch's scripts require CSV dataset.
CSV annotations should contain pairs of image;annotation, e.g.
```
<absolute_path_to_image>/image_1.nii;<absolute_path_to_seg>/seg_image_1.nii
<absolute_path_to_image>/image_2.nii;<absolute_path_to_seg>/seg_image_2.nii
...
<absolute_path_to_image>/image_n.nii;<absolute_path_to_seg>/seg_image_n.nii
```