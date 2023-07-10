import monai


_all_models = ["unet", "res-unet", "attention-unet"]


def instanciate_model(model_name: str, spatial_dims=3, in_channels=1, out_channels=1):
    model = None

    # Common hyperparameters
    channels = (16, 32, 64, 128)
    strides = (2, 2, 2)
    dropout = 0.15

    if model_name == _all_models[0]:  # unet
        model = monai.networks.nets.UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout,
            num_res_units=0,
        )

    elif model_name == _all_models[1]:  # res-unet
        num_res_units = 2
        model = monai.networks.nets.UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout,
            num_res_units=num_res_units,
        )

    elif model_name == _all_models[2]:  # attention-unet
        model = monai.networks.AttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout,
        )

    else:
        raise ValueError(
            "Other model not supported yet \n Supported models: {}".format(_all_models)
        )

    return model
