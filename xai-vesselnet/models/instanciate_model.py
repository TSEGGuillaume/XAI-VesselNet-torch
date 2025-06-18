import monai


_all_models = ["unet", "res-unet", "attention-unet", "swinunetr"]


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

    elif model_name == _all_models[3]: # swinunetr
        # On fixe IRCAD: (480, 352, 416) | Bullitt: (320, 384, 192), divisible par 64, pour un feature_size=24
        # L'input size doit être divisible par 32 pour matcher avec la couche la plus basse (bottleneck), voir ValueError("input image size (img_size) should be divisible by stage-wise image resolution.") https://docs.monai.io/en/1.1.0/_modules/monai/networks/nets/swin_unetr.html#SwinUNETR.__init__
        # TODO: img_size et feature_size sont codés en dur et à modifier à la main selon le dataset. Cette solution n'est pas viable ; réfléchir à une solution dynamique (fichier de cfg ?)
        model = monai.networks.nets.SwinUNETR(
            img_size=(64, 64, 64),
            # img_size=(128, 128, 128),
            in_channels=1,
            out_channels=1,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=24,
            # feature_size=48,
            norm_name='instance',
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3,
            downsample='merging'
        )

    else:
        raise ValueError(
            "Other model not supported yet \n Supported models: {}".format(_all_models)
        )

    return model
