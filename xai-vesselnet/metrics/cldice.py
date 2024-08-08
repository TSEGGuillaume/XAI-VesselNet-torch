import torch

from monai.metrics.metric import CumulativeIterationMetric
from monai.utils import MetricReduction
from monai.metrics.utils import do_metric_reduction

import torch
import torch.nn.functional as F

from skimage.morphology import skeletonize, skeletonize_3d



def soft_erode(img: torch.Tensor) -> torch.Tensor:  # type: ignore
    """
    Perform soft erosion on the input image.
    Adapted from https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L12
    
    Args:
        img: the shape should be (B,C,H,W,[D])

    Returns:
        The eroded image
    """
    if len(img.shape) == 4:
        p1 = -(F.max_pool2d(-img, (3, 1), (1, 1), (1, 0)))
        p2 = -(F.max_pool2d(-img, (1, 3), (1, 1), (0, 1)))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -(F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0)))
        p2 = -(F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0)))
        p3 = -(F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1)))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:  # type: ignore
    """
    Perform soft dilation on the input image.
    Adapted from https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L24

    Args:
        img: the shape should be (B,C,H,W,[D])

    Returns:
        The dilated image
    """
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to perform soft opening on the input image.
    Adapted from https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L31

    Args:
        img: the shape should be BCH(WD)

    Returns:
        The dilated image
    """
    eroded_image = soft_erode(img)
    dilated_image = soft_dilate(eroded_image)
    return dilated_image


def soft_skel(img: torch.Tensor, iter_: int) -> torch.Tensor:
    """
    Perform soft skeletonization on the input image.
    Adapted from https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L35

    Args:
        img: the shape should be BCH(WD)
        iter_: number of iterations for skeletonization

    Returns:
        The skeletonized image
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class clDiceMetric(CumulativeIterationMetric):
    """
    Compute average clDice score for a set of pairs of prediction-groundtruth segmentations.

    It supports both multi-classes and multi-labels tasks.
    Input `y_pred` is compared with ground truth `y`.
    `y_pred` is expected to have binarized predictions and `y` can be single-channel class indices or in the
    one-hot format. The `include_background` parameter can be set to ``False`` to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background. `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]),
    `y` can also be in the format of `B1HW[D]`.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    This class is a derivative of Dice and simply replace the call to DiceMetric by clDiceHelper (https://docs.monai.io/en/stable/metrics.html#mean-dice)

    Args:
        include_background: whether to include Dice computation on the first channel of
            the predicted output. Defaults to ``True``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.
        num_classes: number of input channels (always including the background). When this is None,
            ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
            single-channel class indices and the number of classes is not automatically inferred from data.
    """
    def __init__(
        self,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_empty: bool = True,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty
        self.num_classes = num_classes
        self.dice_helper = clDiceHelper(
            include_background=self.include_background,
            reduction=MetricReduction.NONE,
            get_not_nans=False,
            softmax=False,
            ignore_empty=self.ignore_empty,
            num_classes=self.num_classes,
        )

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean Dice metric. `y` can be single-channel class indices or
                in the one-hot format.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.
        """
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        # compute dice (BxC) for each channel for each batch
        return self.dice_helper(y_pred=y_pred, y=y)  # type: ignore

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction and aggregation logic for the output of `compute_dice`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


class clDiceHelper:
    """
    Compute the clDice score between two tensors `y_pred` and `y`.
    Refer to: https://arxiv.org/abs/2003.07311
    
    This class is a derivative of DiceHelper (https://docs.monai.io/en/stable/_modules/monai/metrics/meandice.html#)

    Notes :  
     - `y_pred` and `y` can be single-channel class indices or in the one-hot format. Prefer one-hot encoding for greater consistency with other metrics.
     - arrays must be of shape [N, C, spatial_dim1, spatial_dim2, [spatial_dim3, ...]]

    """

    def __init__(
        self,
        include_background: bool | None = None,
        sigmoid: bool = False,
        softmax: bool | None = None,
        activate: bool = False,
        get_not_nans: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN_BATCH,
        ignore_empty: bool = True,
        num_classes: int | None = None,
    ) -> None:
        """

        Args:
            include_background: whether to include the score on the first channel
                (default to the value of `sigmoid`, False).
            sigmoid: whether ``y_pred`` are/will be sigmoid activated outputs. If True, thresholding at 0.5
                will be performed to get the discrete prediction. Defaults to False.
            softmax: whether ``y_pred`` are softmax activated outputs. If True, `argmax` will be performed to
                get the discrete prediction. Defaults to the value of ``not sigmoid``.
            activate: whether to apply sigmoid to ``y_pred`` if ``sigmoid`` is True. Defaults to False.
                This option is only valid when ``sigmoid`` is True.
            get_not_nans: whether to return the number of not-nan values.
            reduction: define mode of reduction to the metrics
            ignore_empty: if `True`, NaN value will be set for empty ground truth cases.
                If `False`, 1 will be set if the Union of ``y_pred`` and ``y`` is empty.
            num_classes: number of input channels (always including the background). When this is None,
                ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
                single-channel class indices and the number of classes is not automatically inferred from data.
        """
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.include_background = sigmoid if include_background is None else include_background
        self.softmax = not sigmoid if softmax is None else softmax
        self.activate = activate
        self.ignore_empty = ignore_empty
        self.num_classes = num_classes

    def compute_channel(self, y_pred: torch.Tensor, y: torch.Tensor, sk_y_pred: torch.Tensor, sk_y: torch.Tensor) -> torch.Tensor:
        """ Compute the clDice score.

        Args:
            y_pred (torch.Tensor): the volume of the prediction
            y (torch.Tensor): the volume of the reference
            sk_y_pred (torch.Tensor): the skeleton of the prediction
            sk_y (torch.Tensor): the skeleton of the reference

        Returns:
            torch.Tensor: the clDice score
        """
        def cl_score(vol, skel):
            """[this function computes the skeleton volume overlap]

            Args:
                vol ([bool]): [image]
                skel ([bool]): [skeleton]

            Returns:
                [float]: [computed skeleton volume intersection]
            """
            return torch.sum(torch.masked_select(vol, skel)) / torch.sum(skel)

        y_o = torch.sum(y)
        if y_o > 0:
            tprec = cl_score(y_pred, sk_y)
            tsens = cl_score(y, sk_y_pred)
            return (2.0 * tprec * tsens) / (tprec + tsens)
        if self.ignore_empty:
            return torch.tensor(float("nan"), device=y_o.device)
        denorm = y_o + torch.sum(y_pred)
        if denorm <= 0:
            return torch.tensor(1.0, device=y_o.device)
        return torch.tensor(0.0, device=y_o.device)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y_pred: input predictions with shape (batch_size, num_classes or 1, spatial_dims...).
                the number of channels is inferred from ``y_pred.shape[1]`` when ``num_classes is None``.
            y: ground truth with shape (batch_size, num_classes or 1, spatial_dims...).
        """
        _softmax, _sigmoid = self.softmax, self.sigmoid
        if self.num_classes is None:
            n_pred_ch = y_pred.shape[1]  # y_pred is in one-hot format or multi-channel scores
        else:
            n_pred_ch = self.num_classes
            if y_pred.shape[1] == 1 and self.num_classes > 1:  # y_pred is single-channel class indices
                _softmax = _sigmoid = False

        if _softmax:
            if x     > 1:
                y_pred = torch.argmax(y_pred, dim=1, keepdim=True)

        elif _sigmoid:
            if self.activate:
                y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred > 0.5

        first_ch = 0 if self.include_background else 1
        data = []
        for b in range(y_pred.shape[0]):
            c_list = []
            for c in range(first_ch, n_pred_ch) if n_pred_ch > 1 else [1]:
                x_pred = (y_pred[b, 0] == c) if (y_pred.shape[1] == 1) else y_pred[b, c].bool()
                x = (y[b, 0] == c) if (y.shape[1] == 1) else y[b, c]

                # At the moment, the skeletonization needs skimage library
                # We can use the soft-skeletonization proposed by the authors, but we need to check the outputs first
                # Something like that :
                #
                # if self.soft_skeletonize == True:
                #     skeletonize = soft_skel
                # else:
                #     if y[b, 0].ndimension() == 2:
                #         from skimage.morphology import skeletonize
                #         skeletonize = skeletonize # Not required, just to be clear       
                #     if y[b, 0].ndimension() == 3:
                #         from skimage.morphology import skeletonize_3d
                #         skeletonize = skeletonize_3d
                        
                # sk_x_pred = torch.from_numpy(skeletonize(x_pred)).bool() 
                # sk_x = torch.from_numpy(skeletonize(x)).bool()

                # skeletonize functions does not support gpu.
                # Keep in mind the current device, send x_pred and x to cpu for skeletonize and resend result to `current_device`
                current_device = torch.device(x_pred.get_device())

                if y[b, 0].ndimension() == 2:
                    # We are in a 2D problem
                    sk_x_pred = torch.from_numpy(skeletonize(x_pred.cpu())).bool().to(current_device)
                    sk_x = torch.from_numpy(skeletonize(x.cpu())).bool().to(current_device)
                elif y[b, 0].ndimension() == 3:
                    # We are in a 3D problem
                    sk_x_pred = torch.from_numpy(skeletonize_3d(x_pred.cpu())).bool().to(current_device)
                    sk_x = torch.from_numpy(skeletonize_3d(x.cpu())).bool().to(current_device)
                else:
                    raise NotImplementedError(f"clDice metric only support 2D/3D problems. Data is {y.shape}")

                c_list.append(self.compute_channel(x_pred, x, sk_x_pred, sk_x))
            data.append(torch.stack(c_list))
        data = torch.stack(data, dim=0).contiguous()  # type: ignore

        f, not_nans = do_metric_reduction(data, self.reduction)  # type: ignore
        return (f, not_nans) if self.get_not_nans else f