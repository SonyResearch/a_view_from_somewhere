# Copyright (c) Sony AI Inc.
# All rights reserved.

import torch
import torch.nn as nn

from typing import Type, Callable, Union, List, Optional


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """Returns a 3x3 convolutional layer.

    Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/
    models/resnet.py

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        groups (int, optional): Number of groups in the convolution. Defaults to 1.
        dilation (int, optional): Dilation factor for the convolution. Defaults to 1.

    Returns:
        nn.Conv2d: The 3x3 convolutional layer.

    """
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Returns a 1x1 convolutional layer.

    Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/
    models/resnet.py

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.

    Returns:
        nn.Conv2d: The 1x1 convolutional layer.
    """
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """A basic building block module of a ResNet network.

    Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/
    models/resnet.py

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        downsample (nn.Module, optional): Downsample module to reduce the input size.
            Defaults to None.
        groups (int, optional): Number of groups in the convolution. Defaults to 1.
        base_width (int, optional): Base width of the convolution. Defaults to 64.
        dilation (int, optional): Dilation factor for the convolution. Defaults to 1.
        norm_layer (Callable[..., nn.Module], optional): Type of normalization layer
            to use. Defaults to nn.BatchNorm2d.

    Raises:
        ValueError: If groups is not equal to 1 or base_width is not equal to 64.
        NotImplementedError: If dilation is greater than 1.
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        # If no normalization layer is provided, use nn.BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Check the values of groups, base_width, and dilation
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride
        # != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BasicBlock module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            out (torch.Tensor): The output tensor.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """A bottleneck building block module of a ResNet network.
    Bottleneck in torchvision places the stride for downsampling at 3x3
    convolution(self.conv2) while original implementation places the stride at the
    first 1x1 convolution(self.conv1) according to "Deep residual learning for image
    recognition" https://arxiv.org/abs/1512.03385. This variant is also known as
    ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/
    models/resnet.py

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride of the first convolutional layer. Default: 1.
        downsample (nn.Module, optional): Downsampling layer. Default: None.
        groups (int): Number of groups for the 3x3 convolutional layer. Default: 1.
        base_width (int): Base width of the convolutional layer. Default: 64.
        dilation (int): Dilation rate for the 3x3 convolutional layer. Default: 1.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.BatchNorm2d.
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride
        # != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """Residual Neural Network (ResNet) model.

    Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/
    models/resnet.py

    Args:
        block (Type[Union[BasicBlock, Bottleneck]]): The block type to use for
            the ResNet model (either a BasicBlock or Bottleneck block).
        layers (List[int]): A list of integers indicating the number of blocks
            to use in each layer of the ResNet model.
        num_output_dims (int): The dimensionality of the output embedding.
        zero_init_residual (bool): Whether to initialize the residual connections
            with zero weights.
        groups (int): The number of groups to use in the ResNet model.
        width_per_group (int): The width of each group in the ResNet model.
        replace_stride_with_dilation (Optional[List[bool]]): A list of booleans
            indicating whether to replace the stride with dilation in each layer
            of the ResNet model.
        norm_layer (Optional[Callable[..., nn.Module]]): The normalization layer
            to use in the ResNet model.

    Raises:
        ValueError: If `replace_stride_with_dilation` is not a 3-element tuple or
            None.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_output_dims: int = 128,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super(ResNet, self).__init__()
        self.masks: nn.Embedding
        self.num_masks: int
        self.num_output_dims = num_output_dims

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        self.inplanes = 64

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_output_dims, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """Builds a layer of blocks for the ResNet model.

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): Block type to use in the layer.
            planes (int): Number of output channels for each block in the layer.
            blocks (int): Number of blocks to include in the layer.
            stride (int, optional): Stride for the first block in the layer. Default
                is 1.
            dilate (bool, optional): Whether to apply dilation to the blocks. Default
                is False.

        Returns:
            nn.Sequential: Sequential module containing the blocks in the layer.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def create_masks(self, num_masks: int, num_dimensions: int) -> None:
        """Create a set of masks, where each mask has dimensionality `num_dimensions`.

        Args:
            num_masks (int): The number of masks to create.
            num_dimensions (int): The dimensionality of each mask.
        """
        self.num_masks = num_masks
        self.masks = nn.Embedding(num_masks, num_dimensions)
        nn.init.constant_(self.masks.weight, 1.0)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the ResNet model.

        Args:
            x (torch.Tensor): A torch.Tensor of shape (batch_size, num_channels,
                height, width).

        Returns:
            A torch.Tensor of shape (batch_size, num_output_dims) representing the
            embedding for each input sample.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the ResNet model.

        Args:
            x (torch.Tensor): A torch.Tensor of shape (batch_size, num_channels,
                height, width).

        Returns:
            A torch.Tensor of shape (batch_size, num_output_dims) representing the
            embedding for each input sample.
        """
        return self._forward_impl(x)


def resnet18(num_output_dims: int = 128) -> ResNet:
    """ResNet-18 model from the paper "Deep Residual Learning for Image Recognition":
    https://arxiv.org/pdf/1512.03385.pdf.

    Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/
    models/resnet.py

    Args:
        num_output_dims (int): Number of output dimensions, i.e., embedding
            dimensionality.

    Returns:
        ResNet: A ResNet-18 model.
    """
    return ResNet(
        block=BasicBlock, layers=[2, 2, 2, 2], num_output_dims=num_output_dims
    )
