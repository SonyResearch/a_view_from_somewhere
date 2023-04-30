# Copyright (c) Sony AI Inc.
# All rights reserved.

import torch
import torch.nn.functional as F

from typing import Set


def root_mean_squared_error(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Calculate the root mean squared error between two input tensors.

    Args:
        predictions (torch.Tensor): The predicted values of the dependent variable.
        targets (torch.Tensor): The true values of the dependent variable.

    Returns:
        rmse (torch.Tensor): The root mean squared error between targets and
            predictions.
    """

    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    return rmse


def mean_absolute_error(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Calculate the mean absolute error between two input tensors.

    Args:
        predictions (torch.Tensor): The predicted values of the dependent variable.
        targets (torch.Tensor): The true values of the dependent variable.

    Returns:
        torch.Tensor: The mean absolute error between predictions and targets.
    """
    return torch.mean(torch.abs(predictions - targets))


def sum_negative_values(x: torch.Tensor) -> torch.Tensor:
    """Sum the negative values in the input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The sum of the negative dimensions of the input tensor.
    """
    neg_dims = F.relu(-x)
    return torch.sum(neg_dims)


def get_dims_above_threshold(x: torch.Tensor, threshold: float = 0.1) -> Set[int]:
    """Get the dimension numbers of the dimensions whose maximum value is above
    the threshold.

    Args:
        x (torch.Tensor): Input tensor.
        threshold (float): Threshold value to determine if a dimension is positive.

    Returns:
        Set[int]: The set of dimensions of the input tensor whose maximum value is above
            the threshold.
    """
    max_dim = x.max(dim=0)[0]
    dims_above_threshold = set(torch.where(max_dim > threshold)[0].tolist())
    return dims_above_threshold


def count_positive_dims(x: torch.Tensor) -> torch.Tensor:
    """Count the number of positive elements in an input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The number of positive elements in the input tensor.
    """
    return (x > 0).sum()


def l1_norm(x: torch.Tensor) -> torch.Tensor:
    """Computes the L1 norm of an input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The L1 norm of the input tensor.
    """
    return torch.norm(x, p=1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """Computes the L2 norm of an input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The L2 norm of the input tensor.
    """
    return torch.norm(x, p=2)


def num_correct_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    is_logits: bool = True,
) -> torch.Tensor:
    """Compute the number of correctly predicted samples.

    Args:
        predictions (torch.Tensor): The predicted values of the dependent variable.
        targets (torch.Tensor): The true values of the dependent variable.
        num_classes (int): The number of possible classes.
        is_logits (bool): Whether the predictions are logits or probabilities.

    Returns:
        torch.Tensor: The number of correctly predicted samples.
    """
    if num_classes > 2:
        predicted_classes = torch.argmax(predictions, dim=1)
    else:
        if is_logits:
            probs = torch.sigmoid(predictions)
            predicted_classes = (probs > 0.5).to(torch.long)
        else:
            predicted_classes = (predictions > 0.5).to(torch.long)

    num_correct = torch.sum((predicted_classes == targets)).float()
    return num_correct


def num_correct_classes(
    predicted_classes: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Compute the number of correctly predicted samples.

    Args:
        predicted_classes (torch.Tensor): The predicted values of the dependent
            variable.
        targets (torch.Tensor): The true values of the dependent variable.

    Returns:
        torch.Tensor: Number of correctly predicted samples.
    """
    num_correct = torch.sum(predicted_classes == targets).float()
    return num_correct
