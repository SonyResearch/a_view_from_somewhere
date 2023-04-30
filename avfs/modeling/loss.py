# Copyright (c) Sony AI Inc.
# All rights reserved.

import torch
import torch.nn.functional as F
from typing import Tuple


def odd_one_out_loss(
    embeddings_0: torch.Tensor,
    embeddings_1: torch.Tensor,
    embeddings_2: torch.Tensor,
    odd_one_out_positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the odd-one-out loss between triplet embeddings.

    Args:
        embeddings_0 (torch.Tensor): Position 0 triplet embeddings.
        embeddings_1 (torch.Tensor): Position 1 triplet embeddings.
        embeddings_2 (torch.Tensor): Position 2 triplet embeddings.
        odd_one_out_positions (torch.Tensor): The true values of the dependent variable,
            i.e., the odd-one-out position for each triplet.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the odd-one-out loss and
            the predicted odd-one-out position for each triplet.
    """
    most_similar_pair_positions = 2 - odd_one_out_positions

    similarity_0_1 = (embeddings_0 * embeddings_1).sum(dim=1)
    similarity_0_2 = (embeddings_0 * embeddings_2).sum(dim=1)
    similarity_1_2 = (embeddings_1 * embeddings_2).sum(dim=1)

    pairwise_similarities = torch.vstack(
        [similarity_0_1, similarity_0_2, similarity_1_2]
    ).T

    loss = F.cross_entropy(
        pairwise_similarities, most_similar_pair_positions, reduction="sum"
    )

    # Prediction
    most_similar_idx = torch.argmax(pairwise_similarities, dim=1)
    odd_one_out_position_predictions = 2 - most_similar_idx
    return loss, odd_one_out_position_predictions


def cross_entropy_loss(
    predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "sum"
) -> torch.Tensor:
    """Compute the cross entropy loss between the predictions and targets.

    Args:
        predictions (torch.Tensor): The predicted values of the dependent variable.
        targets (torch.Tensor): The true values of the dependent variable.
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        torch.Tensor: The cross entropy loss.
    """
    return F.cross_entropy(predictions, targets, reduction=reduction)


def binary_cross_entropy_with_logits_loss(
    predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "sum"
) -> torch.Tensor:
    """Compute the binary cross entropy loss between the predictions and targets.
    Args:
        predictions (torch.Tensor): The predicted values of the dependent variable.
        targets (torch.Tensor): The true values of the dependent variable.
        reduction (str):  Specifies the reduction to apply to the output.

    Returns:
        torch.Tensor: The binary cross entropy loss.
    """
    return F.binary_cross_entropy_with_logits(predictions, targets, reduction=reduction)


def mean_squared_error_loss(
    predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "sum"
) -> torch.Tensor:
    """Compute the mean squared error between each element in the predictions and
    targets.

    Args:
        predictions (torch.Tensor): The predicted values of the dependent variable.
        targets (torch.Tensor): The true values of the dependent variable.
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        torch.Tensor: The mean squared error between each element in the predictions
            and targets.
    """
    return F.mse_loss(predictions, targets, reduction=reduction)


def mean_absolute_error_loss(
    predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "sum"
) -> torch.Tensor:
    """Compute the mean absolute error between each element in the predictions and
    targets.

    Args:
        predictions (torch.Tensor): The predicted values of the dependent variable.
        targets (torch.Tensor): The true values of the dependent variable.
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        torch.Tensor: The mean absolute error between each element in the predictions
            and targets.
    """
    return F.l1_loss(predictions, targets, reduction=reduction)
