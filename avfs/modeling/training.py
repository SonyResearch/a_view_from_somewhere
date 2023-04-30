# Copyright (c) Sony AI Inc.
# All rights reserved.

from typing import Any, Dict, Optional, Set, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .metric import l1_norm, l2_norm, sum_negative_values, get_dims_above_threshold
from .loss import odd_one_out_loss
from .utils import learning_rate_adjuster, save_state

import logging


class AVFSTrainer(object):
    """
    AVFSTrainer instance for model training.

    Args:
        model (nn.Module): Model to train.
        optimizer (Optimizer): Optimizer.
        primary_device (str): Primary device to use.
        num_train_batches_to_average (int): Number of batches to average before logging
            training stats.
        num_epochs (int): Number of epochs to train the model.
        save_every_num_epochs (int): Save the model every `num` epochs.
        num_output_dims (int): Number of output embedding dimensions. (If masks are
            trained they have the same dimensionality as the embeddings.)
        learning_rate_cfg (Dict[str, Any]): Configuration for learning rate schedule.
        loss_cfg (Dict[str, Any]): Configuration for the loss function computation.
        monitor_validation_loss (str): Loss to monitor during model validation.
        annotator_labels (Optional[Dict[str, int]]): A dictionary of annotator
            labels (keys) and their corresponding mask indices (values).
        debug (bool): If `True`, the maximum number of batches from the training and
            validation data loaders is set to 10. If `False` (default), the data
            loaders are iterated over until exhausted.
    """

    def __init__(
        self,
        logger: logging.Logger,
        model: nn.Module,
        optimizer: Optimizer,
        primary_device: str,
        num_train_batches_to_average: int,
        num_epochs: int,
        save_every_num_epochs: int,
        num_output_dims: int,
        learning_rate_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        monitor_validation_loss: str,
        annotator_labels: Optional[Dict[str, int]] = None,
        debug: bool = False,
    ) -> None:
        self.logger = logger
        self._num_batches: int
        self._model = model
        self._optimizer = optimizer
        self._primary_device = primary_device
        if not debug:
            self._num_train_batches_to_average = num_train_batches_to_average
        else:
            self._num_train_batches_to_average = 1
        self._num_epochs = num_epochs
        self._save_every_num_epochs = save_every_num_epochs
        self._num_output_dims = num_output_dims
        self._annotator_labels = annotator_labels
        self._monitor_validation_loss = monitor_validation_loss
        self._debug = debug

        self._negative_embedding_values_loss_weight = loss_cfg.get(
            "negative_embedding_values_loss_weight", 0.0
        )
        self._odd_one_out_prediction_loss_weight = loss_cfg.get(
            "odd_one_out_prediction_loss_weight", 0.0
        )
        self._mask_loss_weight = loss_cfg.get("mask_loss_weight", 0.0)
        self._embeddings_sparsity_loss_weight = loss_cfg.get(
            "embeddings_sparsity_loss_weight", 0.0
        )
        self._dims_threshold_value = loss_cfg.get("dims_threshold_value", 0.0)

        self._learning_rate_milestones = learning_rate_cfg.get(
            "learning_rate_milestones"
        )
        self._learning_rate_factor = learning_rate_cfg.get("learning_rate_factor")

        # Initialize stats to monitor the training progress.
        self._stats: Dict[str, float] = {
            "ooo_acc": 0.0,
            "ooo_loss": 0.0,
            "sparsity_loss": 0.0,
            "neg_embed_val_loss": 0.0,
            "total_loss": 0.0,
            "dims_above_thresh": 0.0,
            "embed_val": 0.0,
        }

        if self._mask_loss_weight:
            self._stats["mask_loss"] = 0.0
            self._stats["mask_val"] = 0.0

        self._best_stats = {"best_epoch": 0, monitor_validation_loss: 1e12}

        self._total_num_batch_triplets = 0.0
        self._dims_above_threshold: Set[int] = set()

    def _reset_stats(self) -> None:
        """Reset the stats, count for the total number of batches of triplets, and the
        set of dimensions whose maximum value is above a threshold.

        Returns:
            None
        """
        self._stats = {k: 0.0 for k in self._stats.keys()}
        self._total_num_batch_triplets = 0.0
        self._dims_above_threshold = set()

    def _average_stats(
        self,
        batch_idx: Optional[int] = None,
        epoch_i: Optional[int] = None,
        train_mode: bool = True,
    ) -> None:
        """Average the stats and print to the logger.

        Args:
            batch_idx (int): The current batch index.

        Returns:
            None
        """
        self._stats["ooo_acc"] /= self._total_num_batch_triplets
        self._stats["ooo_loss"] /= self._total_num_batch_triplets
        self._stats["sparsity_loss"] /= self._total_num_batch_triplets
        self._stats["neg_embed_val_loss"] /= self._total_num_batch_triplets
        self._stats["total_loss"] /= self._total_num_batch_triplets
        self._stats["dims_above_thresh"] = len(self._dims_above_threshold)
        self._stats["embed_val"] /= (
            3 * self._total_num_batch_triplets * self._num_output_dims
        )

        if self._mask_loss_weight:
            self._stats["mask_loss"] /= self._total_num_batch_triplets
            self._stats["mask_val"] /= (
                self._total_num_batch_triplets * self._num_output_dims
            )

        if train_mode and batch_idx is not None:
            log_str = (
                f"Train@iter: {batch_idx + 1}/{self._num_batches} - "
                + " - ".join(
                    [
                        f"{stats_key}: {stats_value:.5g}"
                        for stats_key, stats_value in self._stats.items()
                    ]
                )
            )
            self.logger.info(log_str)
        elif not train_mode and epoch_i is not None:
            log_str = f"Val@epoch: {epoch_i + 1}/{self._num_epochs} - " + " - ".join(
                [
                    f"{stats_key}: {stats_value:.5g}"
                    for stats_key, stats_value in self._stats.items()
                ]
            )
            self.logger.info(log_str)

    def _set_mode(self, epoch_i: int, train_mode: bool = True) -> None:
        """Set the model and mask to training or evaluation mode for the current epoch.

        Args:
            epoch_i (int): The current epoch to train.
            train_mode (bool): If `True` (default), the model is put in training
                mode, otherwise the model is put in evaluation mode.

        Returns:
            None
        """
        if train_mode:
            self._model.train()
            if self._mask_loss_weight:
                self._model.masks.train()

            # Print the current epoch to the logger
            self.logger.info(f"Train@epoch: {epoch_i + 1}/{self._num_epochs}")
        else:
            self._model.eval()
            if self._mask_loss_weight:
                self._model.masks.eval()

    def _optimize_model(self, loss) -> None:
        """Optimize the model by updating its parameters using the computed loss.

        Args:
            loss: The computed loss.

        Returns:
            None
        """
        if loss != loss:
            self.logger.error("`NaN` detected in loss.")
            exit()

        # Clear grads, compute grads, and update model params using the optimizer.
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _adjust_learning_rate(self, epoch_i: int) -> None:
        """Adjust the learning rate based on the current epoch.

        Args:
            epoch_i (int): The current epoch to train.

        Returns:
            None
        """
        if self._learning_rate_milestones is not None:
            if (epoch_i + 1) in self._learning_rate_milestones:
                self.logger.info("Adjusting the learning rate.")
                learning_rate_adjuster(
                    optimizer=self._optimizer,
                    learning_rate_factor=self._learning_rate_factor,
                )

    def _process_data(
        self,
        epoch_i: int,
        data_loader: DataLoader,
        train_mode: bool = True,
    ) -> None:
        """Train the model for a single epoch.

        Args:
            epoch_i (int): The current epoch to train.
            data_loader (DataLoader): Data loader.
            train_mode (bool): If `True` (default), the model is put in training
                mode, otherwise the model is put in evaluation mode.

        Returns:
            None
        """
        # Reset the stats and set the model's mode.
        self._reset_stats()
        self._set_mode(epoch_i=epoch_i, train_mode=train_mode)
        self._num_batches = len(data_loader)

        for batch_idx, (
            (face_images_0, face_images_1, face_images_2),
            odd_one_out_positions,
            annotator_ids,
        ) in enumerate(data_loader):
            # Keep track of the number of odd-one-out triplets.
            num_batch_triplets = odd_one_out_positions.shape[0]
            self._total_num_batch_triplets += num_batch_triplets

            # Move the inputs (images) and targets (odd_one_out_positions) to the
            # primary device.
            face_images_0 = face_images_0.to(self._primary_device)
            face_images_1 = face_images_1.to(self._primary_device)
            face_images_2 = face_images_2.to(self._primary_device)
            odd_one_out_positions = odd_one_out_positions.to(self._primary_device)

            # Concatenate the inputs (images).
            face_images = torch.cat((face_images_0, face_images_1, face_images_2), 0)

            # Compute the annotator mask loss.
            annotator_masks = None
            weighted_mask_loss: Union[None, torch.Tensor] = None
            if self._mask_loss_weight:
                annotator_ids = annotator_ids.to(self._primary_device)
                annotator_masks = torch.sigmoid(self._model.masks(annotator_ids))
                mask_loss = l2_norm(annotator_masks)
                weighted_mask_loss = mask_loss * self._mask_loss_weight

            # Encode the face images.
            face_embeddings = self._model(face_images).view(num_batch_triplets * 3, -1)

            # Penalize negative embedding values.
            negative_embedding_values_loss = sum_negative_values(face_embeddings)

            weighted_negative_embedding_values_loss = (
                negative_embedding_values_loss
                * self._negative_embedding_values_loss_weight
            )

            # Zero-out negative embedding values.
            face_embeddings = F.relu(face_embeddings)

            # Embeddings sparsity loss.
            embeddings_sparsity_loss = l1_norm(face_embeddings)
            weighted_embeddings_sparsity_loss = (
                embeddings_sparsity_loss * self._embeddings_sparsity_loss_weight
            )

            # Reverse the concatenation of the face embeddings.
            face_embeddings_0 = face_embeddings[:num_batch_triplets]
            face_embeddings_1 = face_embeddings[
                num_batch_triplets : 2 * num_batch_triplets
            ]
            face_embeddings_2 = face_embeddings[2 * num_batch_triplets :]

            # Project the face embeddings into annotator subspaces.
            if self._mask_loss_weight:
                annotator_weighted_face_embeddings_0 = (
                    face_embeddings_0 * annotator_masks
                )
                annotator_weighted_face_embeddings_1 = (
                    face_embeddings_1 * annotator_masks
                )
                annotator_weighted_face_embeddings_2 = (
                    face_embeddings_2 * annotator_masks
                )
            else:
                annotator_weighted_face_embeddings_0 = face_embeddings_0
                annotator_weighted_face_embeddings_1 = face_embeddings_1
                annotator_weighted_face_embeddings_2 = face_embeddings_2

            # Count the number of dimensions of an input tensor whose maximum value is
            # above the threshold.
            dims_above_threshold = get_dims_above_threshold(
                x=torch.cat(
                    (
                        annotator_weighted_face_embeddings_0,
                        annotator_weighted_face_embeddings_1,
                        annotator_weighted_face_embeddings_2,
                    ),
                    0,
                ),
                threshold=self._dims_threshold_value,
            )
            self._dims_above_threshold = self._dims_above_threshold.union(
                dims_above_threshold
            )

            # Sum the embedding values per dimension.
            embedding_values_sum_per_dimension = torch.sum(
                torch.cat(
                    (
                        annotator_weighted_face_embeddings_0,
                        annotator_weighted_face_embeddings_1,
                        annotator_weighted_face_embeddings_2,
                    ),
                    0,
                )
            )

            # Compute the odd-one-out prediction loss.
            (
                odd_one_out_prediction_loss,
                odd_one_out_position_predictions,
            ) = odd_one_out_loss(
                embeddings_0=annotator_weighted_face_embeddings_0,
                embeddings_1=annotator_weighted_face_embeddings_1,
                embeddings_2=annotator_weighted_face_embeddings_2,
                odd_one_out_positions=odd_one_out_positions,
            )

            weighted_odd_one_out_prediction_loss = (
                odd_one_out_prediction_loss * self._odd_one_out_prediction_loss_weight
            )

            # Compute the number of correct odd-one-out predictions.
            num_correct_odd_out_out_predictions = torch.sum(
                odd_one_out_position_predictions == odd_one_out_positions
            ).float()

            # Add losses and metrics to the stats dictionary.
            self._stats["ooo_acc"] += num_correct_odd_out_out_predictions.item()
            self._stats["ooo_loss"] += weighted_odd_one_out_prediction_loss.item()
            self._stats["sparsity_loss"] += weighted_embeddings_sparsity_loss.item()
            self._stats[
                "neg_embed_val_loss"
            ] += weighted_negative_embedding_values_loss.item()
            self._stats["embed_val"] += embedding_values_sum_per_dimension.item()

            # Add mask loss and metric to the stats dictionary (if computed),
            # then aggregate all losses.
            if self._mask_loss_weight:
                # Sum over annotator mask values.
                sum_annotator_mask_values = torch.sum(annotator_masks)
                self._stats["mask_val"] += sum_annotator_mask_values.item()

                total_loss = (
                    weighted_odd_one_out_prediction_loss
                    + weighted_embeddings_sparsity_loss
                    + weighted_negative_embedding_values_loss
                )

                # Add the weighted mask loss to the total loss.
                if weighted_mask_loss is not None:
                    self._stats["mask_loss"] += weighted_mask_loss.item()
                    total_loss = total_loss + weighted_mask_loss
            else:
                total_loss = (
                    weighted_odd_one_out_prediction_loss
                    + weighted_embeddings_sparsity_loss
                    + weighted_negative_embedding_values_loss
                )

            self._stats["total_loss"] += total_loss.item()

            if train_mode:
                # Optimize the model.
                self._optimize_model(loss=total_loss)

                if (batch_idx + 1) % self._num_train_batches_to_average == 0:
                    # Average the stats and print to the logger.
                    self._average_stats(batch_idx=batch_idx, train_mode=train_mode)
                    self._reset_stats()

            if self._debug and (batch_idx + 1) >= 10:
                break

        if not train_mode:
            # Average the stats and print to the logger.
            self._average_stats(epoch_i=epoch_i, train_mode=train_mode)
        else:
            # Adjust the learning rate.
            self._adjust_learning_rate(epoch_i=epoch_i)

    def save_best_periodic(self, epoch_i: int, checkpoint_path: str) -> None:
        """Saves the model state and optimizer state periodically during training,
        as well as the best model state based on the validation loss. The saved
        states can be used to resume training or evaluate the model at a later time.

        Args:
            epoch_i (int): The current epoch index.
            checkpoint_path (str): The path where the model states will be saved.

        Returns:
            None
        """
        log_str_list = ["Val@epoch: {:d}/{:d}".format(epoch_i + 1, self._num_epochs)]

        # If the current loss is the best historically.
        if (
            self._stats[self._monitor_validation_loss]
            < self._best_stats[self._monitor_validation_loss]
        ):
            self._best_stats[self._monitor_validation_loss] = self._stats[
                self._monitor_validation_loss
            ]
            self._best_stats["best_epoch"] = epoch_i + 1

            best_save_filename = f"best_{self._monitor_validation_loss}.pth"
            save_state(
                save_path=checkpoint_path,
                save_filename=best_save_filename,
                model=self._model,
                optimizer=self._optimizer,
                current_epoch=epoch_i + 1,
                num_epochs=self._num_epochs,
                training_stats=self._stats,
                annotator_labels=self._annotator_labels,
            )

            log_str_list.append(
                f"best_{self._monitor_validation_loss}: "
                f"{self._best_stats[self._monitor_validation_loss]:.5g}"
            )

            best_log_str = " - ".join(log_str_list)
            self.logger.info(best_log_str)

        # Save periodically as well as the final epoch.
        if (epoch_i + 1) % self._save_every_num_epochs == 0 or (
            epoch_i + 1
        ) == self._num_epochs:
            periodic_save_filename = f"epoch{epoch_i + 1}.pth"
            save_state(
                save_path=checkpoint_path,
                save_filename=periodic_save_filename,
                model=self._model,
                optimizer=self._optimizer,
                current_epoch=epoch_i + 1,
                num_epochs=self._num_epochs,
                training_stats=self._stats,
                annotator_labels=self._annotator_labels,
            )

        # Print the best stats when reaching the end of training.
        if (epoch_i + 1) == self._num_epochs:
            final_log_str = (
                f"best@epoch: {self._best_stats['best_epoch']}/{self._num_epochs} - "
                f"{self._monitor_validation_loss}: "
                f"{self._best_stats[self._monitor_validation_loss]:.5g}"
            )
            self.logger.info(final_log_str)

    def train(
        self,
        epoch_i: int,
        data_loader: DataLoader,
        train_mode: bool = True,
    ) -> None:
        """Train the model for a single epoch.

        Args:
            epoch_i (int): The current epoch to train.
            data_loader (DataLoader): Data loader.
            train_mode (bool): If `True` (default), the model is put in training
                mode, otherwise the model is put in evaluation mode.

        Returns:
            None
        """
        self._process_data(
            epoch_i=epoch_i, data_loader=data_loader, train_mode=train_mode
        )

    @torch.no_grad()
    def validate(
        self,
        epoch_i: int,
        data_loader: DataLoader,
        train_mode: bool = False,
    ) -> None:
        """Validate the model.

        Args:
            epoch_i (int): The current epoch to train.
            data_loader (DataLoader): Data loader.
            train_mode (bool): If `False` (default), the model is put in evaluation
                mode, otherwise the model is put in training mode.

        Returns:
            None
        """
        self._process_data(
            epoch_i=epoch_i, data_loader=data_loader, train_mode=train_mode
        )
