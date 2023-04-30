# Copyright (c) Sony AI Inc.
# All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Optional, Union

import inspect

import torch
import torch.nn as nn
from torch import optim

import random
import numpy as np
import logging
import secrets
from datetime import datetime
import yaml

from .face_encoder import resnet18
from .logger import create_logger


def freeze_model_gradients(
    model: nn.Module,
    freeze_encoder_gradients: bool = False,
    freeze_mask_gradients: bool = False,
) -> None:
    """Freezes or unfreezes the gradients of the model's parameters.

    Args:
        model (nn.Module): The model whose parameters will be frozen or unfrozen.
        freeze_encoder_gradients (bool): If True, freezes the encoder gradients; if
            False (default), unfreezes the gradients.
        freeze_mask_gradients (bool): If False (default), unfreezes the mask
            gradients (if the model has masks); if True the gradients will remain
            frozen.

    Returns:
        None

    Raises:
        AttributeError: If trying to unfreeze mask gradients but the masks do not
            exist.
    """
    # Freeze all gradients if `freeze_encoder_gradients` is True.
    for param in model.parameters():
        param.requires_grad = not freeze_encoder_gradients

    # Unfreeze the mask gradients if `freeze_mask_gradients` is False.
    if not freeze_mask_gradients:
        if hasattr(model, "masks") and hasattr(model.masks, "weight"):
            model.masks.weight.requires_grad = True
        else:
            raise AttributeError(
                "Tried to change `requires_grad` to `True` for the "
                "model's masks but no masks exist."
            )


def move_model_to_device(
    model: nn.Module,
    devices: List[int],
    data_parallel: bool = False,
    primary_device: str = "cuda:0",
) -> nn.Module:
    """Moves a model to a specified device.

    Args:
        model (nn.Module): Model to move.
        devices (List[int]): A list of device IDs to use for data
            parallelism.
        data_parallel (bool): If True, the model is wrapped in a data
            parallel container. Default is False.
        primary_device (str): The primary device to use if data parallelism is not
            enabled. Default is `cuda:0`.

    Returns:
        model (nn.Module): The model on the specified device.
    """
    if data_parallel and (len(devices) > 1) and (primary_device.startswith("cuda")):
        # Move model to a DataParallel container.
        device_list = [int(device) for device in devices]
        model = nn.DataParallel(model, device_list)

    # Move the model to the primary device.
    model = model.to(device=torch.device(primary_device))
    return model


def separate_model_params(
    model: nn.Module,
) -> Tuple[
    List[Union[torch.Tensor, None, Any]],
    List[Union[torch.Tensor, None, Any]],
    List[torch.Tensor],
]:
    """Separates batch normalization bias parameters and frozen parameters from the
    other parameters.

    Args:
        model (nn.Module): The model to separate parameters for.

    Returns:
         Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]: A tuple
            containing a list of batch normalization bias parameters, a list of frozen
            parameters, and a list of parameters that are not batch normalization
            bias parameters or frozen.

    Raises:
        ValueError: The number of parameters separated does not equal the total
            number of model parameters.
    """
    bnorm_bias_params = []
    frozen_params = []
    other_params = []

    for model_module in model.modules():
        if isinstance(model_module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            if model_module.weight.requires_grad:
                other_params.append(model_module.weight)
            else:
                frozen_params.append(model_module.weight)
            if model_module.bias is not None and model_module.bias.requires_grad:
                bnorm_bias_params.append(model_module.bias)
            else:
                frozen_params.append(
                    model_module.bias
                ) if model_module.bias is not None else None

        elif isinstance(model_module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if model_module.weight.requires_grad:
                bnorm_bias_params.append(model_module.weight)
            else:
                frozen_params.append(model_module.weight)
            if model_module.bias is not None and model_module.bias.requires_grad:
                bnorm_bias_params.append(model_module.bias)
            else:
                frozen_params.append(
                    model_module.bias
                ) if model_module.bias is not None else None
        elif isinstance(model_module, nn.Embedding):
            if model_module.weight.requires_grad:
                other_params.append(model_module.weight)
            else:
                frozen_params.append(model_module.weight)

    num_bnorm_bias_params = sum(p.numel() for p in bnorm_bias_params)
    num_frozen_params = sum(p.numel() for p in frozen_params)
    num_other_params = sum(p.numel() for p in other_params)

    num_model_params = sum(p.numel() for p in model.parameters())
    num_model_params_separated = (
        num_bnorm_bias_params + num_frozen_params + num_other_params
    )

    if num_model_params != num_model_params_separated:
        raise ValueError(
            "The number of parameters separated does not equal the total number of "
            "model parameters."
        )

    return bnorm_bias_params, frozen_params, other_params


def learning_rate_adjuster(
    optimizer: optim.Optimizer, learning_rate_factor: Union[int, float, None]
) -> None:
    """Adjusts the learning rate of the optimizer based on a given configuration.

    Args:
        optimizer (optim.Optimizer): The optimizer whose learning rate will be adjusted.
        learning_rate_factor (Union[int, float]): The factor by which to multiply the
            current learning rate.

    Returns:
        None.

    Raises:
        TypeError: If the `learning_rate_factor` is not a float or int.
        ValueError: If the `learning_rate_factor` value is not greater than zero.
    """
    if learning_rate_factor is not None:
        if not isinstance(learning_rate_factor, (float, int)):
            raise TypeError("`learning_rate_factor` must be a float or int.")
        if learning_rate_factor <= 0:
            raise ValueError("`learning_rate_factor` must be greater than zero.")
        for params in optimizer.param_groups:
            params["lr"] *= learning_rate_factor


def save_state(
    save_path: str,
    save_filename: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    current_epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
    training_stats: Optional[Dict[str, float]] = None,
    annotator_labels: Optional[Dict[str, int]] = None,
) -> None:
    """Save the state of a model and optimizer.

    Args:
        save_path (str): The path where the models will be saved.
        save_filename (str): The filename to use when saving.
        model (nn.Module): The model to be saved.
        optimizer (optim.Optimizer, optional): The optimizer to be saved.
        current_epoch (int, optional): The current epoch.
        num_epochs (int, optional): The maximum number of epochs.
        training_stats (Dict[str, float], optional): A dictionary of training
            statistics to save.
        annotator_labels (Optional[Dict[str, int]]): A dictionary of annotator
            labels (keys) and their corresponding mask indices (values).

    Returns:
        None

    Raises:
        ValueError: If the `save_path` does not exist.
    """
    chkpt_save_filepath = os.path.join(save_path, f"{save_filename}")
    if not os.path.isdir(save_path):
        raise ValueError(f"{save_path} does not exist.")

    chkpt: Dict[str, Any] = {
        "current_epoch": current_epoch,
        "num_epochs": num_epochs,
        "training_stats": training_stats,
        "model": model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict(),
        "optimizer": optimizer,
        "annotator_labels": annotator_labels,
    }

    torch.save(chkpt, chkpt_save_filepath)


def reload_state_dict(
    chkpt_path: str,
    chkpt_filename: str = "best_total_loss.pth",
    primary_device: str = "cuda:0",
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    freeze_encoder_gradients: bool = False,
    freeze_mask_gradients: bool = False,
    model_strict_load: bool = True,
    resume: bool = False,
) -> int:
    """Reload a state dict to the model and optimizer.

    Args:
        chkpt_path (str): The path where the checkpoint was saved.
        chkpt_filename (str): The filename of the saved checkpoint.
           Default is `best_total_loss.pth`.
        primary_device (str): The primary device to use. Default is `cuda:0`.
        model (nn.Module, optional): The model to load the saved state dict to.
           Default is None.
        optimizer (optim.Optimizer, optional): The optimizer to load the saved state
            dict to. Default is None.
        freeze_encoder_gradients (bool): If True, freezes the encoder gradients; if
            False (default), unfreezes the gradients. Default is False.
        freeze_mask_gradients (bool): If False (default), unfreezes the mask
            gradients (if the model has masks); if True the gradients will remain
            frozen. Default is False.
        model_strict_load (bool): If True (default), the model state dict
           will be loaded strictly, raising an error for unmatched keys; if False,
           unmatched keys will be ignored. Default is True.
        resume (bool): The optimizer state dict will be loaded and training will resume
            from the last saved epoch, raising an error for unmatched keys.
            Default is False.

    Returns:
        int: If loading the optimizer's saved state the function
        returns an int representing the current epoch of the optimizer. Otherwise,
        the function returns 0.

    Raises:
        FileNotFoundError: If the `chkpt_filepath` does not exist.
        ValueError: If there is a mismatch between the model's saved state dict and the
            current model when model_strict_load is True, or if there is a mismatch
            between the optimizer's saved state dict and the current optimizer
            when resume is True.
        Exception: If there is an error loading the model or optimizer state dict.
    """
    if model or optimizer:
        chkpt_filepath = os.path.join(chkpt_path, chkpt_filename)
        if not os.path.isfile(chkpt_filepath):
            raise FileNotFoundError(f"{chkpt_filepath} does not exist.")

        saved_state = torch.load(
            chkpt_filepath, map_location=torch.device(primary_device)
        )

        if model:
            try:
                model.load_state_dict(saved_state["model"], strict=model_strict_load)
            except ValueError:
                raise ValueError("ValueError loading the model's saved state dict.")
            except Exception as e:
                raise type(e)(f"Error loading model: {e}") from e

            if freeze_encoder_gradients or freeze_mask_gradients:
                freeze_model_gradients(
                    model=model,
                    freeze_encoder_gradients=freeze_encoder_gradients,
                    freeze_mask_gradients=freeze_mask_gradients,
                )

        if optimizer and resume:
            try:
                optimizer.load_state_dict(saved_state["optimizer"])
                return saved_state.get("current_epoch", 0)
            except ValueError:
                raise ValueError("ValueError loading the optimizer's saved state dict.")
            except Exception as e:
                raise type(e)(f"Error loading optimizer: {e}") from e
    return 0


def load_config(cfg_filepath: str) -> Dict[str, Any]:
    """Loads the configuration file using YAML.

    Args:
        cfg_filepath (str): The file path of the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the model training configuration.

    Raises:
        ValueError: If the `cfg_filepath` does not exist.
    """
    # Ensure that the config file exists
    if not os.path.isfile(cfg_filepath):
        raise ValueError(f"{cfg_filepath} does not exist.")

    # Load config YAML file.
    with open(cfg_filepath, "r") as yaml_file:
        cfg: Dict[str, Any] = yaml.safe_load(yaml_file)

    config_check(cfg=cfg)
    return cfg


def config_check(cfg: Dict[str, Any]) -> None:
    """Check the validity of the keys and values in the configuration dictionary.

    Args:
        cfg (Dict[str, Any]): A dictionary of configuration keys and values.

    Returns:
        None

    Raises:
        ValueError: If there are missing keys in the `cfg` dictionary or any of its
            nested dictionaries.
        TypeError: If `cfg` is not a dictionary or if any of the values in the
            dictionary or its nested dictionaries are of an unexpected type.
    """

    def validate_primary_cfg():
        primary_cfg = [
            "misc_cfg",
            "setup_cfg",
            "model_cfg",
            "loss_cfg",
            "learning_rate_cfg",
            "optim_cfg",
            "data_cfg",
            "loader_cfg",
        ]

        missing_keys = set(primary_cfg) - set(cfg.keys())
        if len(missing_keys) != 0:
            raise ValueError(f"The following primary keys are missing: {missing_keys}.")

        for key_name in primary_cfg:
            if not isinstance(cfg[key_name], dict):
                raise TypeError(
                    f"Value associated with `{key_name}` is not a dictionary."
                )

    def validate_misc_cfg():
        misc_cfg = {
            "experiment_name": [str],
            "monitor_validation_loss": [str],
            "model_strict_load": [bool],
            "resume": [bool],
            "num_train_batches_to_average": [int],
            "debug": [bool],
            "num_epochs": [int],
            "save_every_num_epochs": [int],
        }

        if set(misc_cfg.keys()) != set(cfg["misc_cfg"].keys()):
            raise ValueError(f"`misc_cfg` keys must be: {list(misc_cfg.keys())}.")

        for misc_cfg_key, misc_cfg_val in misc_cfg.items():
            if type(cfg["misc_cfg"][misc_cfg_key]) not in misc_cfg_val:
                raise TypeError(
                    f"Value associated with `{misc_cfg_key}` is not one of the "
                    f"following types: `[{misc_cfg_val}]`."
                )

    def validate_setup_cfg():
        setup_cfg = {
            "devices": [list],
            "primary_device": [str],
            "numpy_seed": [int],
            "pytorch_seed": [int],
            "random_seed": [int],
            "cudnn_benchmark": [bool],
            "data_parallel": [bool],
        }

        if set(setup_cfg.keys()) != set(cfg["setup_cfg"].keys()):
            raise ValueError(f"`setup_cfg` keys must be: {list(setup_cfg.keys())}.")

        for setup_cfg_key, setup_cfg_val in setup_cfg.items():
            if type(cfg["setup_cfg"][setup_cfg_key]) not in setup_cfg_val:
                raise TypeError(
                    f"Value associated with `{setup_cfg_key}` is not one of the "
                    f"following types: {setup_cfg_val}."
                )

        if not all([isinstance(x, int) for x in cfg["setup_cfg"]["devices"]]):
            raise TypeError("`devices` must be a list of ints.")

    def validate_model_cfg():
        model_cfg = {
            "chkpt_path": [type(None), str],
            "chkpt_filename": [type(None), str],
            "freeze_encoder_gradients": [bool],
            "freeze_mask_gradients": [bool],
            "posthoc_dims": [type(None), list],
            "num_output_dims": [int],
        }

        if set(model_cfg.keys()) != set(cfg["model_cfg"].keys()):
            raise ValueError(f"`model_cfg` keys must be: {list(model_cfg.keys())}.")

        for model_cfg_key, model_cfg_val in model_cfg.items():
            if type(cfg["model_cfg"][model_cfg_key]) not in model_cfg_val:
                raise TypeError(
                    f"Value associated with `{model_cfg_key}` is not one of the "
                    f"following types: {model_cfg_val}."
                )

        posthoc_dims = cfg["model_cfg"]["posthoc_dims"]
        if isinstance(posthoc_dims, list) and not all(
            [isinstance(x, int) for x in posthoc_dims]
        ):
            raise TypeError("`posthoc_dims` must be a list of ints.")

    def validate_loss_cfg():
        loss_cfg = {
            "odd_one_out_prediction_loss_weight": [float, int],
            "negative_embedding_values_loss_weight": [float, int],
            "mask_loss_weight": [float, int],
            "embeddings_sparsity_loss_weight": [float, int],
            "dims_threshold_value": [float, int],
        }

        if set(loss_cfg.keys()) != set(cfg["loss_cfg"].keys()):
            raise ValueError(f"`loss_cfg` keys must be: {list(loss_cfg.keys())}.")

        for loss_cfg_key, loss_cfg_val in loss_cfg.items():
            if type(cfg["loss_cfg"][loss_cfg_key]) not in loss_cfg_val:
                raise ValueError(
                    f"Value associated with `{loss_cfg_key}` is not one of the "
                    f"following types: `[{loss_cfg_val}]`."
                )

        if any([x < 0 for x in cfg["loss_cfg"].values()]):
            raise ValueError("`loss_cfg` values must be >= 0.")

        monitor_validation_loss = cfg["misc_cfg"]["monitor_validation_loss"]
        loss_names = ["ooo_loss", "sparsity_loss", "neg_embed_val_loss", "total_loss"]
        if monitor_validation_loss not in loss_names:
            raise ValueError(
                f"Invalid monitor loss `{monitor_validation_loss}`. "
                f"Options are: {', '.join(loss_names)}."
            )

    def validate_learning_rate_cfg():
        learning_rate_cfg = {
            "learning_rate_milestones": [list, type(None)],
            "learning_rate_factor": [int, float, type(None)],
        }

        if set(learning_rate_cfg.keys()) != set(cfg["learning_rate_cfg"].keys()):
            raise ValueError(
                f"`learning_rate_cfg` keys must be: {list(learning_rate_cfg.keys())}."
            )

        for learning_rate_cfg_key, learning_rate_cfg_val in learning_rate_cfg.items():
            if (
                type(cfg["learning_rate_cfg"][learning_rate_cfg_key])
                not in learning_rate_cfg_val
            ):
                raise ValueError(
                    f"Value associated with `{learning_rate_cfg_key}` is not one of "
                    f"the following types: `[{learning_rate_cfg_val}]`."
                )

        learning_rate_milestones = cfg["learning_rate_cfg"]["learning_rate_milestones"]
        if isinstance(learning_rate_milestones, list) and not all(
            [isinstance(x, int) for x in learning_rate_milestones]
        ):
            raise ValueError(
                "If `learning_rate_milestones` is not None, then it must be a list "
                "containing ints."
            )

    def validate_optim_cfg():
        optim_cfg = {"method": [str], "lr": [float], "weight_decay": [float]}
        if set(optim_cfg.keys()) != set(cfg["optim_cfg"].keys()):
            raise ValueError(f"`optim_cfg` keys must be: {list(optim_cfg.keys())}.")

        for optim_cfg_key, optim_cfg_val in optim_cfg.items():
            if type(cfg["optim_cfg"][optim_cfg_key]) not in optim_cfg_val:
                raise ValueError(
                    f"Value associated with `{optim_cfg_key}` is not one of the "
                    f"following types: `[{optim_cfg_val}]`."
                )

    def validate_data_cfg():
        data_cfg = {
            "create_masks": [bool],
            "mask_type": [str, type(None)],
            "max_train_subset_prop": [float],
        }
        if set(data_cfg.keys()) != set(cfg["data_cfg"].keys()):
            raise ValueError(f"`data_cfg` keys must be: {list(data_cfg.keys())}.")
        for data_cfg_key, data_cfg_val in data_cfg.items():
            if type(cfg["data_cfg"][data_cfg_key]) not in data_cfg_val:
                raise TypeError(
                    f"Value associated with `{data_cfg_key}` is not one of the "
                    f"following types: `[{data_cfg_val}]`."
                )

        create_masks = cfg["data_cfg"]["create_masks"]
        mask_type = cfg["data_cfg"]["mask_type"]
        mask_types = [
            "annotator_id",
            "gender_id",
            "age_group",
            "ancestry_regions",
            "ancestry_subregions",
            "nationality",
            "intersectional",
        ]
        if create_masks:
            if mask_type is None:
                raise ValueError(
                    "`mask_type` cannot be `None` when `create_masks` is `True`."
                )
            if mask_type not in mask_types:
                raise ValueError(
                    f"If `mask_type` is not `None` it must be one of: {mask_types}."
                )

        else:
            if mask_type is not None:
                raise ValueError(
                    "`mask_type` must be `None` when `create_masks` is `False`."
                )

        if not create_masks and cfg["loss_cfg"]["mask_loss_weight"] != 0:
            raise ValueError(
                "If `create_masks` is `False` then `mask_loss_weight` must be 0."
            )

    def validate_loader_cfg():
        loader_cfg = {
            "train_batch_size": [int],
            "validation_batch_size": [int],
            "num_workers": [int],
            "pin_memory": [bool],
        }
        if set(loader_cfg.keys()) != set(cfg["loader_cfg"].keys()):
            raise ValueError(f"`loader_cfg` keys must be: {list(loader_cfg.keys())}.")

        for loader_cfg_key, loader_cfg_val in loader_cfg.items():
            if type(cfg["loader_cfg"][loader_cfg_key]) not in loader_cfg_val:
                raise TypeError(
                    f"Value associated with `{loader_cfg_key}` is not one of the "
                    f"following types: `[{loader_cfg_val}]`."
                )

    validate_primary_cfg()
    validate_misc_cfg()
    validate_setup_cfg()
    validate_model_cfg()
    validate_loss_cfg()
    validate_learning_rate_cfg()
    validate_optim_cfg()
    validate_data_cfg()
    validate_loader_cfg()


def setup_environment(
    devices: List[int],
    primary_device: str = "cuda:0",
    numpy_seed: int = 2021,
    pytorch_seed: int = 2021,
    random_seed: int = 2021,
    cudnn_benchmark: bool = True,
) -> None:
    """Set up the environment for the experiment.

    Args:
        devices (List[int]): A list of device IDs to use for data
            parallelism.
        primary_device (str): The primary device to use. Default is `cuda:0`.
        numpy_seed (int): The seed to use for the NumPy random number generator.
            Default is 2021.
        pytorch_seed (int): The seed to use for the PyTorch random number generator.
            Default is 2021.
        random_seed (int): The seed to use for the Python random number generator.
            Default is 2021.
        cudnn_benchmark (bool): If True, enables the cuDNN benchmark mode.
            Default is True.

    Returns:
        None

    Raises:
        AttributeError: If unable to set the cuda device to `primary_device`.
        ValueError: If unable to set the cuda device to `primary_device`.
    """
    # Determine if CUDA is available.
    if torch.cuda.is_available() and primary_device.startswith("cuda"):
        # Set device visibility.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(device) for device in devices]
        )

        # Set primary device.
        try:
            torch.cuda.set_device(primary_device)
        except (AttributeError, ValueError) as e:
            raise type(e)(
                f"Unable to set torch cuda device to " f"{primary_device}."
            ) from e

        # Enable cuDNN benchmark mode.
        torch.backends.cudnn.benchmark = cudnn_benchmark

    # Set random, numpy, and PyTorch seeds.
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(pytorch_seed)


def setup_experiment(cfg: Dict[str, Any]) -> Tuple[logging.Logger, Dict[str, Any]]:
    """Sets up a new experiment.

    Args:
        cfg (dict): Config for setting up the experiment.

    Returns:
        Tuple[logging.Logger, Dict[str, Any]]: A tuple containing a Python logger
            object and an updated config dictionary containing the model training
            configuration.
    """
    experiments_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
    )
    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)

    # Create a directory to store specific sets of experiments based on the provided
    # name.
    exp_set_path = os.path.join(experiments_path, cfg["misc_cfg"]["experiment_name"])
    os.makedirs(exp_set_path, exist_ok=True)

    # Generate a random name for the specific experiment prefixed with the current
    # datetime.
    now = datetime.now()
    exp_id = now.strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(5)
    cur_exp_path = os.path.join(exp_set_path, exp_id)

    # Create a directory for the current experiment.
    os.makedirs(cur_exp_path, exist_ok=True)

    metadata_path = os.path.join(cur_exp_path, "metadata")
    os.makedirs(metadata_path, exist_ok=True)

    checkpoint_path = os.path.join(cur_exp_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Create logger and a training log file.
    train_log_filepath = os.path.join(metadata_path, "train.log")
    logger = create_logger(train_log_filepath)
    logger.info("Experiment setup complete.")

    cfg["paths"] = {
        "metadata_path": metadata_path,
        "checkpoint_path": checkpoint_path,
        "train_log_filepath": train_log_filepath,
    }

    return logger, cfg


def save_config_to_yaml(cfg: Dict[str, Any]) -> None:
    """Save a Python dictionary to a YAML file.

    Args:
        cfg (dict): The configuration dictionary to be saved.

    Returns:
        None
    """
    file_path = os.path.join(cfg["paths"]["metadata_path"], "config.yaml")
    with open(file_path, "w") as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)


def setup_model(
    devices: List[int],
    num_output_dims: int = 128,
    data_parallel: bool = False,
    primary_device: str = "cuda:0",
    create_masks: Optional[bool] = None,
    num_masks: Optional[int] = None,
    chkpt_path: Optional[str] = None,
    chkpt_filename: Optional[str] = None,
    freeze_encoder_gradients: bool = False,
    freeze_mask_gradients: bool = False,
    posthoc_dims: Optional[List[int]] = None,
    model_strict_load: bool = True,
) -> Tuple[nn.Module, int]:
    """Sets up a model with a specified number of output dimensions.

    Args:
        devices (List[int]): A list of device IDs to use for data
            parallelism.
        num_output_dims (int): Number of output dimensions, i.e., embedding
            dimensionality. Default is 128.
        data_parallel (bool): If True, the model is wrapped in a data
            parallel container. Default is False.
        primary_device (str): The primary device to use if data parallelism is not
            enabled. Default is `cuda:0`.
        create_masks (bool): If True, create annotator-specific model masks.
            Default is None.
        num_masks (int): Number of masks to create. Default is None.
        chkpt_path (str): The path to the checkpoint file to load the model
            weights from. Default is None.
        chkpt_filename (str): The filename of the checkpoint file to load the model
            weights from. Default is None.
        freeze_encoder_gradients (bool): If True, freeze encoder gradients during model
            weight loading. Default is False.
        freeze_mask_gradients (bool): If True, freeze annotator-specific mask gradients
            during model weight loading. Default is False.
        posthoc_dims (List[int]): A list of indices to retain as a subset of the
            dimensions for posthoc mask training. Default is None.
        model_strict_load (bool): If True (default), the model state dict
            will be loaded strictly, raising an error for unmatched keys; if False,
            unmatched keys will be ignored. Default is True.

    Returns:
        nn.Module: Configured model.
        final_num_output_dims (int): The final number of output dimensions, which may
            differ to the input argument `num_output_dims` if a list of posthoc
            dimensions were provided as an input argument.

    Raises:
        ValueError: If `posthoc_dims` are out of range or are not solely integers.
        TypeError: If `final_num_output_dims` is not an integer.
    """
    # Initialize model.
    model = resnet18(num_output_dims=num_output_dims)

    # Create annotator-specific model masks.
    if create_masks and num_masks:
        model.create_masks(num_masks=num_masks, num_dimensions=num_output_dims)

    # Reload model weights.
    if chkpt_path and chkpt_filename:
        reload_state_dict(
            chkpt_path=chkpt_path,
            chkpt_filename=chkpt_filename,
            primary_device=primary_device,
            model=model,
            freeze_encoder_gradients=freeze_encoder_gradients,
            freeze_mask_gradients=freeze_mask_gradients,
            model_strict_load=model_strict_load,
        )

        # Retain a subset of the dimensions for posthoc mask training.
        if isinstance(posthoc_dims, list) and create_masks and num_masks:
            num_model_dims = model.fc.weight.shape[0]
            num_posthoc_dims = len(posthoc_dims)

            if not (0 < num_posthoc_dims <= num_model_dims):
                raise ValueError(
                    f"{num_posthoc_dims} must be greater than zero and less than the "
                    f"number of model dimensions {num_model_dims}."
                )
            if not all(isinstance(dim_num, int) for dim_num in posthoc_dims):
                raise ValueError("`posthoc_dims` does not solely contain integers.")
            if not (min(posthoc_dims) >= 0 and max(posthoc_dims) < num_model_dims):
                raise ValueError("Some `posthoc_dims` are not valid.")

            model.fc.weight = nn.Parameter(model.fc.weight[posthoc_dims, :])
            model.__dict__.update(num_output_dims=len(posthoc_dims))
            model.masks.weight = nn.Parameter(model.masks.weight[:, posthoc_dims])

    # Get the final number of output dimensions.
    final_num_output_dims = model.__dict__.get("num_output_dims")
    if not isinstance(final_num_output_dims, int):
        raise TypeError("`final_num_output_dims` is not an integer.")

    # Move model to device.
    model = move_model_to_device(
        model=model,
        data_parallel=data_parallel,
        devices=devices,
        primary_device=primary_device,
    )
    return model, final_num_output_dims


def setup_optimizer(
    model: nn.Module,
    optim_cfg: Dict[str, Any],
    resume: bool = False,
    primary_device: str = "cuda:0",
    chkpt_path: Optional[str] = None,
    chkpt_filename: Optional[str] = None,
) -> Tuple[optim.Optimizer, int]:
    """Sets up the optimizer.

    Args:
        model (nn.Module): The model.
        optim_cfg (dict): The optimizer configuration.
        resume (bool): The optimizer state dict will be loaded and training will resume
            from the last saved epoch, raising an error for unmatched keys.
            Default is False.
        primary_device (str): The primary device to use if data parallelism is not
            enabled. Default is `cuda:0`.
        chkpt_path (str): The path to the checkpoint file to load the model
            weights from. Default is None.
        chkpt_filename (str): The filename of the checkpoint file to load the model
            weights from. Default is None.

    Returns:
        Tuple[optim.Optimizer, Optional[int]]: A tuple containing the configured
            optimizer and the current epoch of the saved state dict (0 if not loaded).

    Raises:
        Exception: If the optimizer method is unknown or unexpected parameters are
            provided.
    """
    method = optim_cfg.pop("method")
    optim_fn = getattr(optim, method, None)
    if not optim_fn:
        raise Exception(f"Unknown optimization method: '{method}'")

    expected_args = inspect.getfullargspec(optim_fn.__init__)[0]
    if expected_args[:2] != ["self", "params"]:
        raise ValueError("Optimizer must have `self` and `params` as first two args.")

    if not all(k in expected_args[2:] for k in optim_cfg.keys()):
        raise Exception(
            f"Unexpected parameters: expected {expected_args[2:]}, "
            f"got {optim_cfg.keys()}"
        )

    weight_decay = optim_cfg.get("weight_decay")
    if weight_decay:
        bnorm_bias_params, frozen_params, other_params = separate_model_params(model)

        model_params = [
            {"params": other_params, "weight_decay": weight_decay},
            {"params": bnorm_bias_params},
        ]

        optim_cfg.pop("weight_decay", None)
        optimizer = optim_fn(model_params, **optim_cfg)
    else:
        optimizer = optim_fn(model.parameters(), **optim_cfg)

    saved_state_current_epoch: int = 0
    # Reload weights.
    if chkpt_path and chkpt_filename and resume:
        saved_state_current_epoch = reload_state_dict(
            chkpt_path=chkpt_path,
            chkpt_filename=chkpt_filename,
            primary_device=primary_device,
            optimizer=optimizer,
            resume=resume,
        )
    return optimizer, saved_state_current_epoch
