# Copyright (c) Sony AI Inc.
# All rights reserved.

import os
import torch
import torch.nn as nn

from avfs.modeling.face_encoder import resnet18
from avfs.modeling.utils import move_model_to_device
from avfs.utils.download import download_from_url

from typing import List, Tuple, Optional, Dict

model_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "pretrained_models"
)

avfs_model_registry = {
    "avfs_cph": {
        "path": os.path.join(model_path, "avfs_cph.pth"),
        "dims": [
            70,
            68,
            34,
            72,
            95,
            66,
            124,
            44,
            40,
            73,
            107,
            17,
            75,
            19,
            29,
            97,
            30,
            91,
            3,
            112,
            84,
            1,
        ],
        "zenodo_link": "https://zenodo.org/record/7878655/files/avfs_cph.pth",
    },
    "avfs_c": {
        "path": os.path.join(model_path, "avfs_c.pth"),
        "dims": [
            10,
            0,
            99,
            25,
            54,
            53,
            93,
            62,
            112,
            12,
            70,
            24,
            33,
            107,
            17,
            72,
            116,
            96,
            69,
            75,
            58,
            119,
            65,
            90,
            2,
            88,
            76,
            103,
        ],
        "zenodo_link": "https://zenodo.org/record/7878655/files/avfs_c.pth",
    },
    "avfs_u": {
        "path": os.path.join(model_path, "avfs_u.pth"),
        "dims": [
            70,
            68,
            34,
            72,
            95,
            66,
            124,
            44,
            40,
            73,
            107,
            17,
            75,
            19,
            29,
            97,
            30,
            91,
            3,
            112,
            84,
            1,
        ],
        "zenodo_link": "https://zenodo.org/record/7878655/files/avfs_u.pth",
    },
    "avfs_u_half": {
        "path": os.path.join(model_path, "avfs_u_half.pth"),
        "dims": [
            45,
            10,
            53,
            122,
            74,
            109,
            121,
            108,
            80,
            96,
            88,
            70,
            106,
            103,
            75,
            12,
            89,
            76,
            120,
            36,
        ],
        "zenodo_link": "https://zenodo.org/record/7878655/files/avfs_u_half.pth",
    },
    "avfs_u_quarter": {
        "path": os.path.join(model_path, "avfs_u_quarter.pth"),
        "dims": [77, 117, 78, 64, 60, 80, 65, 12, 46, 75, 89],
        "zenodo_link": "https://zenodo.org/record/7878655/files/avfs_u_quarter.pth",
    },
    "avfs_u_eighth": {
        "path": os.path.join(model_path, "avfs_u_eighth.pth"),
        "dims": [
            2,
            0,
            118,
            100,
            101,
            34,
            38,
            37,
            80,
            117,
            6,
            109,
            27,
            46,
            119,
            75,
            61,
            78,
            70,
        ],
        "zenodo_link": "https://zenodo.org/record/7878655/files/avfs_u_eighth.pth",
    },
}


def load_registered_model(
    devices: List[int],
    data_parallel: bool = False,
    primary_device: str = "cuda:0",
    model_name: str = "avfs_cph",
    eval_mode: bool = True,
) -> Tuple[nn.Module, Optional[Dict[str, int]]]:
    """Loads a registered model and moves it to a specified device.

    Args:
        devices (List[int]): A list of device IDs to use for data parallelism.
        data_parallel (bool): If True, the model is wrapped in a data parallel
            container. Default is False.
        primary_device (str): The primary device to use if data parallelism is not
            enabled. Default is `cuda:0`.
        model_name (str): The name of the registered model to load. Default is
            `avfs_cph`.
        eval_mode (bool): If True, sets the model to evaluation mode. Default is True.

    Returns:
        Tuple[nn.Module, Optional[Dict[str, int]]]: A tuple containing the loaded model
            on the specified device and, if the model is a "c" variant, a dictionary
            mapping annotator labels to integers.

    Raises:
        ValueError: If `model_name` is not a registered model or if an error occurs
            while loading the model's saved state dict.
    """
    if model_name not in list(avfs_model_registry.keys()):
        raise ValueError(
            f"{model_name} is not a registered model. Please try "
            f"one of the following: {list(avfs_model_registry.keys())}"
        )

    annotator_labels = None

    model_url = str(avfs_model_registry[model_name]["zenodo_link"])
    save_filepath = os.path.join(model_path, os.path.basename(model_url))
    if not os.path.isfile(save_filepath):
        download_from_url(download_url=model_url, save_filepath=save_filepath)

    model = resnet18(num_output_dims=128)

    if "_c" in model_name:
        model.create_masks(num_masks=1645, num_dimensions=128)

    model_state_dict_path = avfs_model_registry[model_name]["path"]
    loaded_pth = torch.load(
        model_state_dict_path, map_location=torch.device(primary_device)
    )

    try:
        model.load_state_dict(loaded_pth["model"], strict=True)
    except ValueError:
        raise ValueError("ValueError loading the model's saved state dict.")
    except Exception as e:
        raise type(e)(f"Error loading model: {e}") from e

    dim_idxs = avfs_model_registry[model_name]["dims"]

    model.fc.weight = torch.nn.Parameter(model.fc.weight[dim_idxs, :])
    model.__dict__.update(num_outputs=len(dim_idxs))

    if "_c" in model_name:
        model.masks.weight = torch.nn.Parameter(model.masks.weight[:, dim_idxs])
        annotator_labels = loaded_pth["annotator_labels"]

    # Move model to device.
    model = move_model_to_device(
        model=model,
        data_parallel=data_parallel,
        devices=devices,
        primary_device=primary_device,
    )

    if eval_mode:
        model = model.eval()

    return model, annotator_labels
