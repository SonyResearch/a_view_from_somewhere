# Copyright (c) Sony AI Inc.
# All rights reserved.

import numpy as np
from typing import Dict, Union, List, Tuple
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class AVFSDataLoader(Dataset):
    """
    Data loader for the AVFS dataset.

    Args:
        data (Dict[str, Union[np.ndarray, list]]): Dictionary containing
            odd-one-out triplet image IDs (inputs), odd-one-out judgments
            (targets), and annotator mask IDs.
        img_paths (Dict[int, str]): Dictionary of image IDs (keys) and filepaths
            (values).
        train (bool): If True (default), the data loader is created for training.
        fivecrop_trans (bool): If True, the five-crop transformations is
            used. Default is False.

    Returns:
        None
    """

    def __init__(
        self,
        data: Dict[str, Union[np.ndarray, list]],
        img_paths: Dict[int, str],
        train: bool = True,
        fivecrop_trans: bool = False,
    ) -> None:
        self.odd_one_out_triplets = data["odd_one_out_triplets"]
        self.odd_one_out_positions = data["odd_one_out_positions"]
        self.annotator_masks = data["annotator_masks"]
        self.img_paths = img_paths

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize([128, 128]),
                    transforms.RandomCrop([112, 112]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            if not fivecrop_trans:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize([128, 128]),
                        transforms.CenterCrop([112, 112]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize([128, 128]),
                        transforms.FiveCrop([112, 112]),
                        transforms.Lambda(
                            lambda crops: torch.stack(
                                [transforms.ToTensor()(crop) for crop in crops]
                            )
                        ),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]
                )

    @staticmethod
    def _to_torch_tensor(x: np.ndarray) -> torch.Tensor:
        """
        Converts a numpy array to a tensor.

        Args:
            x (np.ndarray): Numpy array to be converted.

        Returns:
            torch.Tensor: PyTorch tensor of the input NumPy array.
        """
        return torch.from_numpy(np.asarray(x))

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.odd_one_out_positions)

    def __getitem__(
        self, odd_one_out_triplet_id
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
        Union[torch.Tensor, List],
    ]:
        """
        Get a single item from the dataset.

        Args:
            odd_one_out_triplet_id (int): The index of the item to retrieve.

        Returns:
            Tuple[Tuple[Tensor, Tensor, Tensor], Tensor, Union[Tensor, List]]: A tuple
                containing the three images, the odd-one-out position, and the
                annotator label indicating which annotator (or group) annotated
                the odd-one-out position.
        """
        if torch.is_tensor(odd_one_out_triplet_id):
            odd_one_out_triplet_id = odd_one_out_triplet_id.tolist()

        # Get the image paths for the odd-one-out triplet
        odd_one_out_triplet_image_paths = [
            self.img_paths[i] for i in self.odd_one_out_triplets[odd_one_out_triplet_id]
        ]

        image_0, image_1, image_2 = [
            self.transform(Image.open(path)) for path in odd_one_out_triplet_image_paths
        ]

        odd_one_out_position = self._to_torch_tensor(
            self.odd_one_out_positions[odd_one_out_triplet_id]
        ).long()

        # Get the annotator ID of the annotator who made the odd-one-out judgment.
        annotator_id = (
            self._to_torch_tensor(self.annotator_masks[odd_one_out_triplet_id]).long()
            if self.annotator_masks is not None
            else list()
        )
        return (image_0, image_1, image_2), odd_one_out_position, annotator_id


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], fivecrop_trans: bool = False):
        """
        A PyTorch Dataset class for loading images from a list of file paths.

        Args:
            image_paths (List[str]): A list of strings representing the paths to image
                files.
            fivecrop_trans (bool): If True, the five-crop transformations is
                used. Default is False.
        """
        self.image_paths = image_paths

        if not fivecrop_trans:
            self.transform = transforms.Compose(
                [
                    transforms.Resize([128, 128]),
                    transforms.CenterCrop([112, 112]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize([128, 128]),
                    transforms.FiveCrop([112, 112]),
                    transforms.Lambda(
                        lambda crops: torch.stack(
                            [transforms.ToTensor()(crop) for crop in crops]
                        )
                    ),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    @staticmethod
    def _open_image_with_pil(image_path: str) -> Image.Image:
        """
        Opens an image file using the PIL library.

        Args:
            image_path (str): A string representing the path to the image file.

        Returns:
            Image.Image: An Image object with three channels (RGB).

        Raises:
            ValueError: If the image file cannot be opened.
        """
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except IOError as e:
            raise ValueError(f"Failed to open image file {image_path}") from e

        return pil_image

    def __getitem__(self, image_index):
        image_path = self.image_paths[image_index]
        pil_image = self._open_image_with_pil(image_path=image_path)

        return self.transform(pil_image)

    def __len__(self):
        return len(self.image_paths)
