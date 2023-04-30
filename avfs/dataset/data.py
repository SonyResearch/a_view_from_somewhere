# Copyright (c) Sony AI Inc.
# All rights reserved.

import os
import json
import numpy as np
import random

from torch.utils.data import DataLoader

from typing import Any, Dict, Optional, Tuple, List

from .loader import AVFSDataLoader

data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
)
os.makedirs(data_path, exist_ok=True)

avfs_version = "v1"
avfs_data_path = os.path.join(data_path, f"avfs-dataset-{avfs_version}")


class AVFSDataset:
    """Dataset class for the AVFS dataset.

    Args:
        loader_cfg (Dict[str, Any]): Configuration dictionary for the data loader.
        mask_type (str, optional): Type of mask. Default is `annotator_id`.
        random_seed (int, optional): Random seed for reproducibility. Default is 2021.
        max_train_subset_prop (float, optional): Maximum proportion of training data to
            use. Default is 1.
        validation_fivecrop_trans (bool, optional): Whether to use five-crop
            augmentation during validation. Default is False.

    Methods:
        __call__():
            Loads data from the prescreener, triplet, and annotator label sources,
            splits it into training and validation sets, creates loaders for the
            training and validation sets, and returns a dictionary containing relevant
            information.

            Returns:
                A dictionary with the following keys:
                    - `loaders`: A tuple of DataLoader objects for the training and
                        validation sets.
                    - `num_annotators`: An integer representing the number of
                        annotators (or groups of annotators depending on
                        `mask_type`) who annotated the data.
                    - `annotator_labels`: A dictionary mapping annotator IDs (or
                        annotator group IDs depending on `mask_type`) to annotator (or
                        group) labels.

    Raises:
        FileNotFoundError: If prescreener.json file or ooo_train_val.json file do
            not exist.
    """

    def __init__(
        self,
        loader_cfg: Dict[str, Any],
        mask_type: Optional[str] = "annotator_id",
        random_seed: int = 2021,
        max_train_subset_prop: Optional[float] = 1.0,
        validation_fivecrop_trans: bool = False,
    ):
        self.train_validation_data_split: Dict[str, Any]
        self.num_images: int
        self.odd_one_out_annotator_labels: List[Any]
        self.num_annotators: int
        self.image_ids: List[Any]
        self.odd_one_out_annotators: List[Any]
        self.data_splits: List[Any]
        self.odd_one_out_positions: List[Any]
        self.odd_one_out_triplets: List[Any]

        self.loader_cfg = loader_cfg
        self.validation_fivecrop_trans = validation_fivecrop_trans

        self.max_train_subset_prop = max_train_subset_prop
        if not mask_type:
            mask_type = "annotator_id"
        self.mask_type = mask_type
        self.random_seed = random_seed

        # Load annotator prescreener data.
        prescreener_data_filepath = os.path.join(avfs_data_path, "prescreener.json")
        if not os.path.isfile(prescreener_data_filepath):
            raise FileNotFoundError(f"{prescreener_data_filepath} does not exist.")
        with open(prescreener_data_filepath, "r") as prescreener_json_file:
            self.prescreener_data = json.load(prescreener_json_file)

        # Load odd-one-out data.
        odd_one_out_data_filepath = os.path.join(avfs_data_path, "ooo_train_val.json")
        if not os.path.isfile(odd_one_out_data_filepath):
            raise FileNotFoundError(f"{odd_one_out_data_filepath} does not exist.")
        with open(odd_one_out_data_filepath, "r") as odd_one_out_data_json_file:
            self.odd_one_out_data = json.load(odd_one_out_data_json_file)

        # Image data
        self.image_paths = [
            os.path.join(
                data_path,
                f"ffhq/ffhq160x160/{x - x % 1000:05d}/" f"{str(x).zfill(5)}.png",
            )
            for x in range(70000)
        ]

    def __call__(self) -> Tuple[Dict[str, DataLoader[Any]], Any, Dict]:
        """Loads data from the prescreener, triplet, and annotator label sources,
        splits it into training and validation sets, creates loaders for the training
        and validation sets, and returns a dictionary containing relevant information.

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - `loaders: A tuple of DataLoader objects for the training and
                    validation sets.
                - `num_annotators`: An integer representing the number of
                    annotators (or groups of annotators depending on
                    self.mask_type) who annotated the data.
                - `annotator_labels`: A dictionary mapping annotator IDs (or
                    annotator group IDs depending on self.mask_type) to
                    annotator (or group) labels.
        """
        self._convert_age_to_age_group()
        self._get_triplet_data()
        self._get_annotator_labels()
        self._split_into_train_validation()
        self._get_train_validation_loaders()
        return self.loaders, self.num_annotators, self.annotator_labels

    def _convert_age_to_age_group(self) -> None:
        """Converts age integers to age groups for annotators who completed the
        prescreener survey.

        Returns:
            None
        """
        for k, annotator in self.prescreener_data.items():
            age_int = annotator["age"]
            if 18 <= age_int < 20:
                annotator["age_group"] = "18-19"
            elif 20 <= age_int < 30:
                annotator["age_group"] = "20-29"
            elif 30 <= age_int < 40:
                annotator["age_group"] = "30-39"
            elif 40 <= age_int < 50:
                annotator["age_group"] = "40-49"
            elif 50 <= age_int < 121:
                annotator["age_group"] = "50-120"

    def _get_triplet_data(self) -> None:
        """Extracts the odd-one-out triplet IDs, odd-one-out judgments, odd-one-out
        data splits, and odd-one-out annotators.

        Returns:
            None
        """
        self.odd_one_out_triplets = []
        self.odd_one_out_positions = []
        self.data_splits = []
        self.odd_one_out_annotators = []

        # Iterate over each AVFS set of triplets.
        for data in self.odd_one_out_data.values():
            self.odd_one_out_triplets.extend(data["triplet_questions"])
            self.odd_one_out_positions.extend(data["odd_one_out_positions"])
            self.data_splits.extend(data["split"])
            self.odd_one_out_annotators.extend(
                [data["annotator_id"]] * len(data["triplet_questions"])
            )

        print(f"Number of triplets: {len(self.odd_one_out_triplets)}.")

        self.image_ids = np.sort(np.unique(self.odd_one_out_triplets)).tolist()
        self.num_images = len(self.image_ids)

    def _get_annotator_labels(self) -> None:
        """Extracts annotator labels and assigns unique integers to each label.

        If self.mask_type is `annotator_id`, assigns integers to each unique
        `annotator id`. If self.mask_type is one of [`gender_id`, `age_group`,
        `ancestry_regions`, `ancestry_subregions`, `nationality`], assigns integers
        to each unique label in the corresponding field of the `prescreener_data`. If
        self.mask_type is `intersectional`, assigns integers to each unique
        combination of `gender_id`, `age_group`, and `ancestry_regions`.

        Returns:
            None
        """
        unique_annotators = np.unique(self.odd_one_out_annotators)
        annotator_labels = {}

        for annotator in self.prescreener_data.values():
            if annotator["annotator_id"] in unique_annotators:
                if self.mask_type.lower() in [
                    "gender_id",
                    "age_group",
                    "ancestry_regions",
                    "ancestry_subregions",
                    "nationality",
                ]:
                    annotator_labels[annotator["annotator_id"]] = annotator[
                        self.mask_type.lower()
                    ]
                elif self.mask_type.lower() == "intersectional":
                    annotator_labels[annotator["annotator_id"]] = ",".join(
                        [
                            annotator["gender_id"],
                            annotator["age_group"],
                            annotator["ancestry_regions"],
                        ]
                    )

        if self.mask_type.lower() == "annotator_id":
            annotator_labels = {v: k for k, v in enumerate(unique_annotators)}
            num_annotators = len(annotator_labels)
        else:
            unique_annotator_label_map = {
                k: i for i, k in enumerate(np.unique(list(annotator_labels.values())))
            }
            annotator_labels = {
                k: unique_annotator_label_map[annotator_labels[k]]
                for k in unique_annotators
            }
            num_annotators = len(unique_annotator_label_map)

        self.annotator_labels = annotator_labels
        self.num_annotators = num_annotators
        self.odd_one_out_annotator_labels = [
            annotator_labels[x] for x in self.odd_one_out_annotators
        ]

    def _split_into_train_validation(self) -> None:
        """Splits data into train and validation sets.

        Returns:
            None
        """
        num_odd_one_out_triplets = len(self.odd_one_out_triplets)

        random.seed(self.random_seed)
        shuffled_ids = random.sample(
            range(num_odd_one_out_triplets), num_odd_one_out_triplets
        )

        odd_one_out_triplets = [self.odd_one_out_triplets[i] for i in shuffled_ids]
        odd_one_out_positions = [self.odd_one_out_positions[i] for i in shuffled_ids]
        data_splits = [self.data_splits[i] for i in shuffled_ids]
        odd_one_out_annotator_labels = [
            self.odd_one_out_annotator_labels[i] for i in shuffled_ids
        ]

        image_id_num_map = {
            image_id: image_num
            for image_id, image_num in zip(self.image_ids, range(self.num_images))
        }

        odd_one_out_triplets = [
            [image_id_num_map[x] for x in odd_one_out_triplets[i]]
            for i in range(num_odd_one_out_triplets)
        ]

        self.image_num_to_path = {
            image_num: self.image_paths[image_id]
            for image_id, image_num in image_id_num_map.items()
        }

        train_split_flag = np.where(np.array(data_splits) == 1)[0].tolist()
        validation_split_flag = np.where(np.array(data_splits) == 0)[0].tolist()

        odd_one_out_positions = np.array(odd_one_out_positions)

        num_train = sum(data_splits)
        if isinstance(self.max_train_subset_prop, float):
            train_end = min(int(self.max_train_subset_prop * num_train), num_train)
        else:
            train_end = num_train

        train_validation_data_split: Dict[str, Any] = {"train": {}, "validation": {}}
        train_validation_data_split["validation"]["odd_one_out_triplets"] = np.array(
            [odd_one_out_triplets[x] for x in validation_split_flag]
        ).tolist()
        train_validation_data_split["validation"][
            "odd_one_out_positions"
        ] = odd_one_out_positions[validation_split_flag]
        train_validation_data_split["train"]["odd_one_out_triplets"] = np.array(
            [odd_one_out_triplets[x] for x in train_split_flag[:train_end]]
        ).tolist()
        train_validation_data_split["train"][
            "odd_one_out_positions"
        ] = odd_one_out_positions[train_split_flag[:train_end]]

        train_validation_data_split["validation"]["annotator_masks"] = [
            odd_one_out_annotator_labels[x] for x in validation_split_flag
        ]
        train_validation_data_split["train"]["annotator_masks"] = [
            odd_one_out_annotator_labels[x] for x in train_split_flag[:train_end]
        ]

        self.train_validation_data_split = train_validation_data_split

    def _get_train_validation_loaders(self) -> None:
        """Create training and validation data loaders.

        Returns:
            None
        """
        train_dataset = AVFSDataLoader(
            data=self.train_validation_data_split["train"],
            img_paths=self.image_num_to_path,
            train=True,
            fivecrop_trans=False,
        )

        validation_dataset = AVFSDataLoader(
            data=self.train_validation_data_split["validation"],
            img_paths=self.image_num_to_path,
            train=False,
            fivecrop_trans=self.validation_fivecrop_trans,
        )
        train_batch_size = self.loader_cfg.pop("train_batch_size")
        validation_batch_size = self.loader_cfg.pop("validation_batch_size")

        train_loader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            drop_last=True,
            batch_size=train_batch_size,
            **self.loader_cfg,
        )

        validation_loader = DataLoader(
            dataset=validation_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=validation_batch_size,
            **self.loader_cfg,
        )

        self.loaders = {"train": train_loader, "validation": validation_loader}
