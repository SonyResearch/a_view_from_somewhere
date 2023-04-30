# Copyright (c) Sony AI Inc.
# All rights reserved.

import os
import json
from PIL import Image, ImageDraw
import numpy as np
import random
from IPython.display import display
from typing import Dict, Any, List, Optional

data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
)
os.makedirs(data_path, exist_ok=True)

avfs_version = "v1"
avfs_data_path = os.path.join(data_path, f"avfs-dataset-{avfs_version}")


def predict_random_triplet(
    image_paths: List[str],
    image_embeddings: np.ndarray,
    annotator_mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> None:
    """Given a list of image paths and their embeddings, predict the position of the
    odd-one-out image among three randomly selected images.

    Args:
        image_paths (List[str]): A list of file paths to the images.
        image_embeddings (np.ndarray): The embeddings for the images. The shape should
            be (num_images, embedding_dim).
        annotator_mask (Optional[np.ndarray]): A binary mask for the annotator's
            embeddings. Should be a 1D numpy array of the same length as the embedding
            dimension. If not provided, no mask will be applied.
        seed (Optional[int]): The random seed to use. If not provided, a random seed
            will be generated each time this function is called.

    Returns:
        None

    Raises:
        ValueError: If the length of `image_paths` is less than 3.
    """
    if len(image_paths) < 3:
        raise ValueError(
            f"There are not enough images to create a triplet. The length of "
            f"`image_paths` is {len(image_paths)}."
        )

    if seed is not None:
        random.seed(seed)

    random_idxs = random.sample(list(range(len(image_paths))), 3)
    triplet_image_paths = [image_paths[random_idx] for random_idx in random_idxs]
    # Open three images
    images = [
        Image.open(image_path).convert("RGB") for image_path in triplet_image_paths
    ]

    # Odd-one-out probabilities
    triplet_image_embeddings = image_embeddings[random_idxs, :]
    if annotator_mask is not None:
        triplet_image_embeddings = [
            embedding * annotator_mask for embedding in triplet_image_embeddings
        ]

    sim_0_1 = np.exp(np.dot(triplet_image_embeddings[0], triplet_image_embeddings[1]))
    sim_0_2 = np.exp(np.dot(triplet_image_embeddings[0], triplet_image_embeddings[2]))
    sim_1_2 = np.exp(np.dot(triplet_image_embeddings[1], triplet_image_embeddings[2]))

    total = sim_0_1 + sim_0_2 + sim_1_2

    sim_0_1 /= total
    sim_0_2 /= total
    sim_1_2 /= total
    pairwise_similarities = np.array([sim_0_1, sim_0_2, sim_1_2])

    # Prediction
    most_similar_idx = np.argmax(pairwise_similarities)
    odd_one_out_position_predictions = 2 - most_similar_idx

    # Resize images to the same size
    size = (160, 160)
    images = [image.resize(size) for image in images]

    # Create a new blank image for the grid
    grid_size = (3, 1)
    grid_width = size[0] * grid_size[0]
    grid_height = size[1] * grid_size[1]
    grid_image = Image.new("RGB", (grid_width, grid_height))

    # Paste images into the grid
    for i in range(len(images)):
        if i == odd_one_out_position_predictions:
            # Add a red border around the odd-one-out image
            draw = ImageDraw.Draw(images[i])
            draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline="red", width=5)
        grid_image.paste(images[i], (size[0] * i, 0))

    display(grid_image)


def dimension_top10_bottom10(
    image_paths: List[str], embeddings: np.ndarray, dim: int
) -> None:
    """Display the top 10 and bottom 10 images for a given dimension of the embeddings
    np.ndarray.

    Args:
        image_paths (List[str]): A list of filepaths of the images.
        embeddings (np.ndarray): A 2D numpy array of shape (n_samples, n_features)
            containing the embeddings.
        dim (int): The dimension to use for sorting the embeddings.

    Returns:
        None

    Raises:
        ValueError: If `dim` is out of range.
    """
    if not 0 <= dim < embeddings.shape[1]:
        raise ValueError(f"Valid values for `dim` are in [0, {embeddings.shape[1]}).")

    grid_size = (1, 10)
    top10 = [image_paths[x] for x in np.argsort(embeddings[:, dim])[::-1][:10]]
    bottom10 = [image_paths[x] for x in np.argsort(embeddings[:, dim])[:10]]

    titles = [
        f"top 10/{embeddings.shape[0]} @ dimension {dim}",
        f"bottom 10/{embeddings.shape[0]} @ dimension {dim}",
    ]
    for title, image_set in zip(titles, [top10, bottom10]):
        # Resize images to the same size
        size = (160, 160)
        images = [Image.open(image_path).convert("RGB") for image_path in image_set]
        images = [image.resize(size) for image in images]

        # Create a new blank image for the grid
        grid_width = size[0] * grid_size[1]
        grid_height = size[1]
        grid_image = Image.new("RGB", (grid_width, grid_height))

        # Paste images into the grid
        for i in range(grid_size[1]):
            grid_image.paste(images[i], (i * size[0], 0))

        print(title)

        display(grid_image)


class AnnotatorInfo:
    """A class to handle information about annotators and their prescreening data.

    Raises:
        FileNotFoundError: If the prescreener.json file does not exist.
    """

    def __init__(self):
        # Load annotator prescreener data.
        prescreener_data_filepath = os.path.join(avfs_data_path, "prescreener.json")
        if not os.path.isfile(prescreener_data_filepath):
            raise FileNotFoundError(f"{prescreener_data_filepath} does not exist.")

        with open(prescreener_data_filepath, "r") as prescreener_json_file:
            prescreener_data = json.load(prescreener_json_file)

        prescreener_data = {
            prescreener_data[k]["annotator_id"]: v for k, v in prescreener_data.items()
        }

        self.prescreener_data = prescreener_data

    def annotator_identity(self, annotator_id: str) -> Dict[str, Any]:
        """
        Get the annotator identity information for a given annotator ID.

        Args:
            annotator_id (str): The annotator ID to look up.

        Returns:
            Dict[str, Any]: A dictionary containing the annotator identity information,
                with the "annotator_id" key removed.

        Raises:
            KeyError: If the annotator ID does not exist in the prescreener data.
        """
        try:
            return {
                k: v
                for k, v in self.prescreener_data[annotator_id].items()
                if k != "annotator_id"
            }
        except KeyError:
            raise KeyError(
                f"Annotator ID `{annotator_id}` does not exist in the "
                f"prescreener data."
            ) from None
