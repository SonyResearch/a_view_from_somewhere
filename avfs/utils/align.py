# Copyright (c) Sony AI Inc.
# All rights reserved.

import os
import dlib
from PIL import Image
import numpy as np
import scipy
import scipy.ndimage
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from avfs.utils.download import download_compressed_file

model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pretrained_models"
)
predictor_filepath = os.path.join(model_path, "shape_predictor_68_face_landmarks.dat")
if not os.path.isfile(predictor_filepath):
    download_compressed_file(
        save_path=model_path,
        url="http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    )

predictor = dlib.shape_predictor(predictor_filepath)
detector = dlib.get_frontal_face_detector()


def get_landmark(image_path: str) -> Optional[np.ndarray]:
    """Detects facial landmarks in an image file using the dlib library.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray or None: An array of 68 (x, y) coordinate pairs representing the
            facial landmarks, or None if the face could not be detected or
            multiple faces were detected.
    """
    # Load image.
    img = dlib.load_rgb_image(image_path)

    # Run detector.
    dets = detector(img, 1)
    if len(dets) != 1:
        return None

    shape = None
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)

    if shape is None:
        return None

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    return np.array(a)


def align_face(image_path: str, image_size: int = 160) -> None:
    """Aligns a face in an image.

    Args:
        image_path (str): The path to the input image file.
        image_size (int): The size of the output image. Default is 160.

    Returns:
        Tuple[Image.Image, np.ndarray]: The aligned image and the transformation matrix.
    """
    # Defaults.
    output_size: int = image_size
    transform_size: int = 4096
    enable_padding: bool = True

    # Get face landmarks.
    lm: np.ndarray = get_landmark(image_path=image_path)
    if lm is None:
        return

    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)

    # This results in larger crops compared to the original FFHQ dataset, such as
    # the ones used in FFHQ-Aging. If you want to use the same crop size as the
    # original FFHQ, replace 2.2 with 1.8.
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 2.2)

    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load image.
    img = Image.open(image_path).convert("RGB")

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))

        np_img = np.array(img)
        np_img = np.pad(
            np.float32(np_img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect"
        )
        h, w, _ = np_img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        np_img += (
            scipy.ndimage.gaussian_filter(np_img, [blur, blur, 0]) - np_img
        ) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        np_img += (np.median(np_img, axis=(0, 1)) - np_img) * np.clip(mask, 0.0, 1.0)

        img = Image.fromarray(np.uint8(np.clip(np.rint(np_img), 0, 255)), "RGB")
        quad += pad[:2]

    # Transform.
    img = img.transform(
        (transform_size, transform_size),
        Image.QUAD,
        (quad + 0.5).flatten(),
        Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    img.save(image_path)
    return


def align_faces(
    image_paths: List[str], image_size: int = 160, max_workers: int = 10
) -> None:
    """Aligns faces in a list of images using a helper function `align_face_helper`.

    Args:
        image_paths (List[str]): List of filepaths to the images.
        image_size (int, optional): Output size of the aligned images. Default is 160.
        max_workers (int, optional): Maximum number of worker threads. Default is 10.

    Returns:
        None. The aligned images are saved to disk.
    """
    max_workers = min(max_workers, cpu_count() - 1)

    # Define a helper function to align a single face.
    def align_face_helper(image_path):
        # Align the face and return it as a numpy array.
        return align_face(image_path, image_size)

    # Use a thread pool to align the faces in parallel.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _ = executor.map(align_face_helper, image_paths)
