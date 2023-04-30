# Copyright (c) Sony AI Inc.
# All rights reserved.

import os
import sys
import requests
import html
import hashlib
import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage
import threading
import queue
import time
import json
import uuid
import glob
import shutil

from avfs.utils import pydrive_utils
from collections import OrderedDict

from typing import Union, Tuple, Dict, Any, List, Optional

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 24 * 60 * 60

json_spec = dict(
    file_url="https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA",
    file_path="ffhq-dataset-v2.json",
    file_size=267793842,
    file_md5="425ae20f06a4da1d4dc0f46d40ba5fd6",
)

license_specs = {
    "json": dict(
        file_url="https://drive.google.com/uc?id=1SHafCugkpMZzYhbgOz0zCuYiy-hb9lYX",
        file_path="LICENSE.txt",
        file_size=1610,
        file_md5="724f3831aaecd61a84fe98500079abc2",
    ),
    "images": dict(
        file_url="https://drive.google.com/uc?id=1sP2qz8TzLkzG2gjwAa4chtdB31THska4",
        file_path="images1024x1024/LICENSE.txt",
        file_size=1610,
        file_md5="724f3831aaecd61a84fe98500079abc2",
    ),
    "thumbs": dict(
        file_url="https://drive.google.com/uc?id=1iaL1S381LS10VVtqu-b2WfF9TiY75Kmj",
        file_path="thumbnails128x128/LICENSE.txt",
        file_size=1610,
        file_md5="724f3831aaecd61a84fe98500079abc2",
    ),
    "wilds": dict(
        file_url="https://drive.google.com/uc?id=1rsfFOEQvkd6_Z547qhpq5LhDl2McJEzw",
        file_path="in-the-wild-images/LICENSE.txt",
        file_size=1610,
        file_md5="724f3831aaecd61a84fe98500079abc2",
    ),
    "tfrecords": dict(
        file_url="https://drive.google.com/uc?id=1SYUmqKdLoTYq-kqsnPsniLScMhspvl5v",
        file_path="tfrecords/ffhq/LICENSE.txt",
        file_size=1610,
        file_md5="724f3831aaecd61a84fe98500079abc2",
    ),
}


def download_file(
    session: requests.Session,
    file_spec: dict,
    stats: dict,
    chunk_size: int = 128,
    num_attempts: int = 10,
) -> None:
    """Downloads a file from a given URL using the provided session object and saves it
    to a specified file path. It also performs validation by comparing file size,
    file MD5 hash, pixel size, and pixel MD5 hash (if provided).

    Adapted from: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    Args:
        session (requests.Session): A requests.Session object used to make the
            download request.
        file_spec (dict): A dictionary containing the file URL and file path.
        stats (dict): A dictionary containing statistics on the download progress.
        chunk_size (int): The number of bytes to read at a time. Default is 128.
        num_attempts (int): The number of times to attempt to download the file.
            Default is 10.

    Returns:
        None

    Raises:
        IOError: If the downloaded file size or MD5 hash is incorrect, or if the pixel
            size or MD5 hash is incorrect (if provided).
        IOError: If the Google Drive download quota is exceeded.
        Exception: If all attempts to download the file have failed.
    """
    file_path = file_spec["file_path"]
    file_url = file_spec["file_url"]
    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + ".tmp." + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in res.iter_content(chunk_size=chunk_size << 10):
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)
                        with stats["lock"]:
                            stats["bytes_done"] += len(chunk)

            # Validate.
            if "file_size" in file_spec and data_size != file_spec["file_size"]:
                raise IOError("Incorrect file size", file_path)
            if (
                "file_md5" in file_spec
                and data_md5.hexdigest() != file_spec["file_md5"]
            ):  # noqa
                raise IOError("Incorrect file MD5", file_path)
            if "pixel_size" in file_spec or "pixel_md5" in file_spec:
                with PIL.Image.open(tmp_path) as image:
                    if (
                        "pixel_size" in file_spec
                        and list(image.size) != file_spec["pixel_size"]
                    ):  # noqa
                        raise IOError("Incorrect pixel size", file_path)
                    if (
                        "pixel_md5" in file_spec
                        and hashlib.md5(np.array(image)).hexdigest()
                        != file_spec["pixel_md5"]
                    ):  # noqa
                        raise IOError("Incorrect pixel MD5", file_path)
            break

        except Exception as e:
            with stats["lock"]:
                stats["bytes_done"] -= data_size

            # Handle known failure cases.
            if 0 < data_size < 8192:
                with open(tmp_path, "rb") as f:
                    data = f.read()
                data_str = data.decode("utf-8")

                # Google Drive virus checker nag.
                links = [
                    html.unescape(link)
                    for link in data_str.split('"')
                    if "export=download" in link
                ]
                if len(links) == 1:
                    if attempts_left:
                        file_url = requests.compat.urljoin(file_url, links[0])
                        continue

                # Google Drive quota exceeded.
                if "Google Drive - Quota exceeded" in data_str:
                    if not attempts_left:
                        raise IOError(
                            "Google Drive download quota exceeded -- "
                            "please try again later"
                        )

            # Last attempt => raise error.
            if not attempts_left:
                raise e

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path)  # atomic

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + ".tmp.*"):
        try:
            os.remove(filename)
        except Exception as e:
            print(f"Could not remove {filename}. Error: {e}")
            pass


def choose_bytes_unit(num_bytes: Union[int, float]) -> Tuple[str, int]:
    """Choose the appropriate units for a given number of bytes.

    Adapted from: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    Args:
        num_bytes (Union[int, float]): The number of bytes.

    Returns:
        Tuple[str, int]: A tuple containing the units (str) and the conversion
            factor (int).

    Raises:
        TypeError: If `num_bytes` is not an int or float.
        ValueError: If `num_bytes` is negative.
    """
    if not isinstance(num_bytes, (int, float)):
        raise TypeError("`num_bytes` must be an int or float.")

    if num_bytes < 0:
        raise ValueError("`num_bytes` must be non-negative.")

    b = int(np.rint(num_bytes))  # round to the nearest integer
    if b < (1000**1):
        # Less than 1000 bytes
        return "B", (1**0)
    elif b < (1000**2):
        # Between 1 KB and 1 MB
        return "kB", (1**10)
    elif b < (1000**3):
        # Between 1 MB and 1 GB
        return "MB", (1**20)
    elif b < (1000**4):
        # Between 1 GB and 1 TB
        return "GB", (1**30)
    else:
        # 1 TB or more
        return "TB", (1**40)


def format_time(seconds: Union[int, float]) -> str:
    """Format a duration in seconds as a string.

    Adapted from: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    Args:
        seconds (Union[int, float]): The duration in seconds.

    Returns:
        str: The duration formatted as a string.

    Raises:
        TypeError: If `seconds` is not an int or float.
        ValueError: If `seconds` is negative.
    """
    if not isinstance(seconds, (int, float)):
        raise TypeError("`seconds` must be an int or float.")

    if seconds < 0:
        raise ValueError("`seconds` must be non-negative.")

    s = int(np.rint(seconds))

    if s < SECONDS_PER_MINUTE:
        # Less than 1 minute
        return f"{s}s"
    elif s < SECONDS_PER_HOUR:
        # Between 1 minute and 1 hour
        minutes = s // SECONDS_PER_MINUTE
        seconds = s % SECONDS_PER_MINUTE
        return f"{minutes}m {seconds:02d}s"
    elif s < SECONDS_PER_DAY:
        # Between 1 hour and 1 day
        hours = s // SECONDS_PER_HOUR
        minutes = (s // SECONDS_PER_MINUTE) % 60
        return f"{hours}h {minutes:02d}m"
    elif s < 100 * SECONDS_PER_DAY:
        # Between 1 day and 100 days
        days = s // SECONDS_PER_DAY
        hours = (s // SECONDS_PER_HOUR) % 24
        return f"{days}d {hours:02d}h"
    else:
        # More than 100 days
        return ">100d"


def download_files(
    file_specs: List[dict],
    dst_dir: str = ".",
    output_size: int = 160,
    check_invalid_images: bool = False,
    drive: Optional[str] = None,
    num_threads: int = 32,
    status_delay: float = 0.2,
    timing_window: int = 50,
    **download_kwargs: Any,
) -> None:
    """Download files from `file_specs` to `dst_dir` using `num_threads` worker threads.

    Adapted from: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    Args:
        file_specs (List[dict]): A list of dictionaries containing information about
            the files to be downloaded. Each dictionary must have the following keys:
            - "file_path" (str): The path to the file to be downloaded.
            - "file_size" (int): The size of the file to be downloaded in bytes.
        dst_dir (str, optional): The destination directory where the files will be
            saved. Default is ".".
        output_size (int, optional): The size (in pixels) to which the images should
            be resized. Default is 160.
        check_invalid_images (bool, optional): Whether to skip invalid images.
            If True, the function will attempt to open each image using
            PIL.Image.open() and skip those that raise an exception. Default is False.
        drive (str, optional): The name of the Google Drive account to use for
            downloading files. Default is None.
        num_threads (int, optional): The number of worker threads to use for
            downloading files. Default is 32.
        status_delay (float, optional): The number of seconds to wait for exceptions
            in the worker threads. Default is 0.2.
        timing_window (int, optional): The number of previous download times to
            consider when calculating the download speed. Defaults to 50.
        **download_kwargs (Any): Any additional keyword arguments to pass to the
            download function.

    Returns:
        None
    """
    # Determine which files to download.
    done_specs = {}
    for spec in file_specs:
        if os.path.isfile(spec["file_path"].replace("in-the-wild-images", dst_dir)):
            if check_invalid_images:
                try:
                    PIL.Image.open(
                        spec["file_path"].replace("in-the-wild-images", dst_dir)
                    )
                    done_specs.update({spec["file_path"]: spec})
                except Exception as e:
                    print(f"Exception {e} occurred. Continuing ...")
                    continue
            else:
                done_specs.update({spec["file_path"]: spec})

    missing_specs = [spec for spec in file_specs if spec["file_path"] not in done_specs]
    files_total = len(file_specs)
    bytes_total = sum(spec["file_size"] for spec in file_specs)
    stats = dict(
        files_done=len(done_specs),
        bytes_done=sum(spec["file_size"] for spec in done_specs.values()),
        lock=threading.Lock(),
    )
    if len(done_specs) == files_total:
        print("All files already downloaded -- skipping.")
        return

    # Launch worker threads.
    spec_queue: queue.Queue = queue.Queue()
    exception_queue: queue.Queue = queue.Queue()
    for spec in missing_specs:
        spec_queue.put(spec)
    thread_kwargs = dict(
        spec_queue=spec_queue,
        exception_queue=exception_queue,
        stats=stats,
        dst_dir=dst_dir,
        output_size=output_size,
        drive=drive,
        download_kwargs=download_kwargs,
    )
    for _thread_idx in range(min(num_threads, len(missing_specs))):
        threading.Thread(
            target=_download_thread, kwargs=thread_kwargs, daemon=True
        ).start()

    # Monitor status until done.
    bytes_unit, bytes_div = choose_bytes_unit(bytes_total)
    spinner = "/-\\|"
    timing: List[Tuple[float, int]] = []
    while True:
        spinner = spinner[1:] + spinner[:1]
        if drive is not None:
            with stats["lock"]:
                files_done = stats["files_done"]

            print(
                f"\r{spinner[0]} done processing {files_done}/{files_total} files",
                end="",
                flush=True,
            )
        else:
            with stats["lock"]:
                files_done = stats["files_done"]
                bytes_done = stats["bytes_done"]
            timing = timing[max(len(timing) - timing_window + 1, 0) :] + [
                (time.time(), bytes_done)
            ]  # noqa
            bandwidth = max(
                (timing[-1][1] - timing[0][1])
                / max(timing[-1][0] - timing[0][0], 1e-8),
                0,
            )
            bandwidth_unit, bandwidth_div = choose_bytes_unit(bandwidth)
            eta = format_time((bytes_total - bytes_done) / max(bandwidth, 1))

            print(
                "\r%s %6.2f%% done processed %d/%d files  "
                "%-13s  %-10s  ETA: %-7s "
                % (
                    spinner[0],
                    bytes_done / bytes_total * 100,
                    files_done,
                    files_total,
                    "downloaded %.2f/%.2f %s"
                    % (bytes_done / bytes_div, bytes_total / bytes_div, bytes_unit),
                    "%.2f %s/s" % (bandwidth / bandwidth_div, bandwidth_unit),
                    "done"
                    if bytes_total == bytes_done
                    else "..."
                    if len(timing) < timing_window or bandwidth == 0
                    else eta,
                ),
                end="",
                flush=True,
            )

        if files_done == files_total:
            print()
            break

        try:
            exc_info = exception_queue.get(timeout=status_delay)
            raise exc_info[1].with_traceback(exc_info[2])
        except queue.Empty:
            pass


def _download_thread(
    spec_queue: queue.Queue,
    exception_queue: queue.Queue,
    stats: Dict[str, Any],
    dst_dir: str,
    output_size: int,
    drive: Any,
    **download_kwargs: Any,
) -> None:
    """Download a file from a given file specification in a separate thread.

    Adapted from: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    Args:
        spec_queue (queue.Queue): A queue object containing the file specifications to
            be downloaded.
        exception_queue (queue.Queue): A queue object to hold exceptions encountered
            during downloading.
        stats (Dict[str, Any]): A dictionary object to hold the download statistics.
        dst_dir (str): A string specifying the directory to download files to.
        output_size (int): An integer specifying the size of the output image.
        drive (Any): An instance of PyDrive used for downloading files from
            Google Drive.
        download_kwargs (Any): Any additional keyword arguments to pass to the download
            function.

    Returns:
        None
    """
    with requests.Session() as session:
        while not spec_queue.empty():
            spec = spec_queue.get()
            try:
                if drive is not None:
                    pydrive_utils.pydrive_download(
                        drive, spec["file_url"], spec["file_path"]
                    )
                else:
                    download_file(session, spec, stats, **download_kwargs)

                if spec["file_path"].endswith(".png"):
                    align_in_the_wild_image(spec, dst_dir, output_size)
                    os.remove(spec["file_path"])
            except:  # noqa
                exception_queue.put(sys.exc_info())

            with stats["lock"]:
                stats["files_done"] += 1


def align_in_the_wild_image(
    spec: dict,
    dst_dir: str,
    output_size: int,
    transform_size: int = 4096,
    enable_padding: bool = True,
) -> None:
    """This function aligns an in-the-wild image using facial landmarks and saves the
    aligned image to the destination directory.

    Adapted from: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    Args:
        spec (dict): The dictionary containing facial landmarks and file path of
            the in-the-wild image.
        dst_dir (str): The destination directory where the aligned image will be saved.
        output_size (int): The size of the output image.
        transform_size (int, optional): The size of the transform. Defaults to 4096.
        enable_padding (bool, optional): Whether to enable padding. Defaults to True.

    Returns:
        None
    """
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    item_idx = int(os.path.basename(spec["file_path"])[:-4])

    # Parse landmarks.
    lm = np.array(spec["face_landmarks"])
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

    # Load in-the-wild image.
    src_file = spec["file_path"]
    if not os.path.isfile(src_file):
        print("\nCannot find source image. Run '--wilds' before '--align'.")
        return
    img = PIL.Image.open(src_file)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
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

        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(np_img), 0, 255)), "RGB")
        quad += pad[:2]

    # Transform.
    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    dst_subdir = os.path.join(dst_dir, f"{item_idx - item_idx % 1000:05d}")
    os.makedirs(dst_subdir, exist_ok=True)
    img.save(os.path.join(dst_subdir, f"{item_idx:05d}.png"))


def run(
    data_dir: str,
    resolution: int,
    debug: bool,
    pydrive: bool,
    cmd_auth: bool,
    check_invalid_images: bool,
    ffhq_image_idxs_path: Optional[str] = None,
    **download_kwargs: Any,
) -> None:
    """Download and process images from the FFHQ dataset.

    Adapted from: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    Args:
        data_dir (str): Path to the directory where the downloaded files will be saved.
        resolution (int): The resolution of the output images.
        debug (bool): If True, only download a small subset of the images for
            debugging purposes.
        pydrive (bool): If True, use PyDrive to download files from Google Drive.
        cmd_auth (bool):  True if command-line authentication is desired,
            False otherwise.
        check_invalid_images (bool): If True, check for and remove invalid image
            files after downloading.
        ffhq_image_idxs_path (str, optional): Path to a text file containing a list of
            strings representing the FFHQ image IDs to download.
        **download_kwargs (Any): Any additional keyword arguments to pass to the
            download function.

    Returns:
        None
    """
    if pydrive:
        drive = pydrive_utils.create_drive_manager(cmd_auth)
    else:
        drive = None

    license_path = os.path.join(data_dir, "LICENSE.txt")
    json_spec["file_path"] = os.path.join(data_dir, str(json_spec["file_path"]))
    for k in license_specs.keys():
        license_specs[k]["file_path"] = os.path.join(
            data_dir, str(license_specs[k]["file_path"])
        )

    if not os.path.isfile(str(json_spec["file_path"])) or not os.path.isfile(
        license_path
    ):
        print("Downloading JSON metadata...")
        download_files(
            file_specs=[json_spec, license_specs["json"]],
            dst_dir=data_dir,
            drive=drive,
            **download_kwargs,
        )

    print("Parsing JSON metadata...")
    with open(str(json_spec["file_path"]), "rb") as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)

    # Image ids to download.
    ffhq_image_idxs = [str(x) for x in range(70000)]
    if ffhq_image_idxs_path:
        try:
            ffhq_image_idxs = np.loadtxt(ffhq_image_idxs_path, dtype=str)
        except FileNotFoundError:
            raise FileNotFoundError(f"{ffhq_image_idxs_path} not found.")

    # Update file paths.
    specs = []
    for idx, item in json_data.items():
        if idx in ffhq_image_idxs:
            spec = item["in_the_wild"]
            spec["file_path"] = os.path.join(data_dir, spec["file_path"])
            specs.append(spec)

    specs += [license_specs["wilds"]]

    if len(specs):
        output_size = resolution
        dst_dir = os.path.join(data_dir, f"ffhq{output_size}x{output_size}")
        np.random.shuffle(specs)
        if debug:
            specs = specs[:50]
        print("Downloading %d files..." % len(specs))
        download_files(
            file_specs=specs,
            dst_dir=dst_dir,
            output_size=output_size,
            check_invalid_images=check_invalid_images,
            drive=drive,
            **download_kwargs,
        )

    if os.path.isdir(os.path.join(data_dir, "in-the-wild-images")):
        shutil.rmtree(os.path.join(data_dir, "in-the-wild-images"))
