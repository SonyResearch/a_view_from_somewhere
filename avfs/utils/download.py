# Copyright (c) Sony AI Inc.
# All rights reserved.

import os
import urllib.request
import shutil
import bz2
import zipfile
import requests

from typing import Optional


def download_compressed_file(
    save_path: str, url: str, extract_path: Optional[str] = None
) -> None:
    """Download a compressed file from the given URL and save it to the specified path.

    Args:
        save_path (str): The directory to save the downloaded file in.
        url (str): The URL to download the file from.
        extract_path (str, optional): The directory to save the downloaded file in,
            which overrides the extract_path.

    Returns:
        None
    """
    os.makedirs(save_path, exist_ok=True)
    file_name = url.split("/")[-1]
    file_path = os.path.join(save_path, file_name)

    if extract_path is None:
        extract_path = file_path[:-4]

    if os.path.isfile(file_path):
        print(f"{file_name} already exists at {save_path}.")
    else:
        with urllib.request.urlopen(url) as response, open(file_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)

    if file_name.endswith(".bz2"):
        with open(extract_path, "wb") as out_file, bz2.BZ2File(file_path) as bz2_file:
            shutil.copyfileobj(bz2_file, out_file)

        print(f"Extracted {file_name} to {extract_path}.")

    if file_name.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"Extracted {file_name} to {extract_path}.")


def download_from_url(
    download_url: str, save_filepath: str, chunk_size: int = 8192
) -> None:
    """
    Downloads a file from the given URL and saves it to the specified filepath.

    Args:
        download_url (str): The URL of the file to download.
        save_filepath (str): The filepath to save the downloaded file to.
        chunk_size (int): The size of each chunk to download at a time.
            Default is 8192.

    Returns:
        None
    """
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(save_filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)


def agree_to_license(url: str) -> bool:
    """Prompt the user to agree to license from a given URL.

    Args:
        url (str): The URL of the license to display.

    Returns:
        agreed (bool): True if the user agrees to the license, False otherwise.
    """
    agreed = False
    while not agreed:
        response = input(
            f"Do you agree to the license: {url}?" f"\nEnter 'yes' or 'no': "
        )
        if response.lower() == "yes":
            print("You have agreed to the license.")
            agreed = True
            return agreed
        elif response.lower() == "no":
            print("You cannot proceed without agreeing to the license.")
            agreed = False
            return agreed
        else:
            print("Invalid response. Please enter 'yes' or 'no'.")
    return agreed
