# Copyright (c) Sony AI Inc.
# All rights reserved.

import os
import re

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def create_drive_manager(cmd_auth: bool) -> GoogleDrive:
    """Creates a Google Drive manager and authorizes it using command-line or local
    webserver authentication.

    Adapted from: https://github.com/royorel/FFHQ-Aging-Dataset/blob/master/
    pydrive_utils.py

    Args:
        cmd_auth (bool): True if command-line authentication is desired,
        False otherwise.

    Returns:
        GoogleDrive: A Google Drive manager object.

    Raises:
        ValueError: If authorization fails and no valid credentials are found.
    """
    google_auth = GoogleAuth()

    if cmd_auth:
        google_auth.CommandLineAuth()
    else:
        google_auth.LocalWebserverAuth()

    google_auth.Authorize()
    print("Access authorized to the Google Drive API.")

    if not google_auth.credentials:
        raise ValueError("Authorization failed. No valid credentials were found.")

    return GoogleDrive(google_auth)


def extract_files_id(link: str) -> str:
    """Extracts the file ID from a Google Drive file link.

    Adapted from: https://github.com/royorel/FFHQ-Aging-Dataset/blob/master/
    pydrive_utils.py

    Args:
        link (str): A Google Drive file link.

    Returns:
        str: The file ID.

    Raises:
        ValueError: If the link is invalid and no file ID can be extracted.
    """
    match_obj = re.search(r"(?<=/d/|id=|rs/).+?(?=/|$)", link)
    if match_obj:
        file_id = match_obj.group(0)
        return file_id
    else:
        raise ValueError(f"Invalid link: {link}. Could not extract file ID.")


def pydrive_download(drive: GoogleDrive, link: str, save_path: str) -> None:
    """Downloads a file from Google Drive using the PyDrive library.

    Adapted from: https://github.com/royorel/FFHQ-Aging-Dataset/blob/master/
    pydrive_utils.py

    Args:
        drive (GoogleDrive): A Google Drive manager object.
        link (str): A Google Drive file link.
        save_path (str): The path to save the downloaded file.

    Returns:
        None

    Raises:
        Exception: If there is an error while downloading the file.
    """
    file_id = extract_files_id(link)
    if not file_id:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        pydrive_file = drive.CreateFile({"id": file_id})
        pydrive_file.GetContentFile(save_path)
    except Exception as error:
        raise Exception(f"Error while downloading file {file_id}: {error}")
