# Copyright (c) Sony AI Inc.
# All rights reserved.

import argparse
import os

from avfs.utils.download import (
    download_compressed_file,
    agree_to_license,
    download_from_url,
)
from avfs.utils.download_ffhq import run
from avfs.build_avfs import avfs_model_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avfs_data", action="store_true", help="Download AVFS dataset."
    )
    parser.add_argument(
        "--ffhq_data", action="store_true", help="Download FFHQ/FFHQ-Aging dataset."
    )
    parser.add_argument(
        "--avfs_models", action="store_true", help="Download AVFS pretrained models."
    )
    parser.add_argument(
        "--num_processes", type=int, default=99, help="Number of processes."
    )
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), "data")
    model_path = os.path.join(os.path.dirname(__file__), "pretrained_models")

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    download_compressed_file(
        save_path=model_path,
        url="http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    )

    if args.avfs_data or args.avfs_models:
        print("Downloading AVFS data and/or models.")
        url = "https://github.com/SonyResearch/a_view_from_somewhere/LICENSE"
        agreed = agree_to_license(url=url)
        if not agreed:
            exit()

        if args.avfs_data:
            download_compressed_file(
                save_path=data_path,
                url="https://zenodo.org/record/7878655/files/avfs-dataset-v1.zip",
                extract_path=data_path,
            )

        if args.avfs_models:
            for k in list(avfs_model_registry.keys()):
                model_url = avfs_model_registry[k]["zenodo_link"]
                save_filepath = os.path.join(model_path, os.path.basename(model_url))
                if not os.path.isfile(save_filepath):
                    download_from_url(
                        download_url=model_url, save_filepath=save_filepath
                    )

    if args.ffhq_data:
        print("Downloading and preprocessing FFHQ data.")
        url = "https://github.com/NVlabs/ffhq-dataset/blob/master/LICENSE.txt"
        agreed = agree_to_license(url=url)
        if not agreed:
            exit()

        client_secrets_file = os.path.join(
            os.path.dirname(__file__), "client_secrets.json"
        )
        if not os.path.isfile(client_secrets_file):
            print(f"Cannot find: {client_secrets_file}")
            exit()

        print("Downloading FFHQ data.")
        run(
            data_dir=os.path.join(data_path, "ffhq"),
            pydrive=True,
            cmd_auth=True,
            resolution=160,
            check_invalid_images=True,
            debug=False,
            ffhq_image_idxs_path=os.path.join(
                data_path, "ffhq_image_ids_to_download.txt"
            ),
        )


if __name__ == "__main__":
    main()
