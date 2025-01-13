import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, dst_path: Path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(dst_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {dst_path.name}",
        ) as progress:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.update(len(chunk))


def unzip_archive(src_path: str, dst_folder: str):
    with zipfile.ZipFile(src_path, "r") as zip_ref:
        zip_ref.extractall(dst_folder)
