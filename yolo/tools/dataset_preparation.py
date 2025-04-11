import zipfile
from pathlib import Path
from typing import Optional

from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from yolo.config.config import DatasetConfig
from yolo.utils.logger import logger


def download_file(url, destination: Path):
    """
    Downloads a file from the specified URL to the destination path with progress logging.
    """
    import requests
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            "{task.completed}/{task.total} bytes",
            "•",
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"📥 Downloading {destination.name }...", total=total_size)
            with open(destination, "wb") as file:
                for data in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                    file.write(data)
                    progress.update(task, advance=len(data))
    logger.info(":white_check_mark: Download completed.")


def unzip_file(source: Path, destination: Path):
    """
    Extracts a ZIP file to the specified directory and removes the ZIP file after extraction.
    """
    logger.info(f"Unzipping {source.name}...")
    with zipfile.ZipFile(source, "r") as zip_ref:
        zip_ref.extractall(destination)
    source.unlink()
    logger.info(f"Removed {source}.")


def check_files(directory, expected_count=None):
    """
    Returns True if the number of files in the directory matches expected_count, False otherwise.
    """
    files = [f.name for f in Path(directory).iterdir() if f.is_file()]
    return len(files) == expected_count if expected_count is not None else bool(files)


def prepare_dataset(dataset_cfg: DatasetConfig, task: str):
    """
    Prepares dataset by downloading and unzipping if necessary.
    """
    # TODO: do EDA of dataset
    data_dir = Path(dataset_cfg.path)
    for data_type, settings in dataset_cfg.auto_download.items():
        base_url = settings["base_url"]
        for dataset_type, dataset_args in settings.items():
            if dataset_type != "annotations" and dataset_cfg.get(task, task) != dataset_type:
                continue
            file_name = f"{dataset_args.get('file_name', dataset_type)}.zip"
            url = f"{base_url}{file_name}"
            local_zip_path = data_dir / file_name
            extract_to = data_dir / data_type if data_type != "annotations" else data_dir
            final_place = extract_to / dataset_type

            final_place.mkdir(parents=True, exist_ok=True)
            if check_files(final_place, dataset_args.get("file_num")):
                logger.info(f":white_check_mark: Dataset {dataset_type: <12} already verified.")
                continue

            if not local_zip_path.exists():
                download_file(url, local_zip_path)
            unzip_file(local_zip_path, extract_to)

            if not check_files(final_place, dataset_args.get("file_num")):
                logger.error(f"Error verifying the {dataset_type} dataset after extraction.")


def prepare_weight(download_link: Optional[str] = None, weight_path: Path = Path("v9-c.pt")):
    weight_name = weight_path.name
    if download_link is None:
        download_link = "https://github.com/WongKinYiu/yolov9mit/releases/download/v1.0-alpha/"
    weight_link = f"{download_link}{weight_name}"

    if not weight_path.parent.is_dir():
        weight_path.parent.mkdir(parents=True, exist_ok=True)

    import requests
    if weight_path.exists():
        logger.info(f"Weight file '{weight_path}' already exists.")
    try:
        download_file(weight_link, weight_path)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to download the weight file: {e}")
