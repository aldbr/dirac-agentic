from __future__ import annotations
from pathlib import Path
import requests

from dirac_dataset.log import logger as logging

logger = logging.getChild(__name__)


def download_pdf(url: str, out_dir: Path, local_name: str) -> Path | None:
    """Download a PDF from the given URL and save it to the specified directory.
    Only saves if content-type is PDF and file starts with %PDF.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    file = out_dir / local_name

    logger.debug("GET %s", url)

    r = requests.get(
        url,
        stream=True,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        },
    )
    r.raise_for_status()

    # Check content-type header
    content_type = r.headers.get("content-type", "").lower()
    if "pdf" not in content_type:
        logger.warning(
            "URL %s does not appear to be a PDF (content-type: %s), skipping.",
            url,
            content_type,
        )
        return None

    with open(file, "wb") as fh:
        for chunk in r.iter_content(chunk_size=8192):
            fh.write(chunk)
    # Check PDF magic number
    try:
        with open(file, "rb") as fh:
            magic = fh.read(4)
        if magic != b"%PDF":
            logger.warning(
                "File %s does not start with %%PDF magic number, deleting. (URL: %s)",
                file,
                url,
            )
            file.unlink(missing_ok=True)
            return None
    except Exception as e:
        logger.warning("Error checking PDF magic number for %s: %s", file, e)
        file.unlink(missing_ok=True)
        return None
    logger.info("Saved %s", file)
    return file
