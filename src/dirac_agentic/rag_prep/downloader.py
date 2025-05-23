from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import requests

from dirac_agentic.log import logger as logging

logger = logging.getChild(__name__)


def clone_repo(url: str, dest_root: Path, depth: int = 1) -> Path:
    """Clone a git repository to the specified destination directory."""
    name = url.rsplit("/", 1)[-1].removesuffix(".git")
    dest = dest_root / name
    if dest.exists():
        shutil.rmtree(dest)
    logger.info("Cloning %s to %s", url, dest)
    subprocess.run(
        ["git", "clone", "--depth", str(depth), url, str(dest)],
        check=True,
    )
    return dest


def download_pdf(url: str, out_dir: Path) -> Path:
    """Download a PDF from the given URL and save it to the specified directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    file = out_dir / (url.rsplit("/", 1)[-1] or "download.pdf")

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

    with open(file, "wb") as fh:
        for chunk in r.iter_content(chunk_size=8192):
            fh.write(chunk)
    logger.info("Saved %s", file)
    return file
