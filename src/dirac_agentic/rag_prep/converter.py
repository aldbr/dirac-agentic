from __future__ import annotations
import re
from pathlib import Path
import fitz

from dirac_agentic.log import logger as logging

logger = logging.getChild(__name__)
_RST_RE = re.compile(r"^\s*\.\. .*::.*$", re.MULTILINE)


def rst_or_md_to_txt(path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".rst":
        text = _RST_RE.sub("", text)
    out = out_dir / f"{path.stem}.txt"
    out.write_text(text, encoding="utf-8")
    logger.info("Converted %s to %s", path, out)
    return out


def pdf_to_txt(pdf: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    text = "\n".join(p.get_text() for p in doc)
    out = out_dir / f"{pdf.stem}.txt"
    out.write_text(text, encoding="utf-8")
    logger.info("Extracted %s to %s", pdf, out)
    return out
