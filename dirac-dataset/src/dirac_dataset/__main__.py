#!/usr/bin/env python
"""
Main CLI entry point for dirac-dataset - now using 3-layer architecture

Examples
--------
python -m dirac_dataset gen-dataset --repos-file repos.json --pdfs-file pdfs.txt
python -m dirac_dataset load-db --repos-file repos.json --pdfs-file pdfs.txt --db-type milvus
"""

import typer
from dirac_dataset.presentation.gen_dataset import generate_dataset
from dirac_dataset.presentation.db_loader import load_db

app = typer.Typer(add_completion=False)

# Add commands directly
app.command("gen-dataset")(generate_dataset)
app.command("load-db")(load_db)


if __name__ == "__main__":
    app()
