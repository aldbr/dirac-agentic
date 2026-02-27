"""CLI entry point for dirac-dataset.

Usage:
    python -m dirac_dataset gen-dataset --repos-file repos.json --pdfs-file pdfs.json --out ./my_dataset
    python -m dirac_dataset push-to-hub ./my_dataset --repo-id myorg/dirac-docs
"""

import typer

from dirac_dataset.cli import gen_dataset, push_to_hub

app = typer.Typer(add_completion=False)

app.command("gen-dataset")(gen_dataset)
app.command("push-to-hub")(push_to_hub)


if __name__ == "__main__":
    app()
