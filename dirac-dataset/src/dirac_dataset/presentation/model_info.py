#!/usr/bin/env python
"""Model information CLI - Presentation Layer.

Provides CLI tools for displaying embedding model information.
"""

from typing import Optional, Literal
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from dirac_dataset.business_logic.embedding_factory import (
    get_model_summary,
    get_model_info,
    search_models,
    list_models,
    get_recommended_model,
    get_default_model,
    ModelTier,
)

# Export the print function for other presentation modules
__all__ = ["print_model_summary", "app"]

console = Console()
app = typer.Typer(add_completion=False, help="Embedding model information utilities")


def print_model_summary() -> None:
    """Print a human-readable summary of available models."""
    summary = get_model_summary()
    stats = summary["statistics"]
    recommendations = summary["recommendations"]
    models_by_tier = summary["models_by_tier"]

    # Main statistics panel
    stats_content = (
        f"Total models: [bold]{stats['total_models']}[/bold]\n"
        f"CPU models: [green]{stats['cpu_models']}[/green]\n"
        f"GPU models: [yellow]{stats['gpu_models']}[/yellow]\n"
        f"Multi-GPU models: [red]{stats['multi_gpu_models']}[/red]\n"
        f"Dynamic models: [blue]{stats['is_dynamic']}[/blue]"
    )

    console.print(Panel(stats_content, title="📊 Model Statistics", expand=False))

    # Recommendations panel
    rec_content = f"Default: [bold]{recommendations['default']}[/bold]"
    if recommendations["cpu"]:
        rec_content += f"\nBest CPU: [green]{recommendations['cpu']}[/green]"
    if recommendations["gpu"]:
        rec_content += f"\nBest GPU: [yellow]{recommendations['gpu']}[/yellow]"

    console.print(Panel(rec_content, title="🎯 Recommendations", expand=False))

    # Models by tier
    for tier, models in models_by_tier.items():
        if not models:
            continue

        table = Table(title=f"{tier.upper()} Models ({len(models)} available)")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Description", style="white", max_width=50)
        table.add_column("Downloads", style="magenta", justify="right")
        table.add_column("Default", style="green", justify="center")

        # Show top 5 models per tier
        for model_data in models[:5]:
            default_mark = "🌟" if model_data["is_default"] else ""
            table.add_row(
                model_data["name"],
                model_data["info"].description,
                model_data["downloads_formatted"],
                default_mark,
            )

        if len(models) > 5:
            table.add_row("...", f"({len(models) - 5} more models)", "", "")

        console.print(table)


@app.command("summary")
def summary_command():
    """Display a comprehensive summary of available embedding models."""
    print_model_summary()


@app.command("info")
def info_command(
    model_name: str = typer.Argument(
        ..., help="Name of the model to get information for"
    ),
):
    """Get detailed information about a specific model."""
    model_info = get_model_info(model_name)

    if not model_info:
        console.print(f"[red]Model '{model_name}' not found.[/red]")

        # Suggest similar models
        suggestions = search_models(model_name)
        if suggestions:
            console.print("\n[yellow]Did you mean:[/yellow]")
            for suggestion in suggestions[:3]:
                console.print(f"  • {suggestion}")

        raise typer.Exit(1)

    # Create detailed info panel
    info_content = (
        f"[bold]Full ID:[/bold] {model_info.id}\n"
        f"[bold]Tier:[/bold] {model_info.tier}\n"
        f"[bold]Library:[/bold] {model_info.library_name}\n"
        f"[bold]Pipeline:[/bold] {model_info.pipeline_tag}\n"
        f"[bold]Downloads:[/bold] {model_info.downloads:,}\n"
        f"[bold]Default:[/bold] {'Yes' if model_info.default else 'No'}\n"
        f"[bold]CPU Friendly:[/bold] {'Yes' if model_info.is_cpu_friendly() else 'No'}\n"
        f"[bold]Requires GPU:[/bold] {'Yes' if model_info.requires_gpu() else 'No'}\n\n"
        f"[bold]Description:[/bold]\n{model_info.description}"
    )

    console.print(
        Panel(info_content, title=f"📋 {model_info.get_display_name()}", expand=False)
    )


@app.command("list")
def list_command(
    tier: Optional[str] = typer.Option(
        None, "--tier", "-t", help="Filter by hardware tier (cpu/gpu/multi-gpu)"
    ),
    sort_by: str = typer.Option(
        "downloads", "--sort", "-s", help="Sort by: downloads, name, default"
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of models to show"
    ),
):
    """List available models with flexible filtering and sorting."""
    # Validate tier
    valid_tiers = {"cpu", "gpu", "multi-gpu"}
    if tier and tier not in valid_tiers:
        console.print(
            f"[red]Invalid tier '{tier}'. Valid options: {', '.join(valid_tiers)}[/red]"
        )
        raise typer.Exit(1)

    # Validate sort_by
    valid_sorts = {"downloads", "name", "default"}
    if sort_by not in valid_sorts:
        console.print(
            f"[red]Invalid sort option '{sort_by}'. Valid options: {', '.join(valid_sorts)}[/red]"
        )
        raise typer.Exit(1)

    # Type cast for mypy - these are validated above
    tier_typed: Optional[ModelTier] = tier  # type: ignore[assignment]
    sort_by_typed: Literal["downloads", "name", "default"] = sort_by  # type: ignore[assignment]
    models = list_models(tier=tier_typed, sort_by=sort_by_typed)

    if not models:
        tier_str = f" for tier '{tier}'" if tier else ""
        console.print(f"[yellow]No models found{tier_str}.[/yellow]")
        return

    # Create table
    table = Table()
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Tier", style="yellow")
    table.add_column("Downloads", style="magenta", justify="right")
    table.add_column("Default", style="green", justify="center")

    # Show limited results
    for model_name in models[:limit]:
        model_info = get_model_info(model_name)
        if model_info:
            default_mark = "🌟" if model_info.default else ""
            table.add_row(
                model_name, model_info.tier, f"{model_info.downloads:,}", default_mark
            )

    title = "Models"
    if tier:
        title += f" ({tier.upper()} tier)"
    if len(models) > limit:
        title += f" (showing {limit} of {len(models)})"

    table.title = title
    console.print(table)


@app.command("search")
def search_command(
    query: str = typer.Argument(
        ..., help="Search term to match against model names and descriptions"
    ),
    limit: int = typer.Option(
        5, "--limit", "-l", help="Maximum number of results to show"
    ),
):
    """Search for models by name or description."""
    results = search_models(query)

    if not results:
        console.print(f"[yellow]No models found matching '{query}'.[/yellow]")
        return

    console.print(f"[green]Found {len(results)} model(s) matching '{query}':[/green]\n")

    for i, model_name in enumerate(results[:limit], 1):
        model_info = get_model_info(model_name)
        if model_info:
            console.print(f"{i}. [cyan]{model_name}[/cyan] ({model_info.tier})")
            console.print(f"   {model_info.description}")
            if model_info.default:
                console.print("   🌟 Default model")
            console.print()

    if len(results) > limit:
        console.print(f"[dim]... and {len(results) - limit} more results[/dim]")


@app.command("recommend")
def recommend_command(
    tier: Optional[str] = typer.Option(
        None, "--tier", "-t", help="Hardware tier (cpu/gpu/multi-gpu)"
    ),
):
    """Get recommended models for different use cases."""
    valid_tiers = {"cpu", "gpu", "multi-gpu"}
    if tier and tier not in valid_tiers:
        console.print(
            f"[red]Invalid tier '{tier}'. Valid options: {', '.join(valid_tiers)}[/red]"
        )
        raise typer.Exit(1)

    if tier:
        tier_typed: ModelTier = tier  # type: ignore[assignment]
        recommended = get_recommended_model(tier_typed)
        if recommended:
            model_info = get_model_info(recommended)
            console.print(
                f"[green]Recommended {tier.upper()} model:[/green] [cyan]{recommended}[/cyan]"
            )
            if model_info:
                console.print(f"Description: {model_info.description}")
                console.print(f"Downloads: {model_info.downloads:,}")
        else:
            console.print(f"[yellow]No models found for tier '{tier}'.[/yellow]")
    else:
        # Show recommendations for all tiers
        default = get_default_model()
        cpu_rec = get_recommended_model("cpu")
        gpu_rec = get_recommended_model("gpu")
        multi_gpu_rec = get_recommended_model("multi-gpu")

        console.print("[green]Model Recommendations:[/green]\n")
        console.print(f"🎯 [bold]Default:[/bold] {default}")

        if cpu_rec:
            console.print(f"💻 [bold]Best CPU:[/bold] {cpu_rec}")
        if gpu_rec:
            console.print(f"🚀 [bold]Best GPU:[/bold] {gpu_rec}")
        if multi_gpu_rec:
            console.print(f"⚡ [bold]Best Multi-GPU:[/bold] {multi_gpu_rec}")


if __name__ == "__main__":
    app()
