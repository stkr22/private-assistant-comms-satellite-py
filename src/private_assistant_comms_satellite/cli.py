"""CLI interface for Private Assistant Communications Satellite."""

import pathlib
from typing import Annotated

import typer
import yaml
from private_assistant_commons import skill_logger
from rich.console import Console

from private_assistant_comms_satellite import __version__
from private_assistant_comms_satellite.main import start_satellite
from private_assistant_comms_satellite.utils.config import Config

# AIDEV-NOTE: Rich console integration provides consistent styling with private-commons logging
console = Console()
app = typer.Typer(
    rich_markup_mode="rich",
    help="[bold blue]Private Assistant Communications Satellite[/bold blue] - Edge device voice interaction",
)


@app.command()
def start(
    config_path: Annotated[
        pathlib.Path,
        typer.Argument(
            envvar="PRIVATE_ASSISTANT_API_CONFIG_PATH",
            help="Path to YAML configuration file",
        ),
    ] = pathlib.Path("local_config.yaml"),
) -> None:
    """Start the Private Assistant Communications Satellite.

    Provides edge device voice interaction with wake word detection,
    speech processing, and MQTT communication.
    """
    try:
        console.print("[green]Starting Private Assistant Communications Satellite[/green]")
        console.print(f"[dim]Using config: {config_path}[/dim]")

        # Initialize logger early for consistent logging
        logger = skill_logger.SkillLogger.get_logger("Private Assistant Comms Satellite")
        logger.info("Starting satellite application from CLI")

        start_satellite(config_path)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        logger.info("Satellite application interrupted by user")
    except FileNotFoundError:
        console.print(f"[red]Error: Configuration file not found: {config_path}[/red]")
        console.print("[dim]Please check the file path or set PRIVATE_ASSISTANT_API_CONFIG_PATH[/dim]")
        raise typer.Exit(1) from None
    except ImportError as e:
        if "sounddevice" in str(e).lower():
            console.print("[red]Error: sounddevice not available[/red]")
            console.print("[dim]System audio dependencies may be required (see README.md)[/dim]")
        else:
            console.print(f"[red]Import Error: {e}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.error("Unexpected error in satellite application: %s", e, exc_info=True)
        raise typer.Exit(1) from e


@app.command()
def version() -> None:
    """Show the version of the Private Assistant Communications Satellite."""
    console.print(f"[blue]Private Assistant Communications Satellite[/blue] version [green]{__version__}[/green]")


@app.command()
def config_template() -> None:
    """Generate a template configuration file with default values."""
    # Create default config instance and serialize to YAML
    default_config = Config()
    config_dict = default_config.model_dump()

    # Add helpful comments for key sections
    config_template = f"""# Private Assistant Communications Satellite Configuration
# Generated from default Config class values

{yaml.dump(config_dict, default_flow_style=False, sort_keys=True)}
"""

    config_path = pathlib.Path("local_config.yaml")

    if config_path.exists() and not typer.confirm(f"Configuration file {config_path} already exists. Overwrite?"):
        console.print("[yellow]Configuration template generation cancelled[/yellow]")
        return

    config_path.write_text(config_template)
    console.print(f"[green]Configuration template created: {config_path}[/green]")
    console.print("[dim]Edit this file to customize your satellite settings[/dim]")


if __name__ == "__main__":
    app()
