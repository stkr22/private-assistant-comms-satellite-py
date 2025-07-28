"""CLI interface for Private Assistant Communications Satellite."""

import pathlib
from typing import Annotated

import typer
from private_assistant_commons import skill_logger
from rich.console import Console

from private_assistant_comms_satellite import __version__
from private_assistant_comms_satellite.main import start_satellite

# AIDEV-NOTE: Rich console integration provides consistent styling with private-commons logging
console = Console()
app = typer.Typer(
    rich_markup_mode="rich",
    help="[bold blue]Private Assistant Communications Satellite[/bold blue] - Edge device voice interaction",
)


@app.command()
def main(
    config_path: Annotated[
        pathlib.Path, 
        typer.Argument(
            envvar="PRIVATE_ASSISTANT_API_CONFIG_PATH",
            help="Path to YAML configuration file",
        )
    ] = pathlib.Path("local_config.yaml"),
) -> None:
    """Start the Private Assistant Communications Satellite.
    
    The satellite provides edge device voice interaction capabilities including:
    - Wake word detection using OpenWakeWord
    - Speech-to-text transcription via API
    - Text-to-speech synthesis via API  
    - MQTT communication with the assistant ecosystem
    - Voice activity detection using Silero VAD
    
    Configuration can be provided via:
    - Command line argument: --config-path path/to/config.yaml
    - Environment variable: PRIVATE_ASSISTANT_API_CONFIG_PATH
    - Default: local_config.yaml in current directory
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
        if "pyaudio" in str(e).lower():
            console.print("[red]Error: PyAudio not available[/red]")
            console.print("[dim]Install audio dependencies with: uv sync --group audio[/dim]")
            console.print("[dim]Or install system dependencies first (see README.md)[/dim]")
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
    """Generate a template configuration file.
    
    Creates a local_config.yaml file with all available configuration options
    and their default values for easy customization.
    """
    config_template = """# Private Assistant Communications Satellite Configuration
# This file contains all available configuration options with their default values

# Wake word detection settings
wakework_detection_threshold: 0.6  # Confidence threshold for wake word detection (0.0-1.0)
path_or_name_wakeword_model: "./hey_nova.onnx"  # Path to wake word model file
name_wakeword_model: "hey_nova"  # Name of the wake word model

# API endpoints for speech processing
speech_transcription_api: "http://localhost:8000/transcribe"  # STT API endpoint
speech_transcription_api_token: null  # Optional API token for STT service
speech_synthesis_api: "http://localhost:8080/synthesizeSpeech"  # TTS API endpoint
speech_synthesis_api_token: null  # Optional API token for TTS service

# Device and room identification
client_id: "default_hostname"  # Unique identifier for this satellite (defaults to hostname)
room: "livingroom"  # Room name for this satellite device

# Audio device configuration
output_device_index: 1  # Audio output device index
input_device_index: 1   # Audio input device index

# Audio processing settings
max_command_input_seconds: 15     # Maximum recording time for voice commands
max_length_speech_pause: 1.0      # Maximum pause duration before stopping recording
samplerate: 16000                 # Audio sample rate in Hz
chunk_size: 512                   # Audio processing chunk size
chunk_size_ow: 1280              # OpenWakeWord chunk size

# Voice Activity Detection (VAD)
vad_threshold: 0.6  # VAD confidence threshold (0.0-1.0, 1.0 is speech)
vad_trigger: 1      # Number of chunks to cross threshold before activation

# MQTT broker configuration
mqtt_server_host: "localhost"  # MQTT broker hostname
mqtt_server_port: 1883         # MQTT broker port

# MQTT topic configuration (optional overrides)
broadcast_topic: "assistant/comms_bridge/broadcast"  # Broadcast topic for system messages
base_topic_overwrite: null    # Override for base topic (defaults to assistant/comms_bridge/all/{client_id})
input_topic_overwrite: null   # Override for input topic (defaults to {base_topic}/input)
output_topic_overwrite: null  # Override for output topic (defaults to {base_topic}/output)

# Audio file paths
start_listening_path: "sounds/start_listening.wav"  # Sound played when starting to listen
stop_listening_path: "sounds/stop_listening.wav"    # Sound played when stopping listening
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