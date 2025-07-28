# Private Assistant Comms Satellite

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)
[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Owner: stkr22

**READ THE AGENTS.md**

The satellite is an open source library to work with the private assistant oecosystem built to run on edge devices. It allows the other components to interact speech based with the user and listen for user keywords to activate.

## Installation

### Basic Installation
```bash
uv sync
```

### Audio Support (for running on devices)

For audio processing capabilities, install system dependencies and audio group:

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install -y libasound2-dev libportaudio2 libportaudiocpp0 portaudio19-dev

# Install audio dependencies
uv sync --group audio
```

**Note**: Audio dependencies are optional and not required for testing or development without hardware audio.
