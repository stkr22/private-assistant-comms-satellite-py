"""The satellite is an open source library to work with the private assistant oecosystem built to run on edge
devices. It allows the other components to interact speech based with the user and listen for user keywords
to activate."""

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installs
    __version__ = "dev"

__all__ = ["__version__"]
