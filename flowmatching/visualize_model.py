#!/usr/bin/env python
"""Convenience script to visualize the U-Net model architecture.

Run this from the project root as a module:

    python -m flowmatching.visualize_model
    or
    uv run python -m flowmatching.visualize_model
"""

# Import and run visualization from the flowmatching package
from flowmatching.models.visualize import visualize_model_architecture

if __name__ == "__main__":
    visualize_model_architecture()