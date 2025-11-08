#!/usr/bin/env python
"""Convenience script to visualize the U-Net model architecture.

This script can be run directly from the project root:
    python visualize_model.py
    or
    uv run python visualize_model.py
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run visualization
from src.models.visualize import visualize_model_architecture

if __name__ == "__main__":
    visualize_model_architecture()