"""Thin CLI entrypoint for Xenium-native input generation."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xenium_input_generation import main


if __name__ == "__main__":
    main()
