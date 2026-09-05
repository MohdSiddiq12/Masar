"""Module entry point for the congestion-model training workflow."""

from train_model import fetch_real_rows, main

__all__ = ["fetch_real_rows", "main"]


if __name__ == "__main__":
    main()