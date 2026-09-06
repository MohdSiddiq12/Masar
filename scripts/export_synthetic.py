"""Generate local synthetic traffic data as an Excel workbook."""

import argparse
from pathlib import Path

from train_model import generate_synthetic_rows


def export_synthetic_excel(output: str, rows: int = 6000, seed: int = 42) -> Path:
    dataframe = generate_synthetic_rows(rows, seed=seed)
    if set(dataframe["source"].unique()) != {"synthetic"}:
        raise ValueError("Export refused: data is not exclusively synthetic.")

    # Excel stores datetimes without timezone metadata; keep the UTC instant.
    dataframe = dataframe.copy()
    dataframe["timestamp"] = dataframe["timestamp"].dt.tz_localize(None)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_excel(output_path, index=False, sheet_name="synthetic_traffic")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/synthetic_traffic.xlsx")
    args = parser.parse_args()
    output_path = export_synthetic_excel(args.output, args.rows, args.seed)
    print(f"Exported {args.rows} synthetic rows to {output_path}")
    print("Supabase was not accessed; this file contains synthetic data only.")


if __name__ == "__main__":
    main()